
#include <vector_types.h>

#include <optix_device.h>

#include <cuda_runtime.h>

#include "OptiX7Craft.h"
#include "helpers.h"
#include "random.h"


#ifndef M_PI_4f
#define M_PI_4f     0.785398163397448309616f
#endif

__inline__ __device__ float3 tonemap(const float3 in)
{
    // hard coded exposure for sun/sky
    const float exposure = 1.0f / 30.0f;
    float3 x = exposure * in;

    // "filmic" map from a GDC talk by John Hable.  This includes 1/gamma.
    x = fmaxf(x - make_float3(0.004f), make_float3(0.0f));
    float3 ret = (x * (6.2f * x + make_float3(.5f))) / (x * (6.2f * x + make_float3(1.7f)) + make_float3(0.06f));

    return ret;
}

extern "C" {
__constant__ Params params;
}

static __device__ __inline__ float3 schlick(float nDi, const float3& rgb)
{
    float r = fresnel_schlick(nDi, 5, rgb.x, 1);
    float g = fresnel_schlick(nDi, 5, rgb.y, 1);
    float b = fresnel_schlick(nDi, 5, rgb.z, 1);
    return make_float3(r, g, b);
}

// light source samples
static __device__ __inline__ void cosine_sample_hemisphere(const float u1, const float u2, float3& p)
{
    // Uniformly sample disk.
    const float r = sqrtf(u1);
    const float phi = 2.0f * M_PIf * u2;
    p.x = r * cosf(phi);
    p.y = r * sinf(phi);

    // Project up to hemisphere.
    p.z = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.y * p.y));
}

static __device__ __inline__ float2 square_to_disk(const float2& sample)
{
    float phi, r;

    const float a = 2.0f * sample.x - 1.0f;
    const float b = 2.0f * sample.y - 1.0f;

    if (a > -b)
    {
        if (a > b)
        {
            r = a;
            phi = (float)M_PI_4f * (b / a);
        }
        else
        {
            r = b;
            phi = (float)M_PI_4f * (2.0f - (a / b));
        }
    }
    else
    {
        if (a < b)
        {
            r = -a;
            phi = (float)M_PI_4f * (4.0f + (b / a));
        }
        else
        {
            r = -b;
            phi = (b) ? (float)M_PI_4f * (6.0f - (a / b)) : 0.0f;
        }
    }

    return make_float2(r * cosf(phi), r * sinf(phi));
}


static __device__ __inline__ OcclusionPRD getOcclusionPRD()
{
    OcclusionPRD prd;
    prd.attenuation.x = int_as_float( optixGetPayload_0() );
    prd.attenuation.y = int_as_float( optixGetPayload_1() );
    prd.attenuation.z = int_as_float( optixGetPayload_2() );
    return prd;
}

static __device__ __inline__ void setOcclusionPRD( const OcclusionPRD &prd )
{
    optixSetPayload_0( float_as_int(prd.attenuation.x) );
    optixSetPayload_1( float_as_int(prd.attenuation.y) );
    optixSetPayload_2( float_as_int(prd.attenuation.z) );
}

static __forceinline__ __device__ float3 traceSun(
    float3                 origin,
    float3                 direction,
    int                    depth,
    float                  importance,
    float3                 attenuation
)
{
    SunPRD sun_prd;
    sun_prd.depth = depth;
    sun_prd.importance = importance;
    sun_prd.attenuation = attenuation;
    unsigned int u0, u1;
    packPointer(&sun_prd, u0, u1);
    optixTrace(
        params.handle,
        origin,
        direction,
        params.scene_epsilon,
        1e16f,
        0.0f,
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE,
        RAY_TYPE_RADIANCE,
        RAY_TYPE_COUNT,
        RAY_TYPE_RADIANCE,
        u0,
        u1);

    return sun_prd.radiance * sun_prd.attenuation ;
}

static __device__ void phongShadowed()
{
    // this material is opaque, so it fully attenuates all shadow rays
    OcclusionPRD prd;
    prd.attenuation = make_float3(0.f);
    setOcclusionPRD(prd);
}

static __device__ void phongShade( float3 p_Kd,
                            float3 p_Ka,
                            float3 p_Ks,
                            float3 p_Kr,
                            float  p_phong_exp,
                            float3 p_normal)
{
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();
    const float  ray_t    = optixGetRayTmax();

    SunPRD* sun_prd = getPRD<SunPRD>();

    if (sun_prd->countEmitted)
        sun_prd->emitted = make_float3(0.0f);//物体自身不发光
    else
        sun_prd->emitted = make_float3(0.0f);
    sun_prd->attenuation *= p_Kd;
    sun_prd->countEmitted = false;

    float3 hit_point = ray_orig + ray_t * ray_dir;

    // ambient contribution

    float3 result = p_Ka * params.ambient_light_color * p_Kd;
    
    // compute sun light
    DirectionalLight sun = params.sun;

    const float z1 = rnd(sun_prd->seed);
    const float z2 = rnd(sun_prd->seed);

    const int numLightSamples = params.num_lights_sample;
    for (int lightSampleID = 0; lightSampleID < numLightSamples; lightSampleID++) {
        const float3 light_center = hit_point + sun.direction*10.0f;
        const float r1 = rnd(sun_prd->seed);
        const float r2 = rnd(sun_prd->seed);
        const float2 disk_sample = square_to_disk(make_float2(r1, r2));
        const float3 jittered_pos = light_center + sun.radius * disk_sample.x * sun.v0 + sun.radius * disk_sample.y * sun.v1;
        float3 L = normalize(jittered_pos - hit_point);
        const float NdotL = dot(p_normal, L);
        //float weight = 0.0f;
        if (NdotL > 0.0f) {
            OcclusionPRD shadow_prd;
            shadow_prd.attenuation = make_float3(1.0f);

            optixTrace(
                params.handle,
                hit_point + p_normal * params.scene_epsilon,
                L,
                0.01f,
                1e16, // sun light source shoots from infinite distance
                0.0f,
                OptixVisibilityMask(1),
                OPTIX_RAY_FLAG_NONE,
                RAY_TYPE_OCCLUSION,
                RAY_TYPE_COUNT,
                RAY_TYPE_OCCLUSION,
                float3_as_args(shadow_prd.attenuation));

            float3 light_attenuation = shadow_prd.attenuation;

            if (fmaxf(light_attenuation) > 0.0f)
            {
                const float solid_angle = sun.radius * sun.radius * M_PIf;
                float3 Lc = light_attenuation * tonemap(sun.color * solid_angle);
                result += p_Kd * NdotL * Lc / numLightSamples;
                float3 H = normalize(L - ray_dir);
                float nDh = dot(p_normal, H);
                if (nDh > 0)
                {
                    float power = pow(nDh, p_phong_exp);
                    result += p_Ks * power * Lc/ numLightSamples;
                }
            }
        }
    }
    // reflection
    if (fmaxf(p_Kr) > 0)
    {
        // ray tree attenuation
        float new_importance = sun_prd->importance * luminance(p_Kr);
        int new_depth = sun_prd->depth + 1;

        // reflection ray
        // compare new_depth to max_depth - 1 to leave room for a potential shadow ray trace
        if (new_importance >= 0.01f && new_depth <= params.max_depth - 1)
        {
            float3 R = reflect(ray_dir, p_normal);
            result += p_Kr * traceSun(hit_point+p_normal * params.scene_epsilon, R, new_depth, new_importance, sun_prd->attenuation);
        }
    }
    
    
    // compute Point light
    for (int light_id = 0; light_id < params.point_light_sum; light_id++) {
        float Ldist = length(params.point_light[light_id].pos - hit_point);
        float light_fade = 1.0f;
        if (Ldist > 1.0f) {
            light_fade = 1.0f / (pow(Ldist, 1.2f));
        }
        for (int lightSampleID = 0; lightSampleID < numLightSamples; lightSampleID++) {
            float3 du = make_float3(1.0f, 0.0f, 0.0f)*0.5;
            float3 dv = make_float3(0.0f, 0.0f, 1.0f)*0.5;
            const float r1 = rnd(sun_prd->seed);
            const float r2 = rnd(sun_prd->seed);
            BasicLight light = params.point_light[light_id];
            const float3 jittered_pos = light.pos + r1 * du + r2 * dv;
            float Ldist = length(jittered_pos - hit_point);
            float3 L = normalize(jittered_pos - hit_point);
            float nDl = dot(p_normal, L);
            // cast shadow ray
            if (nDl > 0.0f)
            {
                OcclusionPRD shadow_prd;
                shadow_prd.attenuation = make_float3(1.0f);

                optixTrace(
                    params.handle,
                    hit_point + p_normal * params.scene_epsilon,
                    L,
                    0.01f,
                    Ldist - 0.01f,
                    0.0f,
                    OptixVisibilityMask(1),
                    OPTIX_RAY_FLAG_NONE,
                    RAY_TYPE_OCCLUSION,
                    RAY_TYPE_COUNT,
                    RAY_TYPE_OCCLUSION,
                    float3_as_args(shadow_prd.attenuation));

                float3 light_attenuation = shadow_prd.attenuation;

                // If not completely shadowed, light the hit point
                if (fmaxf(light_attenuation) > 0.0f)
                {
                    float3 Lc = light.color * light_attenuation;

                    result += p_Kd * nDl * Lc * light_fade;

                    float3 H = normalize(L - ray_dir);
                    float nDh = dot(p_normal, H);
                    if (nDh > 0)
                    {
                        float power = pow(nDh, p_phong_exp);
                        result += p_Ks * power * Lc * light_fade;
                    }

                }
            }

            // ray tree attenuation
            float new_importance = sun_prd->importance * luminance(p_Kr);
            int new_depth = sun_prd->depth + 1;

            // reflection ray
            // compare new_depth to max_depth - 1 to leave room for a potential shadow ray trace
            if (new_importance >= 0.01f && new_depth <= params.max_depth - 1)
            {
                float3 R = reflect(ray_dir, p_normal);

                result += p_Kr * traceSun(hit_point, R, new_depth, new_importance, sun_prd->attenuation )*light_fade;
            }
        }
    }
    
  
    sun_prd->radiance = result;
    unsigned int u0, u1;
    packPointer(&sun_prd, u0, u1);
    optixSetPayload_0(u0);
    optixSetPayload_1(u1);
}


extern "C" __global__ void __closesthit__checker_radiance()
{
    const HitGroupData* sbt_data = (HitGroupData*) optixGetSbtDataPointer();
    const CheckerPhong &checker = sbt_data->shading.checker;

    float3 Kd, Ka, Ks, Kr;
    float  phong_exp;

    float2 texcoord = make_float2(
        int_as_float( optixGetAttribute_3() ),
        int_as_float( optixGetAttribute_4() ) );
    float2 t  = texcoord * checker.inv_checker_size;
    t.x = floorf(t.x);
    t.y = floorf(t.y);

    int which_check = ( static_cast<int>( t.x ) +
                        static_cast<int>( t.y ) ) & 1;

    if ( which_check )
    {
        Kd = checker.Kd1;
        Ka = checker.Ka1;
        Ks = checker.Ks1;
        Kr = checker.Kr1;
        phong_exp = checker.phong_exp1;
    } else
    {
        Kd = checker.Kd2;
        Ka = checker.Ka2;
        Ks = checker.Ks2;
        Kr = checker.Kr2;
        phong_exp = checker.phong_exp2;
    }

    float3 object_normal = make_float3(
        int_as_float( optixGetAttribute_0() ),
        int_as_float( optixGetAttribute_1() ),
        int_as_float( optixGetAttribute_2() ));
    float3 world_normal = normalize( optixTransformNormalFromObjectToWorldSpace(object_normal) );
    float3 ffnormal  = faceforward( world_normal, -optixGetWorldRayDirection(), world_normal );
    phongShade( Kd, Ka, Ks, Kr, phong_exp, ffnormal );
}

// object with textures
extern "C" __global__ void __closesthit__texture_radiance()
{
    const HitGroupData* sbt_data = (HitGroupData*)optixGetSbtDataPointer();
    Phong phong = sbt_data->shading.metal;

    float3 geometry_normal = make_float3(
        int_as_float(optixGetAttribute_0()),
        int_as_float(optixGetAttribute_1()),
        int_as_float(optixGetAttribute_2()));
    
    float3 shade_normal = geometry_normal;
    // texture coordinate
    float2 coord = make_float2(
        int_as_float(optixGetAttribute_3()),
        int_as_float(optixGetAttribute_4()));

    //which cube face the hitpoint in
    cube_face face = cube_face(optixGetAttribute_5());
    if (sbt_data->has_normal) {
        shade_normal = make_float3(tex2D<float4>(sbt_data->normal_map, coord.x, coord.y)) * 2 - 1.0f;
        if (face == y_up || face == y_down) {
            shade_normal = make_float3(shade_normal.y, shade_normal.z, shade_normal.x);
        }
        else if (face ==x_up || face ==x_down)
            shade_normal = make_float3(shade_normal.z, shade_normal.x, shade_normal.y);
        else {
            shade_normal = make_float3(shade_normal.x, shade_normal.y, shade_normal.z);
        }
    }
    if (sbt_data->has_diffuse) {
        if(face == y_up)
            phong.Kd = make_float3(tex2D<float4>(sbt_data->diffuse_map_y_up, coord.x, coord.y));
        else if (face == y_down)
            phong.Kd = make_float3(tex2D<float4>(sbt_data->diffuse_map_y_down, coord.x, coord.y));
        if (face == x_up)
            phong.Kd = make_float3(tex2D<float4>(sbt_data->diffuse_map_x_up, coord.x, coord.y));
        else if (face == x_down)
            phong.Kd = make_float3(tex2D<float4>(sbt_data->diffuse_map_x_down, coord.x, coord.y));
        if (face == z_up)
            phong.Kd = make_float3(tex2D<float4>(sbt_data->diffuse_map_z_up, coord.x, coord.y));
        else if (face == z_down)
            phong.Kd = make_float3(tex2D<float4>(sbt_data->diffuse_map_z_down, coord.x, coord.y));
    }
    if (sbt_data->has_roughness) {
        phong.phong_exp = pow(128,(tex2D<float4>(sbt_data->roughness_map, coord.x, coord.y).x));
    }
    float3 world_shade_normal = normalize(optixTransformNormalFromObjectToWorldSpace(shade_normal));
    float3 ffnormal = faceforward(world_shade_normal, -optixGetWorldRayDirection(), world_shade_normal);
    phongShade(phong.Kd, phong.Ka, phong.Ks, phong.Kr, phong.phong_exp, ffnormal);

}

// transparent object like glass
extern "C" __global__ void __closesthit__transparency_radiance()
{
    const HitGroupData* sbt_data = (HitGroupData*)optixGetSbtDataPointer();
    Glass glass = sbt_data->shading.glass;

    float3 geometry_normal = make_float3(
        int_as_float(optixGetAttribute_0()),
        int_as_float(optixGetAttribute_1()),
        int_as_float(optixGetAttribute_2()));

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();
    const float  ray_t    = optixGetRayTmax();

    SunPRD* sun_prd = getPRD<SunPRD>();
    if (sun_prd->countEmitted)
        sun_prd->emitted = make_float3(0.0f);// no emittion
    else
        sun_prd->emitted = make_float3(0.0f);

    //sun_prd->attenuation *= p_Kd;
    sun_prd->countEmitted = false;

    float3 t;                                    
    float3 r;
    float3 hit_point = ray_orig + ray_t * ray_dir;
    float3 front_hit_point = hit_point, back_hit_point = hit_point;

    float3 shade_normal = geometry_normal;
    CubeHitType hit_type = (CubeHitType)optixGetHitKind();

    normalize(shade_normal);
    float3 world_shade_normal = normalize(optixTransformNormalFromObjectToWorldSpace(shade_normal));
    if (hit_type == HIT_FROM_OUTSIDE) {
        front_hit_point += params.scene_epsilon * shade_normal;
        back_hit_point -= params.scene_epsilon * shade_normal;
    }
    else {
        front_hit_point -= params.scene_epsilon * shade_normal;
        back_hit_point += params.scene_epsilon * shade_normal;
    }
    const float3 fhp = optixTransformPointFromObjectToWorldSpace(front_hit_point);
    const float3 bhp = optixTransformPointFromObjectToWorldSpace(back_hit_point);

    float reflection = 1.0f;
    float3 result = make_float3(0.0f);

    const int depth = sun_prd->depth;
    float3 beer_attenuation;

    if (dot(world_shade_normal, ray_dir) > 0)
    {
        // Beer's law attenuation
        beer_attenuation = exp(glass.extinction_constant * ray_t);
    }
    else
    {
        beer_attenuation = make_float3(1);
    }

    // refraction
    // compare depth to max_depth - 1 to leave room for a potential shadow ray trace
    if (depth < min(glass.refraction_maxdepth, params.max_depth - 1))
    {
        if (refract(t, ray_dir, world_shade_normal, glass.refraction_index))
        {
            // check for external or internal reflection
            float cos_theta = dot(ray_dir, world_shade_normal);
            if (cos_theta < 0.0f)
                cos_theta = -cos_theta;
            else
                cos_theta = dot(t, world_shade_normal);

            reflection = fresnel_schlick(
                cos_theta,
                glass.fresnel_exponent,
                glass.fresnel_minimum,
                glass.fresnel_maximum);

            float importance =
                sun_prd->importance
                * (1.0f - reflection)
                * luminance(glass.refraction_color * beer_attenuation);
            float3 color = glass.cutoff_color;
            if (importance > glass.importance_cutoff)
            {
                color = traceSun(bhp, t, depth + 1, importance,sun_prd->attenuation * beer_attenuation);
            }
            result += (1.0f - reflection) * glass.refraction_color * color;
        }
        // else TIR
    } // else reflection==1 so refraction has 0 weight
    // reflection
    // compare depth to max_depth - 1 to leave room for a potential shadow ray trace
    float3 color = glass.cutoff_color;
    if (depth < min(glass.reflection_maxdepth, params.max_depth - 1))
    {
        r = reflect(ray_dir, world_shade_normal);

        float importance =
            sun_prd->importance
            * reflection
            * luminance(glass.reflection_color * beer_attenuation);
        if (importance > glass.importance_cutoff)
        {
            color = traceSun(fhp, r, depth + 1, importance,sun_prd->attenuation * beer_attenuation);
        }
    }
    result += reflection * glass.reflection_color * color;
    result = result ;
    sun_prd->attenuation *= beer_attenuation;

    sun_prd->radiance = result;
    unsigned int u0, u1;
    packPointer(&sun_prd, u0, u1);
    optixSetPayload_0(u0);
    optixSetPayload_1(u1);
}

// water object
extern "C" __global__ void __closesthit__water_radiance()

{
    const HitGroupData* sbt_data = (HitGroupData*)optixGetSbtDataPointer();
    Water water = sbt_data->shading.water;
    float3 geometry_normal = make_float3(
        int_as_float(optixGetAttribute_0()),
        int_as_float(optixGetAttribute_1()),
        int_as_float(optixGetAttribute_2()));

    float3 shade_normal = geometry_normal;
    // texture coordinate
    float2 coord = make_float2(
        int_as_float(optixGetAttribute_3()),
        int_as_float(optixGetAttribute_4()));

        //which cube face the hitpoint in
    cube_face face = cube_face(optixGetAttribute_5());

    normalize(shade_normal);
    float3 world_shade_normal = normalize(optixTransformNormalFromObjectToWorldSpace(shade_normal));
    float3 p_normal = faceforward(world_shade_normal, -optixGetWorldRayDirection(), world_shade_normal);
    float3 p_Kd = water.Kd, p_Ka = water.Ka, p_Ks = water.Ks, p_Kr = water.Kr;
    float p_phong_exp = water.phong_exp;

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();
    const float  ray_t = optixGetRayTmax();

    SunPRD* sun_prd = getPRD<SunPRD>();

    float3 hit_point = ray_orig + ray_t * ray_dir;
    if (sun_prd->countEmitted)
        sun_prd->emitted = make_float3(0.0f);//���屾������
    else
        sun_prd->emitted = make_float3(0.0f);
    sun_prd->attenuation *= p_Kd;
    sun_prd->countEmitted = false;
    // ambient contribution
    float3 result = p_Ka * params.ambient_light_color;


    //sun light

    const int numLightSamples = params.num_lights_sample;
    for (int lightSampleID = 0; lightSampleID < numLightSamples; lightSampleID++) {
        DirectionalLight sun = params.sun;
        const float3 light_center = hit_point + sun.direction * 10.0f;
        const float r1 = rnd(sun_prd->seed);
        const float r2 = rnd(sun_prd->seed);
        const float2 disk_sample = square_to_disk(make_float2(r1, r2));
        const float3 jittered_pos = light_center + sun.radius * disk_sample.x * sun.v0 + sun.radius * disk_sample.y * sun.v1;
        float3 L = normalize(jittered_pos - hit_point);

        const float NdotL = dot(p_normal, L);
        if (NdotL > 0.0f) {
            OcclusionPRD shadow_prd;
            shadow_prd.attenuation = make_float3(1.0f);

            optixTrace(
                params.handle,
                hit_point + p_normal * params.scene_epsilon,
                L,
                0.01f,
                1e16,
                0.0f,
                OptixVisibilityMask(1),
                OPTIX_RAY_FLAG_NONE,
                RAY_TYPE_OCCLUSION,
                RAY_TYPE_COUNT,
                RAY_TYPE_OCCLUSION,
                float3_as_args(shadow_prd.attenuation));

            float3 light_attenuation = shadow_prd.attenuation;

            if (fmaxf(light_attenuation) > 0.0f)
            {
                const float solid_angle = sun.radius * sun.radius * M_PIf;

                float3 Lc = light_attenuation * tonemap(sun.color * solid_angle);
                result += p_Kd * NdotL * Lc / numLightSamples;

                float3 H = normalize(L - ray_dir);
                float nDh = dot(p_normal, H);
                if (nDh > 0)
                {
                    float power = pow(nDh, p_phong_exp);
                    result += p_Ks * power * Lc/ numLightSamples;
                }
            }
        }
    }
    if (fmaxf(p_Kr) > 0)
    {
        // ray tree attenuation
        float new_importance = sun_prd->importance * luminance(p_Kr);
        int new_depth = sun_prd->depth + 1;

        // reflection ray
        // compare new_depth to max_depth - 1 to leave room for a potential shadow ray trace
        if (new_importance >= 0.01f && new_depth <= params.max_depth - 1)
        {
            float3 R = reflect(ray_dir, p_normal);
            result += p_Kr * traceSun(hit_point + p_normal * params.scene_epsilon, R, new_depth, new_importance, sun_prd->attenuation*p_Kd);
        }
    }

    float local_transparency = water.transparency;

    float importance_R = sun_prd->importance * luminance(make_float3(local_transparency, local_transparency, local_transparency));
    // refraction ray
    if (importance_R > water.importance_cutoff && sun_prd->depth < min(params.max_depth - 1, water.refraction_maxdepth))
    {
        float3 R;
        refract(R, ray_dir, p_normal, water.refractivity_n);
        result *= (1.0f - local_transparency);
        result += local_transparency * traceSun(hit_point + p_normal * params.scene_epsilon, R, sun_prd->depth + 1, sun_prd->importance, sun_prd->attenuation);
    }

        

    // compute Point light
    sun_prd->radiance = result;

    result = make_float3(0.0f);
    for (int light_id = 0; light_id < params.point_light_sum; light_id++) {        

        float Ldist = length(params.point_light[light_id].pos - hit_point);
        float light_fade = 1.0f;
            if (Ldist > 1.0f) {
                light_fade = 1.0f / (pow(Ldist, 1.2f));
            }
        for (int lightSampleID = 0; lightSampleID < numLightSamples; lightSampleID++) {
            float3 du = make_float3(1.0f, 0.0f, 0.0f) * 0.5;
            float3 dv = make_float3(0.0f, 0.0f, 1.0f) * 0.5;
            const float r1 = rnd(sun_prd->seed);
            const float r2 = rnd(sun_prd->seed);
            BasicLight light = params.point_light[light_id];
            const float3 jittered_pos = light.pos + r1 * du + r2 * dv;
            float Ldist = length(jittered_pos - hit_point);
            float3 L = normalize(jittered_pos - hit_point);

            const float NdotL = dot(p_normal, L);
            if (NdotL > 0.0f) {
                OcclusionPRD shadow_prd;
                shadow_prd.attenuation = make_float3(1.0f);

                optixTrace(
                    params.handle,
                    hit_point + p_normal * params.scene_epsilon,
                    L,
                    0.01f,
                    Ldist-0.01f,
                    0.0f,
                    OptixVisibilityMask(1),
                    OPTIX_RAY_FLAG_NONE,
                    RAY_TYPE_OCCLUSION,
                    RAY_TYPE_COUNT,
                    RAY_TYPE_OCCLUSION,
                    float3_as_args(shadow_prd.attenuation));

                float3 light_attenuation = shadow_prd.attenuation;

                if (fmaxf(light_attenuation) > 0.0f)
                {

                    float3 Lc = light_attenuation * light.color;
                    result += p_Kd * NdotL * Lc / numLightSamples;

                    float3 H = normalize(L - ray_dir);
                    float nDh = dot(p_normal, H);
                    if (nDh > 0)
                    {
                        float power = pow(nDh, p_phong_exp);
                        result += p_Ks * power * Lc/ numLightSamples * light_fade;
                    }
                }
            }
        }
        if (fmaxf(p_Kr) > 0)
        {
            // ray tree attenuation
            float new_importance = sun_prd->importance * luminance(p_Kr);
            int new_depth = sun_prd->depth + 1;

            // reflection ray
            // compare new_depth to max_depth - 1 to leave room for a potential shadow ray trace
            if (new_importance >= 0.01f && new_depth <= params.max_depth - 1)
            {
                float3 R = reflect(ray_dir, p_normal);
                result += p_Kr * traceSun(hit_point + p_normal * params.scene_epsilon, R, new_depth, new_importance, sun_prd->attenuation)* light_fade;
            }
        }

        float local_transparency = water.transparency;

        float importance_R = sun_prd->importance * luminance(make_float3(local_transparency, local_transparency, local_transparency));
        // refraction ray
        if (importance_R > water.importance_cutoff && sun_prd->depth < min(params.max_depth - 1, water.refraction_maxdepth))
        {
            float3 R;
            refract(R, ray_dir, p_normal, water.refractivity_n);
            result *= (1.0f - local_transparency);
            result += local_transparency * traceSun(hit_point + p_normal * params.scene_epsilon, R, sun_prd->depth + 1, sun_prd->importance, sun_prd->attenuation* p_Kd)*light_fade;
        }
    }
    // ray tree attenuation
   

    sun_prd->radiance += result;
    unsigned int u0, u1;
    packPointer(&sun_prd, u0, u1);
    optixSetPayload_0(u0);
    optixSetPayload_1(u1);
    // pass the color back

}

// transparent and emitted, design for point light
extern "C" __global__ void __closesthit__glass_radiance()
{
    const HitGroupData* sbt_data = (HitGroupData*) optixGetSbtDataPointer();
    const Glass &glass = sbt_data->shading.glass;

    //RadiancePRD prd_radiance = getRadiancePRD();
    SunPRD * sun_prd = getPRD<SunPRD>();

    if (sun_prd->countEmitted) {
        sun_prd->emitted = make_float3(0.5,0.6,0.9);//make our point light emitted
    }
    else
        sun_prd->emitted = make_float3(0.0f);

    //sun_prd->attenuation *= p_Kd;
    sun_prd->countEmitted = false;

    float3 object_normal = make_float3(
        int_as_float( optixGetAttribute_0() ),
        int_as_float( optixGetAttribute_1() ),
        int_as_float( optixGetAttribute_2() ));
    object_normal = normalize( object_normal );

    // intersection vectors
    const float3 n = normalize( optixTransformNormalFromObjectToWorldSpace( object_normal) ); // normal
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();                 // incident direction
    const float  ray_t    = optixGetRayTmax();
    float3 t;                                                            // transmission direction
    float3 r;                                                            // reflection direction

    float3 hit_point = ray_orig + ray_t * ray_dir;
    SphereShellHitType hit_type = (SphereShellHitType) optixGetHitKind();
    float3 front_hit_point = hit_point, back_hit_point = hit_point;

    if (hit_type & HIT_OUTSIDE_FROM_OUTSIDE || hit_type & HIT_INSIDE_FROM_INSIDE)
    {
        front_hit_point += params.scene_epsilon * object_normal;
        back_hit_point  -= params.scene_epsilon * object_normal;
    }
    else
    {
        front_hit_point -= params.scene_epsilon * object_normal;
        back_hit_point  += params.scene_epsilon * object_normal;
    }

    const float3 fhp = optixTransformPointFromObjectToWorldSpace( front_hit_point );
    const float3 bhp = optixTransformPointFromObjectToWorldSpace( back_hit_point );

    float reflection = 1.0f;
    float3 result = make_float3(0.0f);

    //const int depth = prd_radiance.depth;
    const int depth = sun_prd->depth;

    float3 beer_attenuation;
    if(dot(n, ray_dir) > 0)
    {
        // Beer's law attenuation
        beer_attenuation = exp(glass.extinction_constant * ray_t);
    } else
    {
        beer_attenuation = make_float3(1);
    }

    // refraction
    // compare depth to max_depth - 1 to leave room for a potential shadow ray trace
    if (depth < min(glass.refraction_maxdepth, params.max_depth - 1))
    {
        if ( refract(t, ray_dir, n, glass.refraction_index) )
        {
            // check for external or internal reflection
            float cos_theta = dot(ray_dir, n);
            if (cos_theta < 0.0f)
                cos_theta = -cos_theta;
            else
                cos_theta = dot(t, n);

            reflection = fresnel_schlick(
                cos_theta,
                glass.fresnel_exponent,
                glass.fresnel_minimum,
                glass.fresnel_maximum);

            //float importance =
            //    prd_radiance.importance
            //   * (1.0f-reflection)
            //   * luminance( glass.refraction_color * beer_attenuation );
            float importance =
                sun_prd->importance
                * (1.0f - reflection)
                * luminance(glass.refraction_color * beer_attenuation);
            float3 color = glass.cutoff_color;
            if ( importance > glass.importance_cutoff )
            {
                //color = traceRadianceRay(bhp, t, depth+1, importance);
                //sun_prd->depth += 1;
                color = traceSun(bhp, t, depth + 1, importance,sun_prd->attenuation);
            }
            result += (1.0f - reflection) * glass.refraction_color * color;
        }
        // else TIR
    } // else reflection==1 so refraction has 0 weight

    // reflection
    // compare depth to max_depth - 1 to leave room for a potential shadow ray trace
    float3 color = glass.cutoff_color;
    if (depth < min(glass.reflection_maxdepth, params.max_depth - 1))
    {
        r = reflect(ray_dir, n);

        //float importance =
        //    prd_radiance.importance
        //    * reflection
        //    * luminance( glass.reflection_color * beer_attenuation );
        float importance =
            sun_prd->importance
            * reflection
            * luminance( glass.reflection_color * beer_attenuation );
        if ( importance > glass.importance_cutoff )
        {
           //color = traceRadianceRay( fhp, r, depth+1, importance );
           color = traceSun(fhp, r, depth + 1, importance,sun_prd->attenuation);

        }
    }
    result += reflection * glass.reflection_color * color;
    sun_prd->attenuation *= beer_attenuation;
    result = result ;

    //printf("%f,%f,%f\n", result.x, result.y, result.z);
    sun_prd->radiance = result;

    unsigned int u0, u1;
    packPointer(&sun_prd, u0, u1);
    optixSetPayload_0(u0);
    optixSetPayload_1(u1);
    //prd_radiance.result = result;
    //setRadiancePRD(prd_radiance);
}

extern "C" __global__ void __anyhit__glass_occlusion()
{
    const HitGroupData* sbt_data = (HitGroupData*) optixGetSbtDataPointer();
    const Glass &glass = sbt_data->shading.glass;

    float3 object_normal = make_float3(
        int_as_float( optixGetAttribute_0() ),
        int_as_float( optixGetAttribute_1() ),
        int_as_float( optixGetAttribute_2() ));

    OcclusionPRD shadow_prd = getOcclusionPRD();

    float3 world_normal = normalize( optixTransformNormalFromObjectToWorldSpace( object_normal ) );
    float nDi = fabs(dot(world_normal, optixGetWorldRayDirection()));

    shadow_prd.attenuation *= 1-fresnel_schlick(nDi, 5, 1-glass.shadow_attenuation, make_float3(1));
    setOcclusionPRD(shadow_prd);

    // Test the attenuation of the light from the glass shell
    if(luminance(shadow_prd.attenuation) < glass.importance_cutoff)
        // The attenuation is so high, > 99% blocked, that we can consider testing to be done.
        optixTerminateRay();
    else
        // There is still some light coming through the glass shell that we should test other occluders.
        // We "ignore" the intersection with the glass shell, meaning that shadow testing will continue.
        // If the ray does not hit another occluder, the light's attenuation from this glass shell
        // (along with other glass shells) is then used.
        optixIgnoreIntersection();
}

extern "C" __global__ void __closesthit__full_occlusion()
{
    phongShadowed();
}
