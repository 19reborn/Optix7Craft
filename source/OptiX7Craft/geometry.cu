#include <optix.h>

#include "OptiX7Craft.h"
#include "helpers.h"
#include "GeometryData.h"

extern "C" {
__constant__ Params params;
}

extern "C" __global__ void __intersection__parallelogram()
{
    const Parallelogram* floor = reinterpret_cast<Parallelogram*>( optixGetSbtDataPointer() );

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();
    const float ray_tmin = optixGetRayTmin(), ray_tmax = optixGetRayTmax();

    float3 n = make_float3( floor->plane );
    float dt = dot(ray_dir, n );
    float t = (floor->plane.w - dot(n, ray_orig))/dt;
    if( t > ray_tmin && t < ray_tmax )
    {
        float3 p = ray_orig + ray_dir * t;
        float3 vi = p - floor->anchor;
        float a1 = dot(floor->v1, vi);
        if(a1 >= 0 && a1 <= 1)
        {
            float a2 = dot(floor->v2, vi);
            if(a2 >= 0 && a2 <= 1)
            {
                optixReportIntersection(
                    t,
                    0,
                    float3_as_args(n),
                    float_as_int( a1 ), float_as_int( a2 )
                    );
            }
        }
    }
}

extern "C" __global__ void __intersection__sphere_shell()
{
    const SphereShell* sphere_shell = reinterpret_cast<SphereShell*>( optixGetSbtDataPointer() );
    const float3  ray_orig = optixGetWorldRayOrigin();
    const float3  ray_dir  = optixGetWorldRayDirection();
    const float   ray_tmin = optixGetRayTmin(), ray_tmax = optixGetRayTmax();

    float3 O = ray_orig - sphere_shell->center;
    float  l = 1 / length(ray_dir);
    float3 D = ray_dir * l;

    float b = dot(O, D), sqr_b = b * b;
    float O_dot_O = dot(O, O);
    float radius1 = sphere_shell->radius1, radius2 = sphere_shell->radius2;
    float sqr_radius1 = radius1 * radius1, sqr_radius2 = radius2*radius2;

    // check if we are outside of outer sphere
    if ( O_dot_O > sqr_radius2 + params.scene_epsilon )
    {
        if ( O_dot_O - sqr_b < sqr_radius2 - params.scene_epsilon )
        {
            float c = O_dot_O - sqr_radius2;
            float root = sqr_b - c;
            if (root > 0.0f) {
                float t = -b - sqrtf( root );
                float3 normal = (O + t * D) / radius2;
                optixReportIntersection(
                    t * l,
                    HIT_OUTSIDE_FROM_OUTSIDE,
                    float3_as_args( normal ) );
            }
        }
    }
    // else we are inside of the outer sphere
    else
    {
        float c = O_dot_O - sqr_radius1;
        float root = b*b-c;
        if ( root > 0.0f )
        {
            float t = -b - sqrtf( root );
            // do we hit inner sphere from between spheres?
            if ( t * l > ray_tmin && t * l < ray_tmax )
            {
                float3 normal = (O + t * D) / (-radius1);
                optixReportIntersection(
                    t * l,
                    HIT_INSIDE_FROM_OUTSIDE,
                    float3_as_args( normal ) );
            }
            else
            {
                // do we hit inner sphere from within both spheres?
                t = -b + (root > 0 ? sqrtf( root ) : 0.f);
                if ( t * l > ray_tmin && t * l < ray_tmax )
                {
                    float3 normal = ( O + t*D )/(-radius1);
                    optixReportIntersection(
                        t * l,
                        HIT_INSIDE_FROM_INSIDE,
                        float3_as_args( normal ) );
                }
                else
                {
                    // do we hit outer sphere from between spheres?
                    c = O_dot_O - sqr_radius2;
                    root = b*b-c;
                    t = -b + (root > 0 ? sqrtf( root ) : 0.f);
                    float3 normal = ( O + t*D )/radius2;
                    optixReportIntersection(
                        t * l,
                        HIT_OUTSIDE_FROM_INSIDE,
                        float3_as_args( normal ) );
                }
            }
        }
        else
        {
            // do we hit outer sphere from between spheres?
            c = O_dot_O - sqr_radius2;
            root = b*b-c;
            float t = -b + (root > 0 ? sqrtf( root ) : 0.f);
            float3 normal = ( O + t*D )/radius2;
            optixReportIntersection(
                t * l,
                HIT_OUTSIDE_FROM_INSIDE,
                float3_as_args( normal ) );
        }
    }
}

#define float3_as_ints( u ) float_as_int( u.x ), float_as_int( u.y ), float_as_int( u.z )

static __device__ float3 get_normal(float t, float3 t0, float3 t1)
{
    float3 neg = make_float3(t == t0.x ? 1 : 0, t == t0.y ? 1 : 0, t == t0.z ? 1 : 0);
    float3 pos = make_float3(t == t1.x ? 1 : 0, t == t1.y ? 1 : 0, t == t1.z ? 1 : 0);
    return pos - neg;
}

static __device__ float2 get_coord(float3 relativeCoord, float3 size)
{
    float2 uv;
    if (fabs(fabs(relativeCoord.z) - size.z) <= 1e-4) {
        uv.x = (relativeCoord.x + size.x) / (2 * size.x);
        uv.y = (relativeCoord.y + size.y) / (2 * size.y);
    }
    else if (fabs(fabs(relativeCoord.x) - size.x) <= 1e-4) {
        uv.x = (relativeCoord.z + size.z) / (2 * size.z);
        uv.y = (relativeCoord.y + size.y) / (2 * size.y);
    }
    else if (fabs(fabs(relativeCoord.y) - size.y) <= 1e-4) {
        uv.x = (relativeCoord.z + size.z) / (2 * size.z);
        uv.y = (relativeCoord.x + size.x) / (2 * size.x);
    }
    else {
        uv.x = 0.0f;
        uv.y = 0.0f;
    }
    return uv;
}

static __device__ cube_face get_face(float3 relativeCoord, float3 size)
{
    if (fabs(relativeCoord.x - size.x) <= 1e-4) {
        return x_up;
    }
    else if (fabs(relativeCoord.x + size.x) <= 1e-4) {
        return x_down;
    }
    else if (fabs(relativeCoord.y - size.y) <= 1e-4) {
        return y_up;
    }
    else if (fabs(relativeCoord.y + size.y) <= 1e-4) {
        return y_down;
    }
    else if (fabs(relativeCoord.z - size.z) <= 1e-4) {
        return z_up;
    }
    else if (fabs(relativeCoord.z + size.z) <= 1e-4) {
        return z_down;
    }
}

extern "C" __global__ void __intersection__cube()
{
    const Cube* cube = reinterpret_cast<Cube*>(optixGetSbtDataPointer());

    const float3 ray_origin = optixGetWorldRayOrigin();
    float3 ray_direction = optixGetWorldRayDirection();
    const float ray_tmin = optixGetRayTmin(), ray_tmax = optixGetRayTmax();
    float3 cubemin = cube->center - cube->size;
    float3 cubemax = cube->center + cube->size;
    float3 t0 = (cubemin - ray_origin) / ray_direction;
    float3 t1 = (cubemax - ray_origin) / ray_direction;
    float3 near = fminf(t0, t1);
    float3 far = fmaxf(t0, t1);
    float tmin = fmaxf(near);
    float tmax = fminf(far);

    float eps = 0.0001f;

    if (tmin <= tmax) {
        bool check_second = true;
        if (tmin >= ray_tmin && tmin <= ray_tmax) {
            float3 normal = get_normal(tmin, t0, t1);
            float3 coord = ray_origin + tmin * ray_direction;
            // ����texture�ϵ�u, v
            float3 relativeCoord = coord - cube->center;
            float2 uv = get_coord(relativeCoord, cube->size);
            cube_face face = get_face(relativeCoord, cube->size);

            optixReportIntersection(
                tmin,
                HIT_FROM_OUTSIDE,
                float3_as_args(normal),
                float_as_int(uv.x),
                float_as_int(uv.y),
                face
            );

            check_second = false;
        }
        if (check_second) {
            if (tmax >= ray_tmin && tmax <= ray_tmax) {
                float3 normal = get_normal(tmax, t0, t1);
                float3 coord = ray_origin + tmax * ray_direction;
                // ����texture�ϵ�u, v
                float3 relativeCoord = coord - cube->center;
                float2 uv = get_coord(relativeCoord, cube->size);
                cube_face face = get_face(relativeCoord, cube->size);

                optixReportIntersection(
                    tmax,
                    HIT_FROM_INSIDE,
                    float3_as_args(normal),
                    float_as_int(uv.x),
                    float_as_int(uv.y),
                    face
                );
            }
        }
    }
}

extern "C" __global__ void __intersection__sphere()
{
    const GeometryData::Sphere* hit_group_data = reinterpret_cast<GeometryData::Sphere*>(optixGetSbtDataPointer());

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();
    const float  ray_tmin = optixGetRayTmin();
    const float  ray_tmax = optixGetRayTmax();

    const float3 O = ray_orig - hit_group_data->center;
    const float  l = 1.0f / length(ray_dir);
    const float3 D = ray_dir * l;
    const float  radius = hit_group_data->radius;

    float b = dot(O, D);
    float c = dot(O, O) - radius * radius;
    float disc = b * b - c;
    if (disc > 0.0f)
    {
        float sdisc = sqrtf(disc);
        float root1 = (-b - sdisc);
        float root11 = 0.0f;
        bool  check_second = true;

        const bool do_refine = fabsf(root1) > (10.0f * radius);

        if (do_refine)
        {
            // refine root1
            float3 O1 = O + root1 * D;
            b = dot(O1, D);
            c = dot(O1, O1) - radius * radius;
            disc = b * b - c;

            if (disc > 0.0f)
            {
                sdisc = sqrtf(disc);
                root11 = (-b - sdisc);
            }
        }

        float  t;
        float3 normal;
        t = (root1 + root11) * l;
        if (t > ray_tmin && t < ray_tmax)
        {
            normal = (O + (root1 + root11) * D) / radius;
            if (optixReportIntersection(t, 0, float3_as_ints(normal), float_as_int(radius)))
                check_second = false;
        }

        if (check_second)
        {
            float root2 = (-b + sdisc) + (do_refine ? root1 : 0);
            t = root2 * l;
            normal = (O + root2 * D) / radius;
            if (t > ray_tmin && t < ray_tmax)
                optixReportIntersection(t, 0, float3_as_ints(normal), float_as_int(radius));
        }
    }
}
