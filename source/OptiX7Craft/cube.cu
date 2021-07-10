#include <optix.h>

#include "OptiX7Craft.h"
#include "helpers.h"

#define float3_as_ints( u ) float_as_int( u.x ), float_as_int( u.y ), float_as_int( u.z )
extern "C" {
__constant__ Params params;
}

static __device__ float3 get_normal(float t, float3 t0, float3 t1)
{
    float3 neg = make_float3(t == t0.x ? 1 : 0, t == t0.y ? 1 : 0, t == t0.z ? 1 : 0);
    float3 pos = make_float3(t == t1.x ? 1 : 0, t == t1.y ? 1 : 0, t == t1.z ? 1 : 0);
    return pos - neg;
}

static __device__ float2 get_coord(float3 relativeCoord, float3 size)
{
    float2 uv;
    if (fabs(fabs(relativeCoord.z) -size.z) <=1e-4) {
        uv.x = (relativeCoord.x + size.x) / (2 * size.x);
        uv.y = (relativeCoord.y + size.y) / (2 * size.y);
    }
    else if (fabs(fabs(relativeCoord.x) - size.x) <=1e-4) {
        uv.x = (relativeCoord.z + size.z) / (2 * size.z);
        uv.y = (relativeCoord.y + size.y) / (2 * size.y);
    }
    else if (fabs(fabs(relativeCoord.y) -size.y) <=1e-4) {
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
    if (fabs(relativeCoord.x -size.x) <=1e-4) {
        return x_up;
    }
    else if (fabs(relativeCoord.x +size.x)<=1e-4) {
        return x_down;
    }
    else if (fabs(relativeCoord.y -size.y)<=1e-4) {
        return y_up;
    }
    else if (fabs(relativeCoord.y +size.y)<=1e-4) {
        return y_down;
    }
    else if (fabs(relativeCoord.z -size.z)<=1e-4) {
        return z_up;
    }
    else if (fabs(relativeCoord.z +size.z)<=1e-4) {
        return z_down;
    }
}


extern "C" __global__ void __intersection__cube()
{
    const Cube* cube = reinterpret_cast<Cube*>( optixGetSbtDataPointer() );

    const float3 ray_origin = optixGetWorldRayOrigin();
    float3 ray_direction  = optixGetWorldRayDirection();
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
            // 计算texture上的u, v
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
                // 计算texture上的u, v
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