#include <optix.h>

#include "optixWhitted.h"
#include "helpers.h"

#define float3_as_ints( u ) float_as_int( u.x ), float_as_int( u.y ), float_as_int( u.z )
extern "C" {
__constant__ Params params;
}

extern "C" __global__ void __intersection__cube()
{
    const Cube* cube = reinterpret_cast<Cube*>( optixGetSbtDataPointer() );

    const float3 ray_orig = optixGetWorldRayOrigin();
    float3 ray_dir  = optixGetWorldRayDirection();
    const float ray_tmin = optixGetRayTmin(), ray_tmax = optixGetRayTmax();

    ray_dir = 1.0f / ray_dir;
    int sign[3];
    sign[0] = (ray_dir.x < 0); 
    sign[1] = (ray_dir.y < 0); 
    sign[2] = (ray_dir.z < 0); 

    float3 bounds[2];
    bounds[0] = cube->center - cube->size, bounds[1] = cube->center + cube->size;

    float tmin, tmax, tymin, tymax, tzmin, tzmax, t; 

    tmin = (bounds[sign[0]].x - ray_orig.x) * ray_dir.x; 
    tmax = (bounds[1-sign[0]].x - ray_orig.x) * ray_dir.x; 
    tymin = (bounds[sign[1]].y - ray_orig.y) * ray_dir.y; 
    tymax = (bounds[1-sign[1]].y - ray_orig.y) * ray_dir.y; 
    float3 normal_min,normal_max, normal; 
    float3 coord;
    float3 normalx = { 1.0f,0.0f,0.0f };
    float3 normaly = { 0.0f,1.0f,0.0f };
    float3 normalz = { 0.0f,0.0f,1.0f };
    normal_min = -normalx*(1- 2* sign[0]);
    normal_max = normalx * (1 - 2 * sign[0]);
    if ((tmin <= tymax) && (tymin <= tmax)){
        if (tymin > tmin) {
            tmin = tymin;
            normal_min = -normaly* (1 - 2 * sign[1]);
        }
        if (tymax < tmax) {
            tmax = tymax;
            normal_max = normaly* (1 - 2 * sign[1]);
        }

        tzmin = (bounds[sign[2]].z - ray_orig.z) * ray_dir.z; 
        tzmax = (bounds[1-sign[2]].z - ray_orig.z) * ray_dir.z;

        if ((tmin <= tzmax) && (tzmin <= tmax)){
 
            if (tzmin > tmin) {
                tmin = tzmin;
                normal_min = -normalz* (1 - 2 * sign[2]);
            }
            if (tzmax < tmax) {
                tmax = tzmax;
                normal_max = normalz *(1 - 2 * sign[2]);
            }
            t = tmin; 
            normal = normal_min;
            if (t < 0) { 
                t = tmax;  
                normal = normal_max;
            } 
            float3 coord = ray_orig + t / ray_dir;

            // 计算texture上的u, v
            float3 relativeCoord = coord - cube->center;
            float2 uv;
            
            if(fabs(relativeCoord.z) == cube->size.z) {
                uv.x = (relativeCoord.x + cube->size.x) / (2*cube->size.x);
                uv.y = (relativeCoord.y + cube->size.y) / (2*cube->size.y);
            } else if(fabs(relativeCoord.x) == cube->size.x) {
                uv.x = (relativeCoord.y + cube->size.y) / (2*cube->size.y);
                uv.y = (relativeCoord.z + cube->size.z) / (2*cube->size.z);
            } else if(fabs(relativeCoord.y) == cube->size.y) {
                // 可能要换下顺序?
                uv.x = (relativeCoord.z + cube->size.z) / (2*cube->size.z);
                uv.y = (relativeCoord.x + cube->size.x) / (2*cube->size.x);
            }


            if( t>=ray_tmin && t<=ray_tmax){
                optixReportIntersection(
                    t,
                    0,
                    float3_as_ints(normal),
                    float2_as_ints(uv)
                    );
            }
        }
    }
}
