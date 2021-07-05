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
    float3 normal; 
    float3 normalx = { 1.0f,0.0f,0.0f };
    float3 normaly = { 0.0f,1.0f,0.0f };
    float3 normalz = { 0.0f,0.0f,1.0f };
    normal = normalx;
    if ((tmin <= tymax) && (tymin <= tmax)){
        if (tymin > tmin) {
            tmin = tymin;
            normal = normaly;
        }
        if (tymax < tmax) {
            tmax = tymax;
            normal = normaly;
        }

        tzmin = (bounds[sign[2]].z - ray_orig.z) * ray_dir.z; 
        tzmax = (bounds[1-sign[2]].z - ray_orig.z) * ray_dir.z;

        if ((tmin <= tzmax) && (tzmin <= tmax)){
 
            if (tzmin > tmin) {
                tmin = tzmin;
                normal = normalz;
            }
            if (tzmax < tmax) {
                tmax = tzmax;
                normal = normalz;
            }
            t = tmin; 
            if (t < 0) { 
                t = tmax;  
            } 

            if( t>=ray_tmin && t<=ray_tmax){
                optixReportIntersection(
                    t,
                    0,
                    float3_as_ints(normal)
                    
                    );
            }
        }
    }
}
