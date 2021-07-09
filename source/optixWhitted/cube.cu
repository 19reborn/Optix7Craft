#include <optix.h>

#include "optixWhitted.h"
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
                0,
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
                    0,
                    float3_as_args(normal),
                    float_as_int(uv.x),
                    float_as_int(uv.y),
                    face
                );
            }
        }
    }
   
    /*
    ray_dir = 1.0f / (ray_dir+make_float3(1e-5));
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
    float3 normalx = make_float3(1.0f,0.0f,0.0f);
    float3 normaly = make_float3(0.0f,1.0f,0.0f);
    float3 normalz = make_float3(0.0f,0.0f,1.0f);
    //normal_min = -normalx*(1- 2* sign[0]);
    //normal_max = -normalx * (1 - 2 * sign[0]);
    normal_min = normalx;
    normal_max = normalx;
    if ((tmin <= tymax) && (tymin <= tmax)){
        if (tymin > tmin) {
            tmin = tymin;
            //normal_min = -normaly* (1 - 2 * sign[1]);
            normal_min = normaly;
        }
        if (tymax < tmax) {
            tmax = tymax;
            //normal_max =  -normaly* (1 - 2 * sign[1]);
            normal_max = normaly;
        }

        tzmin = (bounds[sign[2]].z - ray_orig.z) * ray_dir.z; 
        tzmax = (bounds[1-sign[2]].z - ray_orig.z) * ray_dir.z;

        if ((tmin <= tzmax) && (tzmin <= tmax)){
 
            if (tzmin > tmin) {
                tmin = tzmin;
                //normal_min = -normalz* (1 - 2 * sign[2]);
                normal_min = normalz;
            }
            if (tzmax < tmax) {
                tmax = tzmax;
               //normal_max = -normalz *(1 - 2 * sign[2]);
                normal_max = normalz ;
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


            normal *= (1 - 2 * (dot(relativeCoord, normal) < 0));

            if( t>=ray_tmin && t<=ray_tmax){
                optixReportIntersection(
                    t,
                    0,
                    float3_as_args(normal),
                    float_as_int(uv.x),
                    float_as_int(uv.y)
                    );
            }
        }
    }
    */
}

/*
void __{

      ray_dir = 1.0f / (ray_dir+make_float3(1e-5));
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
        float3 normalx = make_float3(1.0f,0.0f,0.0f);
        float3 normaly = make_float3(0.0f,1.0f,0.0f);
        float3 normalz = make_float3(0.0f,0.0f,1.0f);
        //normal_min = -normalx*(1- 2* sign[0]);
        //normal_max = -normalx * (1 - 2 * sign[0]);
        normal_min = normalx;
        normal_max = normalx;
        if ((tmin <= tymax) && (tymin <= tmax)){
            if (tymin > tmin) {
                tmin = tymin;
                //normal_min = -normaly* (1 - 2 * sign[1]);
                normal_min = normaly;
            }
            if (tymax < tmax) {
                tmax = tymax;
                //normal_max =  -normaly* (1 - 2 * sign[1]);
                normal_max = normaly;
            }

            tzmin = (bounds[sign[2]].z - ray_orig.z) * ray_dir.z;
            tzmax = (bounds[1-sign[2]].z - ray_orig.z) * ray_dir.z;

            if ((tmin <= tzmax) && (tzmin <= tmax)){

                if (tzmin > tmin) {
                    tmin = tzmin;
                    //normal_min = -normalz* (1 - 2 * sign[2]);
                    normal_min = normalz;
                }
                if (tzmax < tmax) {
                    tmax = tzmax;
                   //normal_max = -normalz *(1 - 2 * sign[2]);
                    normal_max = normalz ;
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
                normal *= (1 - 2 * (dot(relativeCoord, normal) < 0));
                if( t>=ray_tmin && t<=ray_tmax){
                    optixReportIntersection(
                        t,
                        0,
                        float3_as_args(normal),
                        float_as_int(uv.x),
                        float_as_int(uv.y)
                        );
                }
            }
        }
 }
 */