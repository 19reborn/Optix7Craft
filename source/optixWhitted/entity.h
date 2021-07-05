#ifndef ENTITY_H_FILE
#define ENTITY_H_FILE

#include <vector_types.h>
#include <optix_types.h>
#include <sutil/vec_math.h>
#include <cuda/GeometryData.h>

struct Entity {
    float3 pos = make_float3( 1.f,0.f,1.f );
    float3 eye = make_float3( 0.f,1.3f,0.f);
    float3 lookat = make_float3(0.f,0.f,0.f );
    float3 up = make_float3(0.f,1.f,0.f);
    float3 acceleration = make_float3(0.f, 0.f, 0.f); 
    float3 velocity = make_float3(0.f, 0.f, 0.f);
    bool isOnGround = true;
    bool isFlying = false;
};

#endif