#ifndef ENTITY_H_FILE
#define ENTITY_H_FILE

#include <vector_types.h>
#include <optix_types.h>
#include <sutil/vec_math.h>
#include <cuda/GeometryData.h>

enum {ENTITY_CREATURE, ENTITY_PARTICLE, ENTITY_OBJECT, ENTITY_ARROW};
class Entity {
public:
    float3 pos = make_float3(1.f, 0.f, 1.f);
    float3 acceleration = make_float3(0.f, 0.f, 0.f); 
    float3 velocity = make_float3(0.f, 0.f, 0.f);
    bool isOnGround = true;
    bool isFlying = false;
    void dx(float3 vec)
    {
        pos += vec;
    }
    void dv(float3 vec)
    {
        velocity += vec;
    }
    void da(float3 vec)
    {
        acceleration += vec;
    }
};

struct Creature : public Entity {
    int type = ENTITY_CREATURE;
    float3 eye = make_float3(0.f, 1.3f, 0.f);
    float3 lookat = make_float3(0.f, 0.f, 0.f);
    float3 up = make_float3(0.f, 1.f, 0.f);
    void dx(float3 vec)
    {
        pos += vec;
        eye += vec;
        lookat += vec;
    }
    //@@todo: link with collidebox
};
#endif