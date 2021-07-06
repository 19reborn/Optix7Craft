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
    CollideBox box = CollideBox(make_float3(0,0,0), make_float3(0, 0, 0));
    virtual void dx(const float delta)
    {
        pos.x += delta;
        box.center.x += delta;
    }
    virtual void dy(const float delta)
    {
        pos.y += delta;
        box.center.y += delta;
    }
    virtual void dz(const float delta)
    {
        pos.z += delta;
        box.center.z += delta;
    }
    virtual void dX(const float3& vec)
    {
        pos += vec;
        box.center += vec;
    }
    virtual void dv(const float3& vec)
    {
        velocity += vec;
    }
    virtual void da(const float3& vec)
    {
        acceleration += vec;
    }
    virtual CollideBox& get_collideBox()
    {
        return box;
    }
    virtual bool collide(const CollideBox& cbox)
    {
        if (CollideBox::collide_check(box, cbox))
        {
            return true;
        }
        return false;
    }
    virtual bool collide_atEntity(Entity*& ent)
    {
        if (CollideBox::collide_check(box, ent->get_collideBox()))
        {
            return true;
        }
        return false;
    }
    

};

struct Creature : public Entity {
    int type = ENTITY_CREATURE;
    float3 eye = make_float3(0.f, 1.3f, 0.f);
    float3 lookat = make_float3(0.f, 0.f, 0.f);
    float3 up = make_float3(0.f, 1.f, 0.f);
    void dx(const float delta)
    {
        pos.x += delta;
        eye.x += delta;
        lookat.x += delta;
        box.center.x += delta;
    }
    void dy(const float delta)
    {
        pos.y += delta;
        eye.y += delta;
        lookat.y += delta;
        box.center.y += delta;
    }
    void dz(const float delta)
    {
        pos.z += delta;
        eye.z += delta;
        lookat.z += delta;
        box.center.z += delta;
    }
    void dX(const float3& vec)
    {
        pos += vec;
        eye += vec;
        lookat += vec;
        box.center += vec;
    }
    //@@todo: link with collidebox
};
#endif