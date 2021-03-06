#include <vector_types.h>

#include <sutil/vec_math.h>
#include <sutil/Matrix.h>
#include <sutil/sutilapi.h>

#include <optix_types.h>
#include <optix.h>

#include "sunsky.hpp"
#include <cuda/GeometryData.h>

enum RayType
{
    RAY_TYPE_RADIANCE  = 0,
    RAY_TYPE_OCCLUSION = 1,
    RAY_TYPE_COUNT
};


struct BasicLight
{
    int id;
    float3  pos;
    float3  color;

}; 

struct DirectionalLight
{

  float3 direction;
  float radius;
  float3 v0;  // basis vectors for area sampling
  float3 v1; 
  float3 color;
  int casts_shadow;
};


struct Params
{
    unsigned int subframe_index;
    float4*      accum_buffer;
    uchar4*      frame_buffer;
    unsigned int width;
    unsigned int height;
    unsigned int samples_per_launch;
    unsigned int num_lights_sample;
    
    BufferView<BasicLight> point_light;             // TODO: make light list
    int point_light_sum;
    DirectionalLight sun;
    PreethamSunSky sky;
    float3       ambient_light_color;
    int          max_depth;
    float        scene_epsilon;
    float        circle;
    float        game_time;

    OptixTraversableHandle handle;

};


struct CameraData
{
    float3       eye;
    float3       U;
    float3       V;
    float3       W;
};


struct MissData
{
    float3 bg_color;
    cudaTextureObject_t  morning_map;
    cudaTextureObject_t  noon_map;
    cudaTextureObject_t  night_map;
};


enum SphereShellHitType {
    HIT_OUTSIDE_FROM_OUTSIDE = 1u << 0,
    HIT_OUTSIDE_FROM_INSIDE  = 1u << 1,
    HIT_INSIDE_FROM_OUTSIDE  = 1u << 2,
    HIT_INSIDE_FROM_INSIDE   = 1u << 3
};

enum CubeHitType {
    HIT_FROM_INSIDE = 1u << 0,
    HIT_FROM_OUTSIDE = 1u << 1,
};


struct SphereShell
{
	float3 	center;
	float 	radius1;
	float 	radius2;
};

struct Cube
{
    float3 	center;
    float3 	size;
};

struct CubeShell {
    float3  center;
    float3  size1;
    float3  size2;
};


struct Parallelogram
{
    Parallelogram() = default;
    Parallelogram( float3 v1, float3 v2, float3 anchor ):
    v1( v1 ), v2( v2 ), anchor( anchor )
    {
        float3 normal = normalize(cross( v1, v2 ));
        float d = dot( normal, anchor );
        this->v1 *= 1.0f / dot( v1, v1 );
        this->v2 *= 1.0f / dot( v2, v2 );
        plane = make_float4(normal, d);
    }
    float4	plane;
    float3 	v1;
    float3 	v2;
    float3 	anchor;
};


struct Phong
{
    float3 Ka;
    float3 Kd;
    float3 Ks;
    float3 Kr;
    float  phong_exp;
};

struct Texture
{
    float3 Ka;
    float3 Kd;
    float3 Ks;
    float3 Kr;
    float  phong_exp;
};

struct Glass
{
    float  importance_cutoff;
    float3 cutoff_color;
    float  fresnel_exponent;
    float  fresnel_minimum;
    float  fresnel_maximum;
    float  refraction_index;
    float3 refraction_color;
    float3 reflection_color;
    float3 extinction_constant;
    float3 shadow_attenuation;
    int    refraction_maxdepth;
    int    reflection_maxdepth;
};


struct CheckerPhong
{
    float3 Kd1, Kd2;
    float3 Ka1, Ka2;
    float3 Ks1, Ks2;
    float3 Kr1, Kr2;
    float  phong_exp1, phong_exp2;
    float2 inv_checker_size;
};

struct Water
{
    float3 Ka;
    float3 Kd;
    float3 Ks;
    float3 Kr;
    float  phong_exp;
    float  importance_cutoff;
    int    refraction_maxdepth;
    float refractivity_n;
    float transparency;
};

struct HitGroupData
{


    union
    {
        GeometryData::Sphere sphere;
        SphereShell          sphere_shell;
        Parallelogram        parallelogram;
        Cube                 cube;
    } geometry;

    union
    {
        Phong           metal;
        Glass           glass;
        Water           water;
        CheckerPhong    checker;
    } shading;    

    bool  has_diffuse;
    cudaTextureObject_t diffuse_map_y_up;
    cudaTextureObject_t diffuse_map_y_down;
    cudaTextureObject_t diffuse_map_x_up;
    cudaTextureObject_t diffuse_map_x_down;
    cudaTextureObject_t diffuse_map_z_up;
    cudaTextureObject_t diffuse_map_z_down;
    bool  has_normal;
    cudaTextureObject_t  normal_map;
    bool  has_roughness;
    cudaTextureObject_t  roughness_map;
};


struct RadiancePRD
{
    float3 result;
    float  importance;
    int    depth;

};

enum LightType {
    Point,
    Directional
};

struct SunPRD {
    
    int depth;
    unsigned int seed;

    // shading state
    bool done;
    float  importance;
    float3 attenuation;
    float3 radiance;
    float3 emitted;
    bool countEmitted;

    LightType type;
};

struct OcclusionPRD
{
    float3 attenuation;
};

struct texture_map {
    ~texture_map()
    {
        if (pixel) delete[] pixel;
    }

    unsigned int * pixel{ nullptr };
    int2     resolution{ -1 };
    
    cudaTextureObject_t textureObject = 0;
};

enum cube_face
{
    x_up,
    x_down,
    y_up,
    y_down,
    z_up,
    z_down
};