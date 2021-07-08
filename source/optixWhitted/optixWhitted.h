//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <cuda/GeometryData.h>

#include <vector_types.h>

#include <sutil/vec_math.h>
#include <sutil/Matrix.h>
#include <sutil/sutilapi.h>

#include <optix_types.h>
#include <optix.h>

#include "sunsky.hpp"

enum RayType
{
    RAY_TYPE_RADIANCE  = 0,
    RAY_TYPE_OCCLUSION = 1,
    RAY_TYPE_COUNT
};


struct BasicLight
{
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

    BasicLight   light;                 // TODO: make light list
    DirectionalLight sun;
    PreethamSunSky sky;
    float3       ambient_light_color;
    int          max_depth;
    float        scene_epsilon;

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
};


enum SphereShellHitType {
    HIT_OUTSIDE_FROM_OUTSIDE = 1u << 0,
    HIT_OUTSIDE_FROM_INSIDE  = 1u << 1,
    HIT_INSIDE_FROM_OUTSIDE  = 1u << 2,
    HIT_INSIDE_FROM_INSIDE   = 1u << 3
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
    float3 origin;
    float3 direction;

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
    
    cudaTextureObject_t textureObject;
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