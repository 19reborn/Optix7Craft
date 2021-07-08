/* 
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <vector_types.h>
#include <optix_device.h>
#include <optix.h>
#include "optixWhitted.h"
#include "helpers.h"

extern "C" {
__constant__ Params params;
}

__inline__ __device__ float3 tonemap(const float3 in)
{
    // hard coded exposure for sun/sky
    const float exposure = 1.0f / 30.0f;
    float3 x = exposure * in;

    // "filmic" map from a GDC talk by John Hable.  This includes 1/gamma.
    x = fmaxf(x - make_float3(0.004f), make_float3(0.0f));
    float3 ret = (x * (6.2f * x + make_float3(.5f))) / (x * (6.2f * x + make_float3(1.7f)) + make_float3(0.06f));

    return ret;
}
static __device__ __inline__ float3 querySkyModel( bool CEL, const float3& direction )
{
  PreethamSunSky sky = params.sky;
  float3 overcast_sky_color = make_float3( 0.0f );
  float3 sunlit_sky_color   = make_float3( 0.0f );

  // Preetham skylight model
  if( sky.m_overcast < 1.0f ) {
    float3 ray_direction = direction;
    if( CEL && dot( ray_direction, sky.m_sun_dir) > 94.0f / sqrtf( 94.0f*94.0f + 0.45f*0.45f) ) {
      sunlit_sky_color = sky.m_sun_color;
    } else {
      float inv_dir_dot_up = 1.f / dot( ray_direction, sky.m_up);
      if(inv_dir_dot_up < 0.f) {
        ray_direction = reflect(ray_direction, sky.m_up);
        inv_dir_dot_up = -inv_dir_dot_up;
      }

      float gamma = dot(sky.m_sun_dir, ray_direction);
      float acos_gamma = acosf(gamma);
      float3 A =  sky.m_c1 * inv_dir_dot_up;
      float3 B =  sky.m_c3 * acos_gamma;
      float3 color_Yxy = ( make_float3( 1.0f ) + sky.m_c0*make_float3( expf( A.x ),expf( A.y ),expf( A.z ) ) ) *
        ( make_float3( 1.0f ) + sky.m_c2*make_float3( expf( B.x ),expf( B.y ),expf( B.z ) ) + sky.m_c4*gamma*gamma );
      color_Yxy *= sky.m_inv_divisor_Yxy;

      color_Yxy.y = 0.33f + 1.2f * ( color_Yxy.y - 0.33f ); // Pump up chromaticity a bit
      color_Yxy.z = 0.33f + 1.2f * ( color_Yxy.z - 0.33f ); //
      float3 color_XYZ = sky.Yxy2XYZ( color_Yxy );
      sunlit_sky_color = sky.XYZ2rgb( color_XYZ ); 
      sunlit_sky_color /= 1000.0f; // We are choosing to return kilo-candellas / meter^2
    }
  }

  // CIE standard overcast sky model
  float Y =  15.0f;
  overcast_sky_color = make_float3( ( 1.0f + 2.0f * fabsf( direction.y ) ) / 3.0f * Y );

  // return linear combo of the two
  return lerp( sunlit_sky_color, overcast_sky_color, sky.m_overcast );
}

extern "C" __global__ void __miss__bg()
{
    const MissData* sbt_data = (MissData*)optixGetSbtDataPointer();
    SunPRD *prd = getPRD<SunPRD>();
    const bool show_sun = (prd->depth == 0);
    const float3 ray_dir = optixGetWorldRayDirection();;
    prd->radiance = ray_dir.y <= -0.0f ? sbt_data->bg_color : tonemap(querySkyModel( show_sun, ray_dir));
    //prd->color = tonemap(prd->radiance * prd->attenuation);
    prd->done = true;
    unsigned int u0, u1;
    packPointer(&prd, u0, u1);
    optixSetPayload_0(u0);
    optixSetPayload_1(u1);
}

