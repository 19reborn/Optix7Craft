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

#include <vector_types.h>
#include <optix_device.h>
#include "optixWhitted.h"
#include "random.h"
#include "helpers.h"
#include <cuda/helpers.h>

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

extern "C" __global__ void __raygen__pinhole_camera()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const CameraData* camera = (CameraData*) optixGetSbtDataPointer();

    const unsigned int image_index = params.width * idx.y + idx.x;
    unsigned int       seed        = tea<16>( image_index, params.subframe_index );
    float3 result = make_float3(0);
    // Subpixel jitter: send the ray through a different position inside the pixel each time,
    // to provide antialiasing. The center of each pixel is at fraction (0.5,0.5)
    float2 subpixel_jitter = params.subframe_index == 0 ?
        make_float2(0.5f, 0.5f) : make_float2(rnd( seed ), rnd( seed ));

    float2 d = ((make_float2(idx.x, idx.y) + subpixel_jitter) / make_float2(params.width, params.height)) * 2.f - 1.f;
    float3 ray_origin = camera->eye;
    float3 ray_direction = normalize(d.x*camera->U + d.y*camera->V + camera->W);
    /*
    RadiancePRD prd;
    prd.importance = 1.f;
    prd.depth = 0;

    optixTrace(
        params.handle,
        ray_origin,
        ray_direction,
        params.scene_epsilon,
        1e16f,
        0.0f,
        OptixVisibilityMask( 1 ),
        OPTIX_RAY_FLAG_NONE,
        RAY_TYPE_RADIANCE,
        RAY_TYPE_COUNT,
        RAY_TYPE_RADIANCE,
        float3_as_args(prd.result),
        reinterpret_cast<unsigned int&>(prd.importance),
        reinterpret_cast<unsigned int&>(prd.depth) );

    result += prd.result;
    */
   
    SunPRD sun_prd;
    sun_prd.importance = 1.f;
    sun_prd.depth = 0;
    sun_prd.seed = seed;
    sun_prd.done = false;
    sun_prd.attenuation = make_float3(1.0f);

    // light from a light source or miss program
    sun_prd.radiance = make_float3(0.0f);
    // next ray to be traced
    sun_prd.origin = make_float3(0.0f);
    sun_prd.direction = make_float3(0.0f);


    for (;;) {
        unsigned int u0, u1;
        packPointer(&sun_prd, u0, u1);
        //optixSetPayload_0(u0);
        //optixSetPayload_1(u1);
        optixTrace(
            params.handle,
            ray_origin,
            ray_direction,
            params.scene_epsilon,
            1e16f,
            0.0f,
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_NONE,
            RAY_TYPE_RADIANCE,
            RAY_TYPE_COUNT,
            RAY_TYPE_RADIANCE,
            u0,
            u1);

        result += sun_prd.radiance * sun_prd.attenuation;

        if (sun_prd.done) {
            break;
        }
        else if (sun_prd.depth >= 2) {
            result += sun_prd.attenuation * make_float3(0.2f,0.2f,0.2f);
            break;
        }

        sun_prd.depth++;

        // Update ray data for the next path segment
        ray_origin = sun_prd.origin;
        ray_direction = sun_prd.direction;
    }
    
    float4 acc_val = params.accum_buffer[image_index];
    if( params.subframe_index > 0 )
    {
        acc_val = lerp( acc_val, make_float4( result, 0.f), 1.0f / static_cast<float>( params.subframe_index+1 ) );
    }
    else
    {
        acc_val = make_float4(result, 0.f);
    }
    params.frame_buffer[image_index] = make_color(tonemap(make_float3(acc_val)));
    params.accum_buffer[image_index] = acc_val;

}


/*
static __forceinline__ __device__ RadiancePRD* getPRD()
{
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<RadiancePRD*>(unpackPointer(u0, u1));
}
*/


static __forceinline__ __device__ void setPayloadOcclusion(bool occluded)
{
    optixSetPayload_0(static_cast<unsigned int>(occluded));
}
