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
#include "OptiX7Craft.h"
#include "random.h"
#include "helpers.h"
#include <cuda/helpers.h>

extern "C" {
__constant__ Params params;
}

extern "C" __global__ void __raygen__pinhole_camera()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const CameraData* camera = (CameraData*) optixGetSbtDataPointer();

    const unsigned int image_index = params.width * idx.y + idx.x;
    unsigned int       seed        = tea<16>( image_index, params.subframe_index );
    float3 result = make_float3(0);

    int i = params.samples_per_launch; //spp
    do {
        float2 subpixel_jitter =  make_float2(rnd(seed), rnd(seed));

        float2 d = ((make_float2(idx.x, idx.y) + subpixel_jitter) / make_float2(params.width, params.height)) * 2.f - 1.f;
        float3 ray_origin = camera->eye;
        float3 ray_direction = normalize(d.x*camera->U + d.y*camera->V + camera->W);

        
        SunPRD sun_prd;
        sun_prd.importance = 1.f;
        sun_prd.depth = 0;
        sun_prd.seed = seed;
        sun_prd.done = false;
        sun_prd.attenuation = make_float3(1.0f);
        sun_prd.emitted = make_float3(0.0f);
        sun_prd.countEmitted = true;
        sun_prd.radiance = make_float3(0.0f);


        unsigned int u0, u1;
        packPointer(&sun_prd, u0, u1);
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
        result += sun_prd.emitted;
        result += sun_prd.radiance * sun_prd.attenuation;

    }
    while (--i);

    float3         accum_color = result / static_cast<float>(params.samples_per_launch);
    if( params.subframe_index > 0 )
    {
        const float                 a = 1.0f / static_cast<float>(params.subframe_index + 1);
        const float3 accum_color_prev = make_float3(params.accum_buffer[image_index]);
        accum_color = lerp(accum_color_prev, accum_color, a);
    }

    params.accum_buffer[image_index] = make_float4(accum_color, 1.0f);
    params.frame_buffer[image_index] = make_color(accum_color);

}


