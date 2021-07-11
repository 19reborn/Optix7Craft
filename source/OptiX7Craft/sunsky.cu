#include <vector_types.h>
#include <optix_device.h>
#include <optix.h>
#include "OptiX7Craft.h"
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
    const float3 ray_dir = optixGetWorldRayDirection();
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_hit = ray_orig + ray_dir;
    const float3 orig = make_float3(ray_orig.x, ray_orig.y - 0.6f, ray_orig.z);
    float radius = 1.0f;

    float3 texcoord = normalize(ray_hit-orig)/2;
    float circle = params.circle;
    float game_time = fmod(params.game_time,circle);
    if (game_time >= circle /2.02) {
        float3 skybox = make_float3(tex2D<float4>(sbt_data->night_map, texcoord.x + 0.5f, texcoord.z + 0.5f));
        prd->radiance = skybox * params.ambient_light_color * 2.0f;
    }
    else if (game_time >= circle / 2.2) {
        float3 skybox = make_float3(tex2D<float4>(sbt_data->night_map, texcoord.x + 0.5f, texcoord.z + 0.5f));
        float3 color1 = skybox * params.ambient_light_color * 2.0f;
        skybox = make_float3(tex2D<float4>(sbt_data->noon_map, texcoord.x + 0.5f, texcoord.z + 0.5f));
        float3 color2 = skybox * params.ambient_light_color * 0.2f + tonemap(querySkyModel(show_sun, texcoord));
        prd->radiance = lerp(color2, color1, (game_time - circle / 2.2) / (circle / 2.02 - circle / 2.2));
    }
    else if (game_time <=  circle / 4 && game_time >= circle/20) {
        float3 skybox = make_float3(tex2D<float4>(sbt_data->morning_map, texcoord.x + 0.5f, texcoord.z + 0.5f));
        prd->radiance = skybox;
    }
    else if (game_time < circle / 20) {
        float3 skybox = make_float3(tex2D<float4>(sbt_data->morning_map, texcoord.x + 0.5f, texcoord.z + 0.5f));
        float3 color1 = skybox;
        skybox = make_float3(tex2D<float4>(sbt_data->night_map, texcoord.x + 0.5f, texcoord.z + 0.5f));
        float3 color2 = skybox * params.ambient_light_color * 2.0f;
        prd->radiance = lerp(color2, color1, game_time / (circle / 20));
    }
    else if (game_time < circle / 3.5) {
        float3 skybox = make_float3(tex2D<float4>(sbt_data->morning_map, texcoord.x + 0.5f, texcoord.z + 0.5f));
        float3 color1 = skybox;
        skybox = make_float3(tex2D<float4>(sbt_data->noon_map, texcoord.x + 0.5f, texcoord.z + 0.5f));
        float3 color2 = skybox * params.ambient_light_color * 0.2f + tonemap(querySkyModel(show_sun, texcoord));
        prd->radiance = lerp(color1, color2, (game_time-circle/4) / (circle /3.5 - circle/4 ));
    }
    else {
        float3 skybox = make_float3(tex2D<float4>(sbt_data->noon_map, texcoord.x + 0.5f, texcoord.z + 0.5f));
        prd->radiance = skybox * params.ambient_light_color * 0.2f + tonemap(querySkyModel(show_sun, ray_dir));
    }
    prd->done = true;
    unsigned int u0, u1;
    packPointer(&prd, u0, u1);
    optixSetPayload_0(u0);
    optixSetPayload_1(u1);
}

