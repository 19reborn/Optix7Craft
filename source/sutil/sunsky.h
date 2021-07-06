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

#pragma once
#include <sutil/Matrix.h>
#include <sutil/sutilapi.h>
#include <optix.h>


#define M_PI       3.14159265358979323846   // pi
//------------------------------------------------------------------------------
//
// Implements the Preetham analytic sun/sky model ( Preetham, SIGGRAPH 99 )
//
//------------------------------------------------------------------------------

namespace sutil {


class PreethamSunSky
{
public:
    SUTILAPI  void init() {
        m_up = make_float3(0.0f, 1.0f, 0.0f);
        m_sun_theta=static_cast<float>(M_PI);
        m_sun_phi = 0.0f;
        m_turbidity = 2.0f;
        m_overcast = 0.0f;
        m_dirty = true;
    }
    SUTILAPI void setSunTheta(float sun_theta) { m_sun_theta = sun_theta; m_dirty = true; }
    SUTILAPI void setSunPhi(float sun_phi) { m_sun_phi = sun_phi;     m_dirty = true; }
    SUTILAPI void setTurbidity(float turbidity) { m_turbidity = turbidity; m_dirty = true; }

    SUTILAPI void setUpDir(const float3& up) { m_up = up; m_dirty = true; }
    SUTILAPI void setOvercast(float overcast) { m_overcast = overcast; }

    SUTILAPI float  getSunTheta() { return m_sun_theta; }
    SUTILAPI float  getSunPhi() { return m_sun_phi; }
    SUTILAPI float  getTurbidity() { return m_turbidity; }

    SUTILAPI float  getOvercast() { return m_overcast; }
    SUTILAPI float3 getUpDir() { return m_up; }
    SUTILAPI float3 getSunDir() { preprocess(); return m_sun_dir; }


    // Query the sun color at current sun position and air turbidity ( kilo-cd / m^2 )
    float3  sunColor()   {
        preprocess();

        // optical mass
        const float cos_sun_theta = cos(m_sun_theta);
        const float m = 1.0f / (cos_sun_theta + 0.15f * powf(93.885f - rad2deg(m_sun_theta), -1.253f));

        float results[38];
        for (int i = 0; i < 38; ++i) {
            results[i] = data[i].sun_spectral_radiance * 10000.0f // convert from 1/cm^2 to 1/m^2;
                / 1000.0f; // convert from micrometer to nanometer

            results[i] *= calculateAbsorption(m_sun_theta, m, data[i].wavelength, m_turbidity, data[i].k_o, data[i].k_wa);
        }


        float X = 0.0f, Y = 0.0f, Z = 0.0f;
        for (int i = 0; i < 38; ++i) {
            X += results[i] * cie_table[i][1] * 10.0f;
            Y += results[i] * cie_table[i][2] * 10.0f;
            Z += results[i] * cie_table[i][3] * 10.0f;
        }
        float3 result = XYZ2rgb(683.0f * make_float3(X, Y, Z)) / 1000.0f; // return result in kcd/m^2

        return result;
    }

    // Query the sky color in a given direction ( kilo-cd / m^2 )
    float3  skyColor(const float3& direction, bool CEL = false)  {
        preprocess();

        float3 overcast_sky_color = make_float3(0.0f, 0.0f, 0.0f);
        float3 sunlit_sky_color = make_float3(0.0f, 0.0f, 0.0f);

        // Preetham skylight model
        if (m_overcast < 1.0f) {
            float3 ray_direction = direction;
            if (CEL && dot(ray_direction, m_sun_dir) > 94.0f / sqrtf(94.0f * 94.0f + 0.45f * 0.45f)) {
                sunlit_sky_color = m_sun_color;
            }
            else {
                float inv_dir_dot_up = 1.f / dot(direction, m_up);
                if (inv_dir_dot_up < 0.f) {
                    ray_direction = reflect(ray_direction, m_up);
                    inv_dir_dot_up = -inv_dir_dot_up;
                }

                float gamma = dot(m_sun_dir, ray_direction);
                float acos_gamma = acos(gamma);
                float3 A = m_c1 * inv_dir_dot_up;
                float3 B = m_c3 * acos_gamma;
                float3 color_Yxy = (make_float3(1.0f) + m_c0 * make_float3(expf(A.x), expf(A.y), expf(A.z))) *
                    (make_float3(1.0f) + m_c2 * make_float3(expf(B.x), expf(B.y), expf(B.z)) + m_c4 * gamma * gamma);
                color_Yxy *= m_inv_divisor_Yxy;

                float3 color_XYZ = Yxy2XYZ(color_Yxy);
                sunlit_sky_color = XYZ2rgb(color_XYZ);
                sunlit_sky_color /= 1000.0f; // We are choosing to return kilo-candellas / meter^2
            }
        }

        // CIE standard overcast sky model
        float Y = 15.0f;
        overcast_sky_color = make_float3((1.0f + 2.0f * fabsf(direction.y)) / 3.0f * Y);

        // return linear combo of the two
        return lerp(sunlit_sky_color, overcast_sky_color, m_overcast);
    }

    // Sample the solid angle subtended by the sun at its current position
     float3 sampleSun()const;

  // Set precomputed Preetham model variables on the given context:
  //   c[0-4]          : 
  //   inv_divisor_Yxy :
  //   sun_dir         :
  //   sun_color       :
  //   overcast        :
  //   up              :



  void  preprocess(){
        if (!m_dirty) return;

        m_dirty = false;


        m_c0 = make_float3(0.1787f * m_turbidity - 1.4630f,
            -0.0193f * m_turbidity - 0.2592f,
            -0.0167f * m_turbidity - 0.2608f);

        m_c1 = make_float3(-0.3554f * m_turbidity + 0.4275f,
            -0.0665f * m_turbidity + 0.0008f,
            -0.0950f * m_turbidity + 0.0092f);

        m_c2 = make_float3(-0.0227f * m_turbidity + 5.3251f,
            -0.0004f * m_turbidity + 0.2125f,
            -0.0079f * m_turbidity + 0.2102f);

        m_c3 = make_float3(0.1206f * m_turbidity - 2.5771f,
            -0.0641f * m_turbidity - 0.8989f,
            -0.0441f * m_turbidity - 1.6537f);

        m_c4 = make_float3(-0.0670f * m_turbidity + 0.3703f,
            -0.0033f * m_turbidity + 0.0452f,
            -0.0109f * m_turbidity + 0.0529f);

        const float sun_theta_2 = m_sun_theta * m_sun_theta;
        const float sun_theta_3 = sun_theta_2 * m_sun_theta;

        const float xi = (4.0f / 9.0f - m_turbidity / 120.0f) *
            (static_cast<float>(M_PI) - 2.0f * m_sun_theta);

        float3 zenith;
        // Preetham paper is in kilocandellas -- we want candellas
        zenith.x = ((4.0453f * m_turbidity - 4.9710f) * tan(xi) - 0.2155f * m_turbidity + 2.4192f) * 1000.0f;
        zenith.y = m_turbidity * m_turbidity * (0.00166f * sun_theta_3 - 0.00375f * sun_theta_2 + 0.00209f * m_sun_theta) +
            m_turbidity * (-0.02903f * sun_theta_3 + 0.06377f * sun_theta_2 - 0.03202f * m_sun_theta + 0.00394f) +
            (0.11693f * sun_theta_3 - 0.21196f * sun_theta_2 + 0.06052f * m_sun_theta + 0.25886f);
        zenith.z = m_turbidity * m_turbidity * (0.00275f * sun_theta_3 - 0.00610f * sun_theta_2 + 0.00317f * m_sun_theta) +
            m_turbidity * (-0.04214f * sun_theta_3 + 0.08970f * sun_theta_2 - 0.04153f * m_sun_theta + 0.00516f) +
            (0.15346f * sun_theta_3 - 0.26756f * sun_theta_2 + 0.06670f * m_sun_theta + 0.26688f);


        float cos_sun_theta = cosf(m_sun_theta);

        float3 divisor_Yxy = (1.0f + m_c0 * expf(m_c1)) *
            (1.0f + m_c2 * expf(m_c3 * m_sun_theta) + m_c4 * cos_sun_theta * cos_sun_theta);

        m_inv_divisor_Yxy = zenith / divisor_Yxy;


        // 
        // Direct sunlight
        //
        m_sun_color = sunColor();

        float sin_sun_theta = sinf(m_sun_theta);
        m_sun_dir = make_float3(cosf(m_sun_phi) * sin_sun_theta,
            sinf(m_sun_phi) * sin_sun_theta,
            cosf(m_sun_theta));
        //Onb onb(m_up);
        //onb.inverse_transform(m_sun_dir);
    }
  float3 calculateSunColor();


  // Represents one entry from table 2 in the paper
  struct Datum  
  {
    float wavelength;
    float sun_spectral_radiance;
    float k_o;
    float k_wa;
  };
  
  static const float cie_table[38][4];          // CIE spectral sensitivy curves
  static const Datum data[38];                  // Table2


  // Calculate absorption for a given wavelength of direct sunlight
  float calculateAbsorption( float sun_theta, // Sun angle from zenith
                                    float m,         // Optical mass of atmosphere
                                    float lambda,    // light wavelength
                                    float turbidity, // atmospheric turbidity
                                    float k_o,       // atten coeff for ozone
                                    float k_wa )    // atten coeff for h2o vapor
      {
        float alpha = 1.3f;                             // wavelength exponent
        float beta = 0.04608f * turbidity - 0.04586f;  // turbidity coefficient
        float ell = 0.35f;                            // ozone at NTP (cm)
        float w = 2.0f;                             // precipitable water vapor (cm)

        float rayleigh_air = expf(-0.008735f * m * powf(lambda, -4.08f));
        float aerosol = expf(-beta * m * powf(lambda, -alpha));
        float ozone = k_o > 0.0f ? expf(-k_o * ell * m) : 1.0f;
        float water_vapor = k_wa > 0.0f ? expf(-0.2385f * k_wa * w * m / powf(1.0f + 20.07f * k_wa * w * m, 0.45f)) : 1.0f;

        return rayleigh_air * aerosol * ozone * water_vapor;
    }

  // Unit conversion helpers
  float3 XYZ2rgb( const float3& xyz );
  float3 Yxy2XYZ( const float3& Yxy );
  float  rad2deg( float rads );

  // Input parameters
  float  m_sun_theta;
  float  m_sun_phi;
  float  m_turbidity;
  float  m_overcast;
  float3 m_up;
  float3 m_sun_color;
  float3 m_sun_dir;

  // Precomputation results
  bool   m_dirty;
  float3 m_c0;
  float3 m_c1;
  float3 m_c2;
  float3 m_c3;
  float3 m_c4;
  float3 m_inv_divisor_Yxy;
};



  
inline float PreethamSunSky::rad2deg( float rads )
{
  return rads * 180.0f / static_cast<float>( M_PI );
}


inline float3 PreethamSunSky::Yxy2XYZ( const float3& Yxy )
{
  // Avoid division by zero in the transformation.
  if( Yxy.z < 1e-4 )
    return make_float3( 0.0f, 0.0f, 0.0f );

  return make_float3(  Yxy.y * ( Yxy.x / Yxy.z ),
                              Yxy.x,
                              ( 1.0f - Yxy.y - Yxy.z ) * ( Yxy.x / Yxy.z ) );
}


inline float3 PreethamSunSky::XYZ2rgb( const float3& xyz)
{
  const float R = dot( xyz, make_float3(  3.2410f, -1.5374f, -0.4986f ) );
  const float G = dot( xyz, make_float3( -0.9692f,  1.8760f,  0.0416f ) );
  const float B = dot( xyz, make_float3(  0.0556f, -0.2040f,  1.0570f ) );
  return make_float3( R, G, B );
}

} // namespace
