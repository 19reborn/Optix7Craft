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

//todo 内存回收！

#include <glad/glad.h> // Needs to be included before gl_interop

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <sampleConfig.h>

#include <sutil/Camera.h>
#include <sutil/Trackball.h>
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/Matrix.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>

#include <GLFW/glfw3.h>
#include <iomanip>
#include <cstring>

#include "optixWhitted.h"

#include <vector>
#include <string>
#include <algorithm>
#include <map>
using std::vector;
using std::string;
using std::map;

//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------

bool              resize_dirty  = false;
bool              minimized     = false;
map<char, bool>   key_value;
float camera_speed = 0.03f; 
int wscnt = 0, adcnt = 0;
// Camera state
sutil::Camera     camera;
sutil::Trackball  trackball;

// Mouse state
int32_t           mouse_button = -1;

const int         max_trace = 12;

//------------------------------------------------------------------------------
//
// Local types
// TODO: some of these should move to sutil or optix util header
//
//------------------------------------------------------------------------------

template <typename T>
struct Record
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT )

    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef Record<CameraData>      RayGenRecord;
typedef Record<MissData>        MissRecord;
typedef Record<HitGroupData>    HitGroupRecord;

struct WhittedState {
    OptixDeviceContext          context                   = 0;
    OptixTraversableHandle      gas_handle                = {};
    CUdeviceptr                 d_gas_output_buffer       = {};

    OptixModule                 geometry_module           = 0;
    OptixModule                 camera_module             = 0;
    OptixModule                 shading_module            = 0;
    OptixModule                 sphere_module             = 0;

    OptixProgramGroup           raygen_prog_group         = 0;
    OptixProgramGroup           radiance_miss_prog_group  = 0;
    OptixProgramGroup           occlusion_miss_prog_group = 0;
    OptixProgramGroup           radiance_glass_sphere_prog_group  = 0;
    OptixProgramGroup           occlusion_glass_sphere_prog_group = 0;
    OptixProgramGroup           radiance_metal_sphere_prog_group  = 0;
    OptixProgramGroup           occlusion_metal_sphere_prog_group = 0;
    OptixProgramGroup           radiance_floor_prog_group         = 0;
    OptixProgramGroup           occlusion_floor_prog_group        = 0;

    OptixPipeline               pipeline                  = 0;
    OptixPipelineCompileOptions pipeline_compile_options  = {};

    CUstream                    stream                    = 0;
    Params                      params;
    Params*                     d_params                  = nullptr;

    OptixShaderBindingTable     sbt                       = {};
};

//------------------------------------------------------------------------------
//
//  Model Classes
//
//------------------------------------------------------------------------------
class cModel {
public:
    static uint32_t OBJ_COUNT;

    cModel() {OBJ_COUNT++;}
    virtual string get_type() = 0;
    virtual void set_bound(float result[6]) = 0;
    virtual uint32_t get_input_flag() = 0;
    virtual void set_hitgroup(WhittedState& state, HitGroupRecord* hgr, int idx) = 0;
};

uint32_t cModel::OBJ_COUNT = 0;

class cSphere: public cModel {
public:
    GeometryData::Sphere args;

    cSphere(float3 c, float r) {
        std::cerr << "[INFO] A Sphere Generated.\n";
        args.center = c;
        args.radius = r;
    }

    string get_type() {return "Sphere";}
    void set_bound(float result[6]) override {
        auto *aabb = reinterpret_cast<OptixAabb*>(result);

        float3 m_min = args.center - args.radius;
        float3 m_max = args.center + args.radius;

        *aabb = {
                m_min.x, m_min.y, m_min.z,
                m_max.x, m_max.y, m_max.z
        };
    }
    uint32_t get_input_flag() override {
        return OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
    }
    void set_hitgroup(WhittedState& state, HitGroupRecord* hgr, int idx) override {
        OPTIX_CHECK( optixSbtRecordPackHeader(
                state.radiance_metal_sphere_prog_group,
                &hgr[idx] ) );
        hgr[idx].data.geometry.sphere = args;
        //todo 这个玩意应该也要能自定义，但是我们肯定不用球，所以不急
        //todo 另外，可以继续继承一些类，那些类能初始化一些特定的这些
        hgr[idx].data.shading.metal = {
                { 0.2f, 0.5f, 0.5f },   // Ka
                { 0.2f, 0.7f, 0.8f },   // Kd
                { 0.9f, 0.9f, 0.9f },   // Ks
                { 0.5f, 0.5f, 0.5f },   // Kr
                64,                     // phong_exp
        };
        OPTIX_CHECK( optixSbtRecordPackHeader(
                state.occlusion_metal_sphere_prog_group,
                &hgr[idx+1] ) );
        hgr[idx+1].data.geometry.sphere = args;
    }

};

class cSphereShell: public cModel {
public:
    SphereShell args;

    cSphereShell(float3 c, float r1, float r2) {
        std::cerr << "[INFO] A SphereShell Generated.\n";
        args.center = c;
        args.radius1 = r1;
        args.radius2 = r2;
    }
    string get_type() {return "SphereShell";}
    void set_bound(float result[6]) override {
        auto *aabb = reinterpret_cast<OptixAabb*>(result);

        float3 m_min = args.center - args.radius2;
        float3 m_max = args.center + args.radius2;

        *aabb = {
                m_min.x, m_min.y, m_min.z,
                m_max.x, m_max.y, m_max.z
        };
    }
    uint32_t get_input_flag() override {
        return OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;
    }
    void set_hitgroup(WhittedState& state, HitGroupRecord* hgr, int idx) override {
        OPTIX_CHECK( optixSbtRecordPackHeader(
                state.radiance_glass_sphere_prog_group,
                &hgr[idx] ) );
        hgr[idx].data.geometry.sphere_shell = args;
        hgr[idx].data.shading.glass = {
                1e-2f,                                  // importance_cutoff
                { 0.034f, 0.055f, 0.085f },             // cutoff_color
                3.0f,                                   // fresnel_exponent
                0.1f,                                   // fresnel_minimum
                1.0f,                                   // fresnel_maximum
                1.4f,                                   // refraction_index
                { 1.0f, 1.0f, 1.0f },                   // refraction_color
                { 1.0f, 1.0f, 1.0f },                   // reflection_color
                { logf(.83f), logf(.83f), logf(.83f) }, // extinction_constant
                { 0.6f, 0.6f, 0.6f },                   // shadow_attenuation
                10,                                     // refraction_maxdepth
                5                                       // reflection_maxdepth
        };
        OPTIX_CHECK( optixSbtRecordPackHeader(
                state.occlusion_glass_sphere_prog_group,
                &hgr[idx+1] ) );
        hgr[idx+1].data.geometry.sphere_shell = args;
        hgr[idx+1].data.shading.glass.shadow_attenuation = { 0.6f, 0.6f, 0.6f };
    }

};

class cRect: public cModel {
public:
    Parallelogram args;

    cRect(float3 v1, float3 v2, float3 anchor) {
        std::cerr << "[INFO] A Rect Generated.\n";
        args = { v1,v2,anchor };

    }
    string get_type() {return "Rect";}
    void set_bound(float result[6]) override {
        // v1 and v2 are scaled by 1./length^2.  Rescale back to normal for the bounds computation.
        const float3 tv1  = args.v1 / dot( args.v1, args.v1 );
        const float3 tv2  = args.v2 / dot( args.v2, args.v2 );
        const float3 p00  = args.anchor;
        const float3 p01  = args.anchor + tv1;
        const float3 p10  = args.anchor + tv2;
        const float3 p11  = args.anchor + tv1 + tv2;

        auto aabb = reinterpret_cast<OptixAabb*>(result);

        float3 m_min = fminf( fminf( p00, p01 ), fminf( p10, p11 ));
        float3 m_max = fmaxf( fmaxf( p00, p01 ), fmaxf( p10, p11 ));
        *aabb = {
                m_min.x, m_min.y, m_min.z,
                m_max.x, m_max.y, m_max.z
        };
    }
    uint32_t get_input_flag() override {
        return OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
    }
    void set_hitgroup(WhittedState& state, HitGroupRecord* hgr, int idx) override {
        OPTIX_CHECK( optixSbtRecordPackHeader(
                state.radiance_floor_prog_group,
                &hgr[idx] ) );
        hgr[idx].data.geometry.parallelogram = args;
        hgr[idx].data.shading.checker = {
                { 0.8f, 0.3f, 0.15f },      // Kd1
                { 0.9f, 0.85f, 0.05f },     // Kd2
                { 0.8f, 0.3f, 0.15f },      // Ka1
                { 0.9f, 0.85f, 0.05f },     // Ka2
                { 0.0f, 0.0f, 0.0f },       // Ks1
                { 0.0f, 0.0f, 0.0f },       // Ks2
                { 0.0f, 0.0f, 0.0f },       // Kr1
                { 0.0f, 0.0f, 0.0f },       // Kr2
                0.0f,                       // phong_exp1
                0.0f,                       // phong_exp2
                { 32.0f, 16.0f }            // inv_checker_size
        };
        OPTIX_CHECK( optixSbtRecordPackHeader(
                state.occlusion_floor_prog_group,
                &hgr[idx+1] ) );
        hgr[idx+1].data.geometry.parallelogram = args;

    }

};

vector<cModel*> modelLst;

//------------------------------------------------------------------------------
//
//  Geometry and Camera data
//
//------------------------------------------------------------------------------

// light
const BasicLight g_light = {
        make_float3( 60.0f, 40.0f, 0.0f ),   // pos
        make_float3( 1.0f, 1.0f, 1.0f )      // color
};

//------------------------------------------------------------------------------
//
// GLFW callbacks
//
//------------------------------------------------------------------------------

static void mouseButtonCallback( GLFWwindow* window, int button, int action, int mods )
{
    double xpos, ypos;
    glfwGetCursorPos( window, &xpos, &ypos );

    if( action == GLFW_PRESS )
    {
        mouse_button = button;
        trackball.startTracking(static_cast<int>( xpos ), static_cast<int>( ypos ));
    }
    else
    {
        mouse_button = -1;
    }
}


static void cursorPosCallback( GLFWwindow* window, double xpos, double ypos )
{
    Params* params = static_cast<Params*>( glfwGetWindowUserPointer( window ) );

    if( mouse_button == GLFW_MOUSE_BUTTON_LEFT )
    {
        trackball.setViewMode( sutil::Trackball::LookAtFixed );
        trackball.updateTracking( static_cast<int>( xpos ), static_cast<int>( ypos ), params->width, params->height );

    }
    else 
    {
        trackball.setViewMode( sutil::Trackball::EyeFixed );
        trackball.updateTracking( static_cast<int>( xpos ), static_cast<int>( ypos ), params->width, params->height );

    }
}


static void windowSizeCallback( GLFWwindow* window, int32_t res_x, int32_t res_y )
{
    // Keep rendering at the current resolution when the window is minimized.
    if( minimized )
        return;

    // Output dimensions must be at least 1 in both x and y.
    sutil::ensureMinimumSize( res_x, res_y );

    Params* params = static_cast<Params*>( glfwGetWindowUserPointer( window ) );
    params->width  = res_x;
    params->height = res_y;
    resize_dirty   = true;
}


static void windowIconifyCallback( GLFWwindow* window, int32_t iconified )
{
    minimized = ( iconified > 0 );
}


static void keyCallback( GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/ )
{

    if( action == GLFW_PRESS )
    {
        if( key == GLFW_KEY_Q ||
            key == GLFW_KEY_ESCAPE )
        {
            glfwSetWindowShouldClose( window, true );
        }
        if (key == GLFW_KEY_W) key_value['w'] = true, wscnt++;
        if (key == GLFW_KEY_A) key_value['a'] = true, adcnt++;
        if (key == GLFW_KEY_S) key_value['s'] = true, wscnt--;
        if (key == GLFW_KEY_D) key_value['d'] = true, adcnt--;
    }
    else if (action == GLFW_RELEASE)
    {
        if (key == GLFW_KEY_W) key_value['w'] = false, wscnt--;
        if (key == GLFW_KEY_A) key_value['a'] = false, adcnt--;
        if (key == GLFW_KEY_S) key_value['s'] = false, wscnt++;
        if (key == GLFW_KEY_D) key_value['d'] = false, adcnt++;
    }
    else if( key == GLFW_KEY_G )
    {
        // toggle UI draw
    }
}


static void scrollCallback( GLFWwindow* window, double xscroll, double yscroll )
{
    if (trackball.wheelEvent((int)yscroll))
    {
    }
}


//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

void printUsageAndExit( const char* argv0 )
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>      File for image output\n";
    std::cerr << "         --no-gl-interop             Disable GL interop for display\n";
    std::cerr << "         --dim=<width>x<height>      Set image dimensions; defaults to 768x768\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    exit( 0 );
}

void initLaunchParams( WhittedState& state )
{
    CUDA_CHECK( cudaMalloc(
            reinterpret_cast<void**>( &state.params.accum_buffer ),
            state.params.width*state.params.height*sizeof(float4)
    ) );
    state.params.frame_buffer = nullptr; // Will be set when output buffer is mapped

    state.params.subframe_index = 0u;

    state.params.light = g_light;
    state.params.ambient_light_color = make_float3( 0.4f, 0.4f, 0.4f );
    state.params.max_depth = max_trace;
    state.params.scene_epsilon = 1.e-4f;

    CUDA_CHECK( cudaStreamCreate( &state.stream ) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_params ), sizeof( Params ) ) );

    state.params.handle = state.gas_handle;
}

static void buildGas(
        const WhittedState &state,
        const OptixAccelBuildOptions &accel_options,
        const OptixBuildInput &build_input,
        OptixTraversableHandle &gas_handle,
        CUdeviceptr &d_gas_output_buffer
)
{
    OptixAccelBufferSizes gas_buffer_sizes;
    CUdeviceptr d_temp_buffer_gas;

    OPTIX_CHECK( optixAccelComputeMemoryUsage(
            state.context,
            &accel_options,
            &build_input,
            1,
            &gas_buffer_sizes));

    CUDA_CHECK( cudaMalloc(
            reinterpret_cast<void**>( &d_temp_buffer_gas ),
            gas_buffer_sizes.tempSizeInBytes));

    // non-compacted output and size of compacted GAS
    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t compactedSizeOffset = roundUp<size_t>( gas_buffer_sizes.outputSizeInBytes, 8ull );
    CUDA_CHECK( cudaMalloc(
            reinterpret_cast<void**>( &d_buffer_temp_output_gas_and_compacted_size ),
            compactedSizeOffset + 8
    ) );

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

    OPTIX_CHECK( optixAccelBuild(
            state.context,
            0,
            &accel_options,
            &build_input,
            1,
            d_temp_buffer_gas,
            gas_buffer_sizes.tempSizeInBytes,
            d_buffer_temp_output_gas_and_compacted_size,
            gas_buffer_sizes.outputSizeInBytes,
            &gas_handle,
            &emitProperty,
            1) );

    CUDA_CHECK( cudaFree( (void*)d_temp_buffer_gas ) );

    size_t compacted_gas_size;
    CUDA_CHECK( cudaMemcpy( &compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost ) );

    if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
    {
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_gas_output_buffer ), compacted_gas_size ) );

        // use handle as input and output
        OPTIX_CHECK( optixAccelCompact( state.context, 0, gas_handle, d_gas_output_buffer, compacted_gas_size, &gas_handle ) );

        CUDA_CHECK( cudaFree( (void*)d_buffer_temp_output_gas_and_compacted_size ) );
    }
    else
    {
        d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
}

void createGeometry( WhittedState &state ) {
    //
    // Build Custom Primitives
    //

    // Load AABB into device memory
    OptixAabb*  aabb = new OptixAabb[cModel::OBJ_COUNT];
    CUdeviceptr d_aabb;

    for(int i=0; i<cModel::OBJ_COUNT; i++) {
        modelLst[i]->set_bound(reinterpret_cast<float*>(&aabb[i]));
    }

    std::cerr << "[INFO] aabb size: " << cModel::OBJ_COUNT * sizeof( OptixAabb ) << std::endl;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_aabb
                            ), cModel::OBJ_COUNT * sizeof( OptixAabb ) ) );
    CUDA_CHECK( cudaMemcpy(
            reinterpret_cast<void*>( d_aabb ),
            aabb,                                       // notice: 这里原来是&aabb, 但它之前居然能正常运作！
            cModel::OBJ_COUNT * sizeof( OptixAabb ),
            cudaMemcpyHostToDevice
    ) );

    // Setup AABB build input
    uint32_t* aabb_input_flags = new uint32_t[cModel::OBJ_COUNT];
    for(int i=0; i<cModel::OBJ_COUNT; i++) {
        aabb_input_flags[i] = modelLst[i]->get_input_flag();
    }

    /* TODO: This API cannot control flags for different ray type */

    // originally 0, 1, 2
    uint32_t* sbt_index = new uint32_t[cModel::OBJ_COUNT];
    for(int i=0; i<cModel::OBJ_COUNT; i++)
        sbt_index[i] = i;

    CUdeviceptr    d_sbt_index;

    size_t size_sbt_index = cModel::OBJ_COUNT * sizeof(uint32_t);

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_sbt_index ), size_sbt_index ) );
    CUDA_CHECK( cudaMemcpy(
            reinterpret_cast<void*>( d_sbt_index ),
            sbt_index,
            size_sbt_index,
            cudaMemcpyHostToDevice ) );

    OptixBuildInput aabb_input = {};
    aabb_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    aabb_input.customPrimitiveArray.aabbBuffers   = &d_aabb;
    aabb_input.customPrimitiveArray.flags         = aabb_input_flags;
    aabb_input.customPrimitiveArray.numSbtRecords = cModel::OBJ_COUNT;
    aabb_input.customPrimitiveArray.numPrimitives = cModel::OBJ_COUNT;
    aabb_input.customPrimitiveArray.sbtIndexOffsetBuffer         = d_sbt_index;
    aabb_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes    = sizeof( uint32_t );
    aabb_input.customPrimitiveArray.primitiveIndexOffset         = 0;


    OptixAccelBuildOptions accel_options = {
            OPTIX_BUILD_FLAG_ALLOW_COMPACTION,  // buildFlags
            OPTIX_BUILD_OPERATION_BUILD         // operation
    };


    buildGas(
            state,
            accel_options,
            aabb_input,
            state.gas_handle,
            state.d_gas_output_buffer);

    CUDA_CHECK( cudaFree( (void*)d_aabb) );
}

void createModules( WhittedState &state )
{
    OptixModuleCompileOptions module_compile_options = {};

    char log[2048];
    size_t sizeof_log = sizeof(log);

    {
        size_t      inputSize = 0;
        const char* input     = sutil::getInputData( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "geometry.cu", inputSize );
        OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
                state.context,
                &module_compile_options,
                &state.pipeline_compile_options,
                input,
                inputSize,
                log,
                &sizeof_log,
                &state.geometry_module ) );
    }

    {
        size_t      inputSize = 0;
        const char* input     = sutil::getInputData( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "camera.cu", inputSize );
        OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
                state.context,
                &module_compile_options,
                &state.pipeline_compile_options,
                input,
                inputSize,
                log,
                &sizeof_log,
                &state.camera_module ) );
    }

    {
        size_t      inputSize = 0;
        const char* input     = sutil::getInputData( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "shading.cu", inputSize );
        OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
                state.context,
                &module_compile_options,
                &state.pipeline_compile_options,
                input,
                inputSize,
                log,
                &sizeof_log,
                &state.shading_module ) );
    }

    {
        size_t      inputSize = 0;
        const char* input     = sutil::getInputData( nullptr, nullptr, "sphere.cu", inputSize );
        OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
                state.context,
                &module_compile_options,
                &state.pipeline_compile_options,
                input,
                inputSize,
                log,
                &sizeof_log,
                &state.sphere_module ) );
    }
}

static void createCameraProgram( WhittedState &state, std::vector<OptixProgramGroup> &program_groups )
{
    OptixProgramGroup           cam_prog_group;
    OptixProgramGroupOptions    cam_prog_group_options = {};
    OptixProgramGroupDesc       cam_prog_group_desc = {};
    cam_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    cam_prog_group_desc.raygen.module = state.camera_module;
    cam_prog_group_desc.raygen.entryFunctionName = "__raygen__pinhole_camera";

    char    log[2048];
    size_t  sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
            state.context,
            &cam_prog_group_desc,
            1,
            &cam_prog_group_options,
            log,
            &sizeof_log,
            &cam_prog_group ) );

    program_groups.push_back(cam_prog_group);
    state.raygen_prog_group = cam_prog_group;
}

static void createGlassSphereProgram( WhittedState &state, std::vector<OptixProgramGroup> &program_groups )
{
    OptixProgramGroup           radiance_sphere_prog_group;
    OptixProgramGroupOptions    radiance_sphere_prog_group_options = {};
    OptixProgramGroupDesc       radiance_sphere_prog_group_desc = {};
    radiance_sphere_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    radiance_sphere_prog_group_desc.hitgroup.moduleIS            = state.geometry_module;
    radiance_sphere_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__sphere_shell";
    radiance_sphere_prog_group_desc.hitgroup.moduleCH            = state.shading_module;
    radiance_sphere_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__glass_radiance";
    radiance_sphere_prog_group_desc.hitgroup.moduleAH            = nullptr;
    radiance_sphere_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;

    char    log[2048];
    size_t  sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
            state.context,
            &radiance_sphere_prog_group_desc,
            1,
            &radiance_sphere_prog_group_options,
            log,
            &sizeof_log,
            &radiance_sphere_prog_group ) );

    program_groups.push_back(radiance_sphere_prog_group);
    state.radiance_glass_sphere_prog_group = radiance_sphere_prog_group;

    OptixProgramGroup           occlusion_sphere_prog_group;
    OptixProgramGroupOptions    occlusion_sphere_prog_group_options = {};
    OptixProgramGroupDesc       occlusion_sphere_prog_group_desc = {};
    occlusion_sphere_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    occlusion_sphere_prog_group_desc.hitgroup.moduleIS            = state.geometry_module;
    occlusion_sphere_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__sphere_shell";
    occlusion_sphere_prog_group_desc.hitgroup.moduleCH            = nullptr;
    occlusion_sphere_prog_group_desc.hitgroup.entryFunctionNameCH = nullptr;
    occlusion_sphere_prog_group_desc.hitgroup.moduleAH            = state.shading_module;
    occlusion_sphere_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__glass_occlusion";

    OPTIX_CHECK_LOG( optixProgramGroupCreate(
            state.context,
            &occlusion_sphere_prog_group_desc,
            1,
            &occlusion_sphere_prog_group_options,
            log,
            &sizeof_log,
            &occlusion_sphere_prog_group ) );

    program_groups.push_back(occlusion_sphere_prog_group);
    state.occlusion_glass_sphere_prog_group = occlusion_sphere_prog_group;
}

static void createMetalSphereProgram( WhittedState &state, std::vector<OptixProgramGroup> &program_groups )
{
    OptixProgramGroup           radiance_sphere_prog_group;
    OptixProgramGroupOptions    radiance_sphere_prog_group_options = {};
    OptixProgramGroupDesc       radiance_sphere_prog_group_desc = {};
    radiance_sphere_prog_group_desc.kind   = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
            radiance_sphere_prog_group_desc.hitgroup.moduleIS           = state.sphere_module;
    radiance_sphere_prog_group_desc.hitgroup.entryFunctionNameIS    = "__intersection__sphere";
    radiance_sphere_prog_group_desc.hitgroup.moduleCH               = state.shading_module;
    radiance_sphere_prog_group_desc.hitgroup.entryFunctionNameCH    = "__closesthit__metal_radiance";
    radiance_sphere_prog_group_desc.hitgroup.moduleAH               = nullptr;
    radiance_sphere_prog_group_desc.hitgroup.entryFunctionNameAH    = nullptr;

    char    log[2048];
    size_t  sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
            state.context,
            &radiance_sphere_prog_group_desc,
            1,
            &radiance_sphere_prog_group_options,
            log,
            &sizeof_log,
            &radiance_sphere_prog_group ) );

    program_groups.push_back(radiance_sphere_prog_group);
    state.radiance_metal_sphere_prog_group = radiance_sphere_prog_group;

    OptixProgramGroup           occlusion_sphere_prog_group;
    OptixProgramGroupOptions    occlusion_sphere_prog_group_options = {};
    OptixProgramGroupDesc       occlusion_sphere_prog_group_desc = {};
    occlusion_sphere_prog_group_desc.kind   = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
            occlusion_sphere_prog_group_desc.hitgroup.moduleIS           = state.sphere_module;
    occlusion_sphere_prog_group_desc.hitgroup.entryFunctionNameIS    = "__intersection__sphere";
    occlusion_sphere_prog_group_desc.hitgroup.moduleCH               = state.shading_module;
    occlusion_sphere_prog_group_desc.hitgroup.entryFunctionNameCH    = "__closesthit__full_occlusion";
    occlusion_sphere_prog_group_desc.hitgroup.moduleAH               = nullptr;
    occlusion_sphere_prog_group_desc.hitgroup.entryFunctionNameAH    = nullptr;

    OPTIX_CHECK_LOG( optixProgramGroupCreate(
            state.context,
            &occlusion_sphere_prog_group_desc,
            1,
            &occlusion_sphere_prog_group_options,
            log,
            &sizeof_log,
            &occlusion_sphere_prog_group ) );

    program_groups.push_back(occlusion_sphere_prog_group);
    state.occlusion_metal_sphere_prog_group = occlusion_sphere_prog_group;
}

static void createFloorProgram( WhittedState &state, std::vector<OptixProgramGroup> &program_groups )
{
    OptixProgramGroup           radiance_floor_prog_group;
    OptixProgramGroupOptions    radiance_floor_prog_group_options = {};
    OptixProgramGroupDesc       radiance_floor_prog_group_desc = {};
    radiance_floor_prog_group_desc.kind   = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    radiance_floor_prog_group_desc.hitgroup.moduleIS               = state.geometry_module;
    radiance_floor_prog_group_desc.hitgroup.entryFunctionNameIS    = "__intersection__parallelogram";
    radiance_floor_prog_group_desc.hitgroup.moduleCH               = state.shading_module;
    radiance_floor_prog_group_desc.hitgroup.entryFunctionNameCH    = "__closesthit__checker_radiance";
    radiance_floor_prog_group_desc.hitgroup.moduleAH               = nullptr;
    radiance_floor_prog_group_desc.hitgroup.entryFunctionNameAH    = nullptr;

    char    log[2048];
    size_t  sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
            state.context,
            &radiance_floor_prog_group_desc,
            1,
            &radiance_floor_prog_group_options,
            log,
            &sizeof_log,
            &radiance_floor_prog_group ) );

    program_groups.push_back(radiance_floor_prog_group);
    state.radiance_floor_prog_group = radiance_floor_prog_group;

    OptixProgramGroup           occlusion_floor_prog_group;
    OptixProgramGroupOptions    occlusion_floor_prog_group_options = {};
    OptixProgramGroupDesc       occlusion_floor_prog_group_desc = {};
    occlusion_floor_prog_group_desc.kind   = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    occlusion_floor_prog_group_desc.hitgroup.moduleIS               = state.geometry_module;
    occlusion_floor_prog_group_desc.hitgroup.entryFunctionNameIS    = "__intersection__parallelogram";
    occlusion_floor_prog_group_desc.hitgroup.moduleCH               = state.shading_module;
    occlusion_floor_prog_group_desc.hitgroup.entryFunctionNameCH    = "__closesthit__full_occlusion";
    occlusion_floor_prog_group_desc.hitgroup.moduleAH               = nullptr;
    occlusion_floor_prog_group_desc.hitgroup.entryFunctionNameAH    = nullptr;

    OPTIX_CHECK_LOG( optixProgramGroupCreate(
            state.context,
            &occlusion_floor_prog_group_desc,
            1,
            &occlusion_floor_prog_group_options,
            log,
            &sizeof_log,
            &occlusion_floor_prog_group ) );

    program_groups.push_back(occlusion_floor_prog_group);
    state.occlusion_floor_prog_group = occlusion_floor_prog_group;
}

static void createMissProgram( WhittedState &state, std::vector<OptixProgramGroup> &program_groups )
{
    OptixProgramGroupOptions    miss_prog_group_options = {};
    OptixProgramGroupDesc       miss_prog_group_desc = {};
    miss_prog_group_desc.kind   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module             = state.shading_module;
    miss_prog_group_desc.miss.entryFunctionName  = "__miss__constant_bg";

    char    log[2048];
    size_t  sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
            state.context,
            &miss_prog_group_desc,
            1,
            &miss_prog_group_options,
            log,
            &sizeof_log,
            &state.radiance_miss_prog_group ) );

    program_groups.push_back(state.radiance_miss_prog_group);

    miss_prog_group_desc.miss = {
            nullptr,    // module
            nullptr     // entryFunctionName
    };
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
            state.context,
            &miss_prog_group_desc,
            1,
            &miss_prog_group_options,
            log,
            &sizeof_log,
            &state.occlusion_miss_prog_group ) );

    program_groups.push_back(state.occlusion_miss_prog_group);
}

void createPipeline( WhittedState &state )
{
    std::vector<OptixProgramGroup> program_groups;

    state.pipeline_compile_options = {
            false,                                                  // usesMotionBlur
            OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,          // traversableGraphFlags
            5,    /* RadiancePRD uses 5 payloads */                 // numPayloadValues
            5,    /* Parallelogram intersection uses 5 attrs */     // numAttributeValues
            OPTIX_EXCEPTION_FLAG_NONE,                              // exceptionFlags
            "params"                                                // pipelineLaunchParamsVariableName
    };

    // Prepare program groups
    createModules( state );
    createCameraProgram( state, program_groups );
    createGlassSphereProgram( state, program_groups );
    createMetalSphereProgram( state, program_groups );
    createFloorProgram( state, program_groups );
    createMissProgram( state, program_groups );

    // Link program groups to pipeline
    OptixPipelineLinkOptions pipeline_link_options = {
            max_trace,                          // maxTraceDepth
            OPTIX_COMPILE_DEBUG_LEVEL_FULL      // debugLevel
    };
    char    log[2048];
    size_t  sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG( optixPipelineCreate(
            state.context,
            &state.pipeline_compile_options,
            &pipeline_link_options,
            program_groups.data(),
            static_cast<unsigned int>( program_groups.size() ),
            log,
            &sizeof_log,
            &state.pipeline ) );

    OptixStackSizes stack_sizes = {};
    for( auto& prog_group : program_groups )
    {
        OPTIX_CHECK( optixUtilAccumulateStackSizes( prog_group, &stack_sizes ) );
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK( optixUtilComputeStackSizes( &stack_sizes, max_trace,
                                             0,  // maxCCDepth
                                             0,  // maxDCDepth
                                             &direct_callable_stack_size_from_traversal,
                                             &direct_callable_stack_size_from_state, &continuation_stack_size ) );
    OPTIX_CHECK( optixPipelineSetStackSize( state.pipeline, direct_callable_stack_size_from_traversal,
                                            direct_callable_stack_size_from_state, continuation_stack_size,
                                            1  // maxTraversableDepth
    ) );
}

void syncCameraDataToSbt( WhittedState &state, const CameraData& camData )
{
    RayGenRecord rg_sbt;

    optixSbtRecordPackHeader( state.raygen_prog_group, &rg_sbt );
    rg_sbt.data = camData;

    CUDA_CHECK( cudaMemcpy(
            reinterpret_cast<void*>( state.sbt.raygenRecord ),
            &rg_sbt,
            sizeof( RayGenRecord ),
            cudaMemcpyHostToDevice
    ) );
}

void createSBT( WhittedState &state )
{
    // Raygen program record
    {
        CUdeviceptr d_raygen_record;
        size_t sizeof_raygen_record = sizeof( RayGenRecord );
        CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &d_raygen_record ),
                sizeof_raygen_record ) );

        state.sbt.raygenRecord = d_raygen_record;
    }

    // Miss program record
    {
        CUdeviceptr d_miss_record;
        size_t sizeof_miss_record = sizeof( MissRecord );
        CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &d_miss_record ),
                sizeof_miss_record*RAY_TYPE_COUNT ) );

        MissRecord ms_sbt[RAY_TYPE_COUNT];
        optixSbtRecordPackHeader( state.radiance_miss_prog_group, &ms_sbt[0] );
        optixSbtRecordPackHeader( state.occlusion_miss_prog_group, &ms_sbt[1] );
        ms_sbt[1].data = ms_sbt[0].data = { 0.34f, 0.55f, 0.85f };

        CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_miss_record ),
                ms_sbt,
                sizeof_miss_record*RAY_TYPE_COUNT,
                cudaMemcpyHostToDevice
        ) );

        state.sbt.missRecordBase          = d_miss_record;
        state.sbt.missRecordCount         = RAY_TYPE_COUNT;
        state.sbt.missRecordStrideInBytes = static_cast<uint32_t>( sizeof_miss_record );
    }

    // Hitgroup program record
    {
        size_t count_records = RAY_TYPE_COUNT * cModel::OBJ_COUNT;
        HitGroupRecord* hitgroup_records = new HitGroupRecord[count_records];

        // Note: Fill SBT record array the same order like AS is built.
        for(int i=0; i<count_records; i+=2) {
            modelLst[i/2]->set_hitgroup(state, hitgroup_records, i);
        }

        CUdeviceptr d_hitgroup_records;
        size_t      sizeof_hitgroup_record = sizeof( HitGroupRecord );
        CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &d_hitgroup_records ),
                sizeof_hitgroup_record*count_records
        ) );

        CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_hitgroup_records ),
                hitgroup_records,
                sizeof_hitgroup_record*count_records,
                cudaMemcpyHostToDevice
        ) );

        state.sbt.hitgroupRecordBase            = d_hitgroup_records;
        state.sbt.hitgroupRecordCount           = count_records;
        state.sbt.hitgroupRecordStrideInBytes   = static_cast<uint32_t>( sizeof_hitgroup_record );
    }
}

static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
              << message << "\n";
}

void createContext( WhittedState& state )
{
    // Initialize CUDA
    CUDA_CHECK( cudaFree( 0 ) );

    OptixDeviceContext context;
    CUcontext          cuCtx = 0;  // zero means take the current context
    OPTIX_CHECK( optixInit() );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &context_log_cb;
    options.logCallbackLevel          = 4;
    OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &context ) );

    state.context = context;
}

//
//
//

void initCameraState()
{
    camera.setEye( make_float3( 8.0f, 2.0f, -4.0f ) );
    camera.setLookat( make_float3( 4.0f, 2.3f, -4.0f ) );
    camera.setUp( make_float3( 0.0f, 1.0f, 0.0f ) );
    camera.setFovY( 60.0f );

    trackball.setCamera( &camera );
    trackball.setMoveSpeed( 10.0f );
    trackball.setReferenceFrame( make_float3( 1.0f, 0.0f, 0.0f ), make_float3( 0.0f, 0.0f, 1.0f ), make_float3( 0.0f, 1.0f, 0.0f ) );
    trackball.setGimbalLock(true);
}

void handleCameraUpdate( WhittedState &state )
{

    float3 direction(make_float3(0.0f));
    if (wscnt != 0 || adcnt != 0)
    {
        float3 camera_target_vector = camera.direction();
        float3 camera_normal_vector = cross(camera.up(), camera.direction());
        if (key_value['w']) direction += normalize(make_float3(camera_target_vector.x, 0, camera_target_vector.z));
        if (key_value['s']) direction -= normalize(make_float3(camera_target_vector.x, 0, camera_target_vector.z));
        if (key_value['a']) direction += normalize(make_float3(camera_normal_vector.x, 0, camera_normal_vector.z));
        if (key_value['d']) direction -= normalize(make_float3(camera_normal_vector.x, 0, camera_normal_vector.z));
        direction = normalize(direction);
    }
    
    camera.setAspectRatio( static_cast<float>( state.params.width ) / static_cast<float>( state.params.height ) );
    camera.setEye(camera.eye() + camera_speed * direction);
    camera.setLookat(camera.lookat() + camera_speed * direction);
    CameraData camData;
    camData.eye = camera.eye();
    camera.UVWFrame( camData.U, camData.V, camData.W );

    syncCameraDataToSbt(state, camData);
}

void handleResize( sutil::CUDAOutputBuffer<uchar4>& output_buffer, Params& params )
{
    if( !resize_dirty )
        return;
    resize_dirty = false;

    output_buffer.resize( params.width, params.height );

    // Realloc accumulation buffer
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( params.accum_buffer ) ) );
    CUDA_CHECK( cudaMalloc(
            reinterpret_cast<void**>( &params.accum_buffer ),
            params.width*params.height*sizeof(float4)
    ) );
}

void updateState( sutil::CUDAOutputBuffer<uchar4>& output_buffer, WhittedState &state )
{
    // Update params on device

    state.params.subframe_index = 0;

    handleCameraUpdate( state );
    handleResize( output_buffer, state.params );
}

void launchSubframe( sutil::CUDAOutputBuffer<uchar4>& output_buffer, WhittedState& state )
{

    // Launch
    uchar4* result_buffer_data = output_buffer.map();
    state.params.frame_buffer = result_buffer_data;
    CUDA_CHECK( cudaMemcpyAsync( reinterpret_cast<void*>( state.d_params ),
                                 &state.params,
                                 sizeof( Params ),
                                 cudaMemcpyHostToDevice,
                                 state.stream
    ) );

    OPTIX_CHECK( optixLaunch(
            state.pipeline,
            state.stream,
            reinterpret_cast<CUdeviceptr>( state.d_params ),
            sizeof( Params ),
            &state.sbt,
            state.params.width,  // launch width
            state.params.height, // launch height
            1                    // launch depth
    ) );
    output_buffer.unmap();
    CUDA_SYNC_CHECK();
}


void displaySubframe(
        sutil::CUDAOutputBuffer<uchar4>&  output_buffer,
        sutil::GLDisplay&                 gl_display,
        GLFWwindow*                       window )
{
    // Display
    int framebuf_res_x = 0;   // The display's resolution (could be HDPI res)
    int framebuf_res_y = 0;   //
    glfwGetFramebufferSize( window, &framebuf_res_x, &framebuf_res_y );
    gl_display.display(
            output_buffer.width(),
            output_buffer.height(),
            framebuf_res_x,
            framebuf_res_y,
            output_buffer.getPBO()
    );
}


void cleanupState( WhittedState& state )
{
    OPTIX_CHECK( optixPipelineDestroy     ( state.pipeline                ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.raygen_prog_group       ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.radiance_metal_sphere_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.occlusion_metal_sphere_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.radiance_glass_sphere_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.occlusion_glass_sphere_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.radiance_miss_prog_group         ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.radiance_floor_prog_group        ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.occlusion_floor_prog_group       ) );
    OPTIX_CHECK( optixModuleDestroy       ( state.shading_module          ) );
    OPTIX_CHECK( optixModuleDestroy       ( state.geometry_module         ) );
    OPTIX_CHECK( optixModuleDestroy       ( state.camera_module           ) );
    OPTIX_CHECK( optixModuleDestroy       ( state.sphere_module           ) );
    OPTIX_CHECK( optixDeviceContextDestroy( state.context                 ) );


    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.raygenRecord       ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.missRecordBase     ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.hitgroupRecordBase ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_gas_output_buffer    ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.params.accum_buffer    ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_params               ) ) );
}

int main( int argc, char* argv[] )
{
    WhittedState state;
    state.params.width  = 768;
    state.params.height = 768;
    sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;

    //
    // Parse command line options
    //
    std::string outfile;

    for( int i = 1; i < argc; ++i )
    {
        const std::string arg = argv[i];
        if( arg == "--help" || arg == "-h" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( arg == "--no-gl-interop" )
        {
            output_buffer_type = sutil::CUDAOutputBufferType::CUDA_DEVICE;
        }
        else if( arg == "--file" || arg == "-f" )
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );
            outfile = argv[++i];
        }
        else if( arg.substr( 0, 6 ) == "--dim=" )
        {
            const std::string dims_arg = arg.substr( 6 );
            int w, h;
            sutil::parseDimensions( dims_arg.c_str(), w, h );
            state.params.width  = w;
            state.params.height = h;
        }
        else
        {
            std::cerr << "Unknown option '" << argv[i] << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    try
    {
        //
        // Add basic models
        //
        modelLst.push_back(new cSphere({ 2.0f, 1.5f, -2.5f }, 1.0f));
        
        for(int i=1; i<=10; i++)
            modelLst.push_back(new cSphereShell({ 4.0f, 0.3f + 2.f*i, -4.0f }, 0.96f, 1.0f));

        modelLst.push_back(new cRect(
            make_float3( 32.0f, 0.0f, 0.0f ),
            make_float3( 0.0f, 0.0f, 16.0f ),
            make_float3( -16.0f, 0.01f, -8.0f )
        ));
        
        modelLst.push_back(new cSphere({ 6.0f, 1.5f, -2.5f }, 1.0f));

        initCameraState();

        initCameraState();
        //
        // Set up OptiX state
        //
        createContext  ( state );
        createGeometry  ( state );
        createPipeline ( state );
        createSBT      ( state );

        initLaunchParams( state );

        //
        // Render loop
        //
        if( outfile.empty() )
        {
            GLFWwindow* window = sutil::initUI( "optixWhitted", state.params.width, state.params.height );
            glfwSetMouseButtonCallback  ( window, mouseButtonCallback   );
            glfwSetCursorPosCallback    ( window, cursorPosCallback     );
            glfwSetWindowSizeCallback   ( window, windowSizeCallback    );
            glfwSetWindowIconifyCallback( window, windowIconifyCallback );
            glfwSetKeyCallback          ( window, keyCallback           );
            glfwSetScrollCallback       ( window, scrollCallback        );
            glfwSetWindowUserPointer    ( window, &state.params         );

            {
                // output_buffer needs to be destroyed before cleanupUI is called
                sutil::CUDAOutputBuffer<uchar4> output_buffer(
                        output_buffer_type,
                        state.params.width,
                        state.params.height
                );

                output_buffer.setStream( state.stream );
                sutil::GLDisplay gl_display;

                std::chrono::duration<double> state_update_time( 0.0 );
                std::chrono::duration<double> render_time( 0.0 );
                std::chrono::duration<double> display_time( 0.0 );

                do
                {
                    auto t0 = std::chrono::steady_clock::now();
                    glfwPollEvents();

                    updateState( output_buffer, state );
                    auto t1 = std::chrono::steady_clock::now();
                    state_update_time += t1 - t0;
                    t0 = t1;

                    launchSubframe( output_buffer, state );
                    t1 = std::chrono::steady_clock::now();
                    render_time += t1 - t0;
                    t0 = t1;

                    displaySubframe( output_buffer, gl_display, window );
                    t1 = std::chrono::steady_clock::now();
                    display_time += t1 - t0;

                    sutil::displayStats( state_update_time, render_time, display_time );

                    glfwSwapBuffers( window );

                    ++state.params.subframe_index;
                }
                while( !glfwWindowShouldClose( window ) );

            }
            sutil::cleanupUI( window );
        }
        else
        {
            if ( output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP )
            {
                sutil::initGLFW(); // For GL context
                sutil::initGL();
            }

            sutil::CUDAOutputBuffer<uchar4> output_buffer(
                    output_buffer_type,
                    state.params.width,
                    state.params.height
            );

            handleCameraUpdate( state );
            handleResize( output_buffer, state.params );
            launchSubframe( output_buffer, state );

            sutil::ImageBuffer buffer;
            buffer.data         = output_buffer.getHostPointer();
            buffer.width        = output_buffer.width();
            buffer.height       = output_buffer.height();
            buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
            sutil::saveImage( outfile.c_str(), buffer, false );

            if( output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP )
            {
                glfwTerminate();
            }
        }

        cleanupState( state );
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
