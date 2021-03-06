#include <glad/glad.h> // Needs to be included before gl_interop

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <sampleConfig.h>

#include <sutil/sutilapi.h>

#include <sutil/Camera.h>
#include <sutil/Trackball.h>
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/Matrix.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>
#define STB_IMAGE_IMPLEMENTATION
#include <sutil/stb_image.h>


#include <GLFW/glfw3.h>
// ImGui
#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>

#include <iomanip>
#include <cstring>

#include "collideBox.h"
#include "OptiX7Craft.h"
#include "random.h"
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <fstream>
using std::vector;
using std::string;
using std::unordered_map;

//--------------------------------------------------------------------------- ---
//
// Settings
//
//------------------------------------------------------------------------------

//Particles settings
bool              isParticle = true;
//Sun height
bool              renewShadowOnTime = false;

//cursor mode
bool              isNoCursor = true;
//--------------------------------------------------------------------------- ---
//
// Globals
//
//------------------------------------------------------------------------------

bool              resize_dirty  = false;
bool              minimized     = false;
float lastframe = 0.f;
float deltatime = 0.f;
float circle = 360.0f;
float game_time;


//Keyboard mapping
unordered_map<char, bool>   key_value;
int wscnt = 0, adcnt = 0, sccnt = 0;
bool sprint = false;


// Camera state
float camera_speed = 1.5f;
sutil::Camera     camera;
sutil::Trackball  trackball;
bool switchcam = true; //Whenever you wanna change printer control, give this bool a TRUE value


// Texture 
std::vector<texture_map*>      texture_list;
std::unordered_map<std::string, size_t> textures;
std::vector<cudaArray_t>         textureArrays;

// Mouse state
int32_t           mouse_button = -1;

const int         max_trace = 10;

// Model state
bool              model_need_update = false;

enum BlockSize {
    NORM,
    THIN,
    BS_SIZE
} CurSize;

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
    OptixModule                 sunsky_module             = 0;

    OptixProgramGroup           raygen_prog_group         = 0;
    OptixProgramGroup           radiance_miss_prog_group  = 0;
    OptixProgramGroup           occlusion_miss_prog_group = 0;
    OptixProgramGroup           radiance_glass_sphere_prog_group = 0;
    OptixProgramGroup           occlusion_glass_sphere_prog_group = 0;
    OptixProgramGroup           radiance_texture_sphere_prog_group  = 0;
    OptixProgramGroup           occlusion_texture_sphere_prog_group = 0;
    OptixProgramGroup           radiance_texture_cube_prog_group = 0;
    OptixProgramGroup           occlusion_texture_cube_prog_group = 0;
    OptixProgramGroup           radiance_glass_cube_prog_group = 0;
    OptixProgramGroup           occlusion_glass_cube_prog_group = 0;
    OptixProgramGroup           radiance_water_cube_prog_group = 0;
    OptixProgramGroup           occlusion_water_cube_prog_group = 0;
    OptixProgramGroup           radiance_floor_prog_group         = 0;
    OptixProgramGroup           occlusion_floor_prog_group        = 0;

    OptixPipeline               pipeline                  = 0;
    OptixPipelineCompileOptions pipeline_compile_options  = {};

    CUstream                    stream                    = 0;
    Params                      params;
    Params*                     d_params                  = nullptr;

    OptixShaderBindingTable     sbt                       = {};
};

// ??????cuda????????????
CUdeviceptr    d_sbt_index;
bool d_sbt_index_allocated = false;
CUdeviceptr d_raygen_record;
bool d_raygen_record_allocated = false;
CUdeviceptr d_miss_record;
bool d_miss_record_allocated = false;
CUdeviceptr d_hitgroup_records;
bool d_hitgroup_records_allocated = false;
bool first_launch = true;

//------------------------------------------------------------------------------
//
//  Geometry Helper Functions
//
//------------------------------------------------------------------------------
inline float pow2(float f) {
    return f*f;
}

inline float calc_distance(float3 a, float3 b) {
    return sqrt(pow2(a.x - b.x) + pow2(a.y - b.y) + pow2(a.z - b.z));
}
inline float cceil(float f)
{
    if (f == ceil(f)) return f + 1.f;
    return ceil(f);
}
inline float ffloor(float f )
{
    if (f == floor(f)) return f - 1.f;
    return floor(f);
}
inline float fsign(float f)
{
    if (f > 0) return 1.f;
    return -1.f;
}
inline float3 f3ceil(float3& a)
{
    return make_float3(cceil(a.x), cceil(a.y), cceil(a.z));
}
inline float3 f3floor(float3& a)
{
    return make_float3(ffloor(a.x), ffloor(a.y), ffloor(a.z));
}
inline float3 f3abs(float3& a)
{
    return make_float3(fabs(a.x), fabs(a.y), fabs(a.z));
}
float3 nearCeil(float3& a,float3& vec)
{
    float3 tar;
    if (vec.x > 0) tar.x = cceil(a.x);
    else tar.x = ffloor(a.x);

    if (vec.y > 0) tar.y = cceil(a.y);
    else tar.y = ffloor(a.y);

    if (vec.z > 0) tar.z = cceil(a.z);
    else tar.z = ffloor(a.z);

    if (vec.x == 0 && vec.y == 0)
    {
        return make_float3(a.x, a.y, tar.z);
    }
    else if (vec.x == 0 && vec.z == 0)
    {
        return make_float3(a.x, tar.y, a.z);
    }
    else if (vec.y == 0 && vec.z == 0)
    {
        return make_float3(tar.x, a.y, a.z);
    }
    else if (vec.x == 0)
    {
        if (fabs(vec.y * (a.z - tar.z)) > fabs(vec.z * (a.y - tar.y)))
        {
            return make_float3(a.x,
                             tar.y,
                             a.z + vec.z * fabs((tar.y - a.y) / vec.y));
        }
        else {
            return make_float3(a.x,
                a.y + vec.y * fabs((tar.z - a.z) / vec.z),
                tar.z);
        }
    }
    else if (vec.y == 0)
    {
        if (fabs(vec.x * (a.z - tar.z)) > fabs(vec.z * (a.x - tar.x)))
        {
            return make_float3(tar.x,
                a.y,
                a.z + vec.z * fabs((tar.x - a.x) / vec.x));
        }
        else {
            return make_float3(a.x + vec.x * fabs((tar.z - a.z) / vec.z),
                a.y,
                tar.z);
        }
    }
    else if (vec.z == 0)
    {
        if (fabs(vec.x * (a.y - tar.y)) > fabs(vec.y * (a.x - tar.x)))
        {
            return make_float3(tar.x,
                a.y + vec.y * fabs((tar.x - a.x) / vec.x),
                a.z);
        }
        else {
            return make_float3(a.x + vec.x * fabs((tar.y - a.y) / vec.y),
                tar.y,
                a.z);
        }
    }
    else {
        float3 ti = f3abs((tar - a) / vec);
        if (ti.x <= ti.y && ti.x <= ti.z)
        {
            return make_float3(tar.x,
                a.y + vec.y * fabs((tar.x - a.x) / vec.x),
                a.z + vec.z * fabs((tar.x - a.x) / vec.x));
        }
        else if (ti.y <= ti.x && ti.y <= ti.z)
        {
            return make_float3(
                a.x + vec.x * fabs((tar.y - a.y) / vec.y),
                tar.y,
                a.z + vec.z * fabs((tar.y - a.y) / vec.y));
        }
        else {
            return make_float3(
                a.x + vec.x * fabs((tar.z - a.z) / vec.z),
                a.y + vec.y * fabs((tar.z - a.z) / vec.z),
                tar.z);
        }
    }
}
float sunAngleScaling(float f)
{
    return (float)fmod(f, 2 * M_PI);
}
int generateIntScaling(int numberGen)
{
    if (numberGen == 0)
    {
        return 1;
    }
    return 0;
}
//------------------------------------------------------------------------------
//
//  Model Classes and Functions
//
//------------------------------------------------------------------------------
enum ModelTexture { // ????????????get_texture_name
    NONE = 0,
    WOOD,
    PLANK,
    BRICK,
    DIRT,
    GRASS,
    IRON,
    GLASS,
    WATER,
    BARK,
    LEAF,
    GRAVEL,
    GOLD,
    CHINA,
    STEEL,
    ROCK,
    MT_SIZE // ????????????????????????????????????
};
ModelTexture curTexture = NONE;
string get_texture_name(ModelTexture tex_id) {
    switch (tex_id) {
        case NONE: return "NONE";   // ???????????????
        case WOOD: return "WOOD";
        case PLANK: return "PLANK";
        case BRICK: return "BRICK";
        case DIRT: return "DIRT";
        case GRASS: return "GRASS";
        case IRON: return "IRON";
        case GLASS: return "GLASS";
        case WATER: return "WATER";
        case BARK: return "BARK";
        case LEAF: return "LEAF";
        case GRAVEL: return "GRAVEL";
        case GOLD: return "GOLD";
        case CHINA: return "CHINA";
        case STEEL: return "STEEL";
        case ROCK: return "ROCK";
        default: return "ERROR";
    }
}

// light
std::vector<BasicLight> g_light;

class cModel {
public:
    static uint32_t OBJ_COUNT;
    int ID;
    bool collidable;    // ??????????????????
    CollideBox collideBox;
    ModelTexture texture_id;

    explicit cModel(const CollideBox& cb, ModelTexture tex_id): 
        collideBox(cb), texture_id(tex_id) {
        ID = ++OBJ_COUNT;
        set_map_modelAt();
    }
    virtual ~cModel() {
        OBJ_COUNT--;
        clear_map_modelAt();
    }
    virtual string get_type() = 0;
    virtual void set_bound(float result[6]) = 0;
    virtual uint32_t get_input_flag() = 0;
    virtual void set_hitgroup(WhittedState& state, HitGroupRecord* hgr, int idx) = 0;
    virtual float3 get_center() {return {0, 0, 0};}
    virtual float get_horizontal_size() { return 0.f; };
    CollideBox& get_collideBox() {return collideBox;}
    void set_map_modelAt();
    void clear_map_modelAt();
    virtual void move_to(float3 pos) = 0;
    void move_delta(float3 delta) { move_to(collideBox.center + delta); }
};

uint32_t cModel::OBJ_COUNT = 0;

struct HashFunc_float3 {  
    std::size_t operator() (const float3 &key) const {  
        using std::size_t;  
        using std::hash;  
  
        return ((hash<float>()(key.x)  
            ^ (hash<float>()(key.y) << 1)) >> 1)  
            ^ (hash<float>()(key.z) << 1);  
    }  
};  
struct EqualKey_float3 {  
    bool operator () (const float3 &lhs, const float3 &rhs) const {  
        return lhs.x  == rhs.x  
            && lhs.y  == rhs.y  
            && lhs.z  == rhs.z;  
    }  
};  
unordered_map<float3, cModel*, HashFunc_float3, EqualKey_float3> modelAt;

void cModel::set_map_modelAt() {
    int xl = (int)floor(collideBox.center.x - collideBox.size.x);
    int xr = (int)floor(collideBox.center.x + collideBox.size.x);
    int yl = (int)floor(collideBox.center.y - collideBox.size.y);
    int yr = (int)floor(collideBox.center.y + collideBox.size.y);
    int zl = (int)floor(collideBox.center.z - collideBox.size.z);
    int zr = (int)floor(collideBox.center.z + collideBox.size.z);
    // ????????????
    for(int i=xl; i<xr; i++) {
        for(int j=yl; j<yr; j++) {
            for(int k=zl; k<zr; k++) {
                //todo ???????????????????????????????????????(
                if(modelAt.count(make_float3((float)i, (float)j, (float)k)) && modelAt[make_float3((float)i, (float)j, (float)k)] != NULL) {
                    std::cerr << "[WARNING] (" << i << ", " << j << ", " << k << ") is not empty!\n";
                }
                modelAt[make_float3((float)i, (float)j, (float)k)] = this;
            }
        }
    }
}

void cModel::clear_map_modelAt() {
    int xl = (int)floor(collideBox.center.x - collideBox.size.x);
    int xr = (int)floor(collideBox.center.x + collideBox.size.x);
    int yl = (int)floor(collideBox.center.y - collideBox.size.y);
    int yr = (int)floor(collideBox.center.y + collideBox.size.y);
    int zl = (int)floor(collideBox.center.z - collideBox.size.z);
    int zr = (int)floor(collideBox.center.z + collideBox.size.z);
    // ????????????
    for(int i=xl; i<xr; i++) {
        for(int j=yl; j<yr; j++) {
            for(int k=zl; k<zr; k++) {
                modelAt[make_float3((float)i, (float)j, (float)k)] = NULL;
            }
        }
    }
}

// ??????cube?????????
template<class T>
void set_hitgroup_cube_general(WhittedState& state, HitGroupRecord* hgr, int idx, T* pmodel) {
    if(pmodel == NULL || pmodel->get_type() != "Cube") {
        std::cerr << "[WARNING] Wrong set_hitgroup call!\n";
        return;
    }
    ModelTexture texture_id = pmodel->texture_id;
    // ??????????????????????????????

    if (texture_id == NONE) {
        OPTIX_CHECK(optixSbtRecordPackHeader(
            state.radiance_texture_cube_prog_group,
            &hgr[idx]));
    }
    else if (texture_id == GLASS) {
        OPTIX_CHECK(optixSbtRecordPackHeader(
            state.radiance_glass_cube_prog_group,
            &hgr[idx]));
    } 
    else if (texture_id == WATER) {
        OPTIX_CHECK(optixSbtRecordPackHeader(
            state.radiance_water_cube_prog_group,
            &hgr[idx]));
    }
    else {    // ????????????????????????else?????????
        OPTIX_CHECK(optixSbtRecordPackHeader(
                state.radiance_texture_cube_prog_group,
                &hgr[idx]));
    }

    hgr[idx].data.geometry.cube = pmodel->args;

    // ?????????????????????
    if (texture_id == NONE) {
        hgr[idx].data.shading.metal = {
                { 0.2f, 0.5f, 0.5f },   // Ka
                { 0.1f, 0.2f, 0.4f },   // Kd   // ????????????????????????
                { 0.9f, 0.9f, 0.9f },   // Ks
                { 0.9f, 0.9f, 0.9f },   // Kr
                64,                     // phong_exp
        };        
        hgr[idx].data.has_diffuse = false;
        hgr[idx].data.has_normal = false;
        hgr[idx].data.has_roughness = false;
    } else if (texture_id == GLASS) {
        hgr[idx].data.shading.glass = {
                1e-2f,                                  // importance_cutoff
                { 0.034f, 0.055f, 0.085f },             // cutoff_color
                8.0f,                                   // fresnel_exponent(??????????????????
                0.5f,                                   // fresnel_minimum
                1.0f,                                   // fresnel_maximum
                1.4f,                                   // refraction_index
                { 1.0f, 1.0f, 1.0f },                   // refraction_color
                { 1.0f, 1.0f, 1.0f },                   // reflection_color
                { logf(.83f), logf(.83f), logf(.83f) }, // extinction_constant
                { 0.6f, 0.6f, 0.6f },                   // shadow_attenuation
                10,                                     // refraction_maxdepth
                5                                       // reflection_maxdepth
        };
    }
    else if (texture_id == WATER) {
        hgr[idx].data.shading.water = {
                { 0.2f, 0.5f, 0.5f },   // Ka
                { 0.65f, 0.8f, 0.9f },   // Kd   // ????????????????????????
                { 0.8f, 0.7f, 0.8f },   // Ks
                { 0.6f, 0.6f, 0.6f },   // Kr
                64,                     // phong_exp
                0.001f,                 // importance_cutoff
                10,                     // refraction_maxdepth
                1.333f,                    // refractivity_n ?????????
                0.8                    // transparency ?????????
        };
    }
    else {
        //?????????????????????????????????ka(ambient), ks(specular), kr(reflection).
        hgr[idx].data.shading.metal = {
                { 0.2f, 0.5f, 0.5f },   // Ka
                { 0.7f, 0.7f, 0.7f },   // Kd   // ????????????????????????
                { 0.2f, 0.2f, 0.2f },   // Ks
                { 0.0f, 0.0f, 0.0f },   // Kr 
                1                      // phong_exp
        };
        if(texture_id == IRON || texture_id == GOLD || texture_id == STEEL) {
            hgr[idx].data.shading.metal.Kr = {0.6f, 0.6f, 0.6f};
        }
        hgr[idx].data.has_diffuse = true;
        
        // ?????????????????????
        hgr[idx].data.diffuse_map_y_up = texture_list[textures[ get_texture_name(texture_id) + "_diffuse" ]]->textureObject;
        hgr[idx].data.diffuse_map_y_down = texture_list[textures[ get_texture_name(texture_id) + "_diffuse" ]]->textureObject;
        hgr[idx].data.diffuse_map_x_up = texture_list[textures[ get_texture_name(texture_id) + "_diffuse" ]]->textureObject;
        hgr[idx].data.diffuse_map_x_down = texture_list[textures[ get_texture_name(texture_id) + "_diffuse" ]]->textureObject;
        hgr[idx].data.diffuse_map_z_up = texture_list[textures[ get_texture_name(texture_id) + "_diffuse" ]]->textureObject;
        hgr[idx].data.diffuse_map_z_down = texture_list[textures[ get_texture_name(texture_id) + "_diffuse" ]]->textureObject;

        if(texture_id == GRASS) {   // ???????????????y_up?????????????????????DIRT
            hgr[idx].data.diffuse_map_y_down = texture_list[textures[ get_texture_name(DIRT) + "_diffuse" ]]->textureObject;
            hgr[idx].data.diffuse_map_x_up = texture_list[textures[ "GRASS_side_diffuse" ]]->textureObject;
            hgr[idx].data.diffuse_map_x_down = texture_list[textures[ "GRASS_side_diffuse"]]->textureObject;
            hgr[idx].data.diffuse_map_z_up = texture_list[textures[ "GRASS_side_diffuse" ]]->textureObject;
            hgr[idx].data.diffuse_map_z_down = texture_list[textures[ "GRASS_side_diffuse" ]]->textureObject;
        } else if(texture_id == BARK) { // ??????up???down?????????
            hgr[idx].data.diffuse_map_y_up = texture_list[textures[ "BARK_top_diffuse" ]]->textureObject;
            hgr[idx].data.diffuse_map_y_down = texture_list[textures[ "BARK_top_diffuse" ]]->textureObject;
        }

        if(texture_id == IRON 
        || texture_id == GRASS 
        || texture_id == BARK
        || texture_id == GOLD
        || texture_id == STEEL) {
            hgr[idx].data.has_normal = false;
        } else {
            hgr[idx].data.has_normal = true;
        }
        hgr[idx].data.normal_map = texture_list[textures[ get_texture_name(texture_id) + "_normal" ]]->textureObject;
        
        if(texture_id == BARK
        || texture_id == GRASS) {
            hgr[idx].data.has_roughness = false;
        } else {
            hgr[idx].data.has_roughness = true;
        }

        hgr[idx].data.roughness_map = texture_list[textures[ get_texture_name(texture_id) + "_roughness" ]]->textureObject;
    }
    
    if (texture_id == GLASS) {
        OPTIX_CHECK(optixSbtRecordPackHeader(
            state.occlusion_glass_cube_prog_group,
            &hgr[idx + 1]));
        hgr[idx + 1].data.shading.glass.shadow_attenuation = { 0.6f, 0.6f, 0.6f };
    }
    else if (texture_id == WATER) {
        OPTIX_CHECK(optixSbtRecordPackHeader(
            state.occlusion_water_cube_prog_group,
            &hgr[idx + 1]));
    }
    else {    // ????????????
        OPTIX_CHECK(optixSbtRecordPackHeader(
            state.occlusion_texture_cube_prog_group,
            &hgr[idx + 1]));
    }

    hgr[idx + 1].data.geometry.cube = pmodel->args;

}

class cSphere: public cModel {
public:
    GeometryData::Sphere args;

    explicit cSphere(float3 c, float r, ModelTexture tex_id=NONE): 
        cModel(CollideBox(c, {r, r, r}), tex_id) {
        args.center = c;
        args.radius = r;
        collidable = true;
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
                state.radiance_texture_sphere_prog_group,
                &hgr[idx] ) );
        hgr[idx].data.geometry.sphere = args;

        hgr[idx].data.shading.metal = {
                { 0.2f, 0.5f, 0.5f },   // Ka
                { 0.2f, 0.7f, 0.8f },   // Kd
                { 0.9f, 0.9f, 0.9f },   // Ks
                { 0.5f, 0.5f, 0.5f },   // Kr
                64,                     // phong_exp
        };
        OPTIX_CHECK( optixSbtRecordPackHeader(
                state.occlusion_texture_sphere_prog_group,
                &hgr[idx+1] ) );
        hgr[idx+1].data.geometry.sphere = args;
    }
    float3 get_center() override {
        return args.center;
    } 
    float get_horizontal_size() override {
        return args.radius;
    }
    void move_to(float3 pos) override {
        args.center = pos;
        clear_map_modelAt();
        collideBox.center = pos;
        set_map_modelAt();
    }
};

class cSphereShell: public cModel {
public:
    SphereShell args;

    explicit cSphereShell(float3 c, float r1, float r2, ModelTexture tex_id=NONE): 
        cModel(CollideBox(c, {0.5f, 0.5f, 0.5f}), tex_id) {
        args.center = c;
        args.radius1 = r1;
        args.radius2 = r2;
        collidable = true;
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
                5.0f,                                   // fresnel_exponent
                0.1f,                                   // fresnel_minimum
                1.0f,                                   // fresnel_maximum
                1.4f,                                   // refraction_index
                { 1.0f, 1.0f, 1.0f },                   // refraction_color
                { 0.5f, 0.5f, 0.5f },                   // reflection_color
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
    float3 get_center() override {
        return args.center;
    } 
    float get_horizontal_size() override {
        return args.radius2;
    }
    void move_to(float3 pos) override {
        args.center = pos;
        clear_map_modelAt();
        collideBox.center = pos;
        set_map_modelAt();
    }
};

class cCube : public cModel {
public:
    Cube args;

    explicit cCube(float3 c, float s, ModelTexture tex_id=NONE): 
        cModel(CollideBox(c, {ceil(2*s)/2.f, ceil(2*s)/2.f, ceil(2*s)/2.f}), tex_id) {
        args.center = c;
        args.size = {s, s, s};
        collidable = true;
    }

    explicit cCube(float3 c, float3 s, ModelTexture tex_id=NONE): 
        cModel(CollideBox(c, { ceil(s.x * 2) / 2, ceil(s.y * 2) / 2, ceil(s.z * 2) / 2 }), tex_id) {
        args.center = c;
        args.size = s;
        collidable = true;
    }

    string get_type() { return "Cube"; }
    void set_bound(float result[6]) override {
        auto* aabb = reinterpret_cast<OptixAabb*>(result);

        float3 m_min = args.center - args.size;
        float3 m_max = args.center + args.size;

        *aabb = {
                m_min.x, m_min.y, m_min.z,
                m_max.x, m_max.y, m_max.z
        };
    }
    uint32_t get_input_flag() override {
        return OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
    }
    void set_hitgroup(WhittedState& state, HitGroupRecord* hgr, int idx) override {
        set_hitgroup_cube_general(state, hgr, idx, this);
    }
    float3 get_center() override {
        return args.center;
    } 
    float get_horizontal_size() override {
        return std::max(args.size.x, args.size.z);
    }
    void move_to(float3 pos) override {
        args.center = pos;
        clear_map_modelAt();
        collideBox.center = pos;
        set_map_modelAt();
    }
};

class cCubeShell : public cModel {
public:
    CubeShell args;

    explicit cCubeShell(float3 c, float s1, float s2, ModelTexture tex_id = NONE) :
        cModel(CollideBox(c, { s2, s2, s2 }), tex_id) {
        args.center = c;
        args.size1 = { s1, s1, s1 };
        args.size2 = { s2, s2, s2 };
        collidable = true;
    }

    explicit cCubeShell(float3 c, float3 s1, float3 s2, ModelTexture tex_id) :
        cModel(CollideBox(c, s2), tex_id) {
        args.center = c;
        args.size1 = s1;
        args.size2 = s2;
        collidable = true;
    }

    string get_type() { return "CubeShell"; }
    void set_bound(float result[6]) override {
        auto* aabb = reinterpret_cast<OptixAabb*>(result);

        float3 m_min = args.center - args.size2;
        float3 m_max = args.center + args.size2;

        *aabb = {
                m_min.x, m_min.y, m_min.z,
                m_max.x, m_max.y, m_max.z
        };
    }
    uint32_t get_input_flag() override {
        return OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
    }
    void set_hitgroup(WhittedState& state, HitGroupRecord* hgr, int idx) override {
        // ?????????????????????program???????????????
        // set_hitgroup_cube_general(state, hgr, idx, this);
    }
    float3 get_center() override {
        return args.center;
    }
    float get_horizontal_size() override {
        return std::max(args.size2.x, args.size2.z);
    }
    void move_to(float3 pos) override {
        args.center = pos;
        clear_map_modelAt();
        collideBox.center = pos;
        set_map_modelAt();
    }
};

 class cRect: public cModel {    // ??????????????????????????????????????????????????????
public:
    Parallelogram args;

    explicit cRect(float3 v1, float3 v2, float3 anchor, ModelTexture tex_id=NONE):
        cModel(CollideBox(anchor, {1.f, 1.f, 1.f}) , tex_id) {
        args = {v1, v2, anchor};
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
    void move_to(float3 pos) {}
};

 enum LightColor {
     COLD = 0,
     WARM,
     WHITE,
     LC_SIZE
 };
LightColor curLightColor = COLD;

class cLightSphere: public cSphereShell {
public:
    static int LIGHTID;
    int lightnum;

    explicit cLightSphere(float3 pos, LightColor lc, float radius, ModelTexture tex_id=NONE):
    cSphereShell(pos, radius-.1f, radius, tex_id) {
        BasicLight bl;
        bl.pos = pos;
        if (lc == COLD) {
            bl.color = { 0.f, 0.5f, 1.f };
        } else if (lc == WARM) {
            bl.color = { 1.f, 0.5f, 0.f };
        } else if (lc == WHITE) {
            bl.color = { 1.f, 1.f, 1.f };
        }
        bl.id = ++LIGHTID;
        g_light.push_back(bl);
        lightnum = bl.id;
    }

    string get_type() {
        return "LightSphere";
    }
};
int cLightSphere::LIGHTID = 0;

vector<cModel*> modelLst;

//Interation Variables
cModel* intersectBlock;
float3 intersectPoint = make_float3(0.f, 114514.1919810f, 0.f);
bool istargeted = false;


bool get_model_at(float3 pos, cModel*& pmodel) {
    float3 key = pos;
    key.x = floor(key.x);
    key.y = floor(key.y);
    key.z = floor(key.z);
    
    if(modelAt.count(key) && modelAt[key] != NULL) {
        pmodel = modelAt[key];
        return true;
    }

    return false;
}

void set_bound_zero(float result[6]) {
    auto* aabb = reinterpret_cast<OptixAabb*>(result);
    *aabb = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
}

//------------------------------------------------------------------------------
//
//  Texture loading Functions
//
//------------------------------------------------------------------------------

void load_texture(std::string file_name, const std::string & name) {
    std::string relaPath = "Textures/" + file_name;
    std::string textureFilename(sutil::sampleDataFilePath(relaPath.c_str()));
    // std::cerr << "[INFO] path: " << textureFilename << std::endl;
    int2 res;
    int   comp;
    unsigned char* image = stbi_load(textureFilename.c_str(),
        &res.x, &res.y, &comp, STBI_rgb_alpha);
    if (image) {
        texture_map* texture = new texture_map;
        texture->resolution = res;
        texture->pixel = (uint32_t*)image;

        /* iw - actually, it seems that stbi loads the pictures
           mirrored along the y axis - mirror them here */
        for (int y = 0; y < res.y / 2; y++) {
            uint32_t* line_y = texture->pixel + y * res.x;
            uint32_t* mirrored_y = texture->pixel + (res.y - 1 - y) * res.x;
            // int mirror_y = res.y - 1 - y;    // ????????????
            for (int x = 0; x < res.x; x++) {
                std::swap(line_y[x], mirrored_y[x]);
            }
        }

        texture_list.push_back(texture);
        textures[name] = texture_list.size() - 1;

    }
    else {
        std::cout
            << "Could not load texture from " << file_name << "!"
            << std::endl;
    }
}

void load_texture_integrated(std::string file_prefix, ModelTexture tex) {
    load_texture(string(file_prefix + "_1K_Color.jpg"), string(get_texture_name(tex) + "_diffuse"));
    load_texture(string(file_prefix + "_1K_Normal.jpg"), string(get_texture_name(tex) + "_normal"));
    load_texture(string(file_prefix + "_1K_Displacement.jpg"), string(get_texture_name(tex) + "_roughness"));
}

// --------------------------------------------- Entity System ---------------------------------------------

enum { ENTITY_CREATURE, ENTITY_PARTICLE, ENTITY_OBJECT, ENTITY_ARROW };
class Entity {
public:
    float3 pos = make_float3(1.f, 0.f, 1.f);
    float3 acceleration = make_float3(0.f, 0.f, 0.f);
    float3 velocity = make_float3(0.f, 0.f, 0.f);
    bool isOnGround = true;
    bool isFlying = false;
    CollideBox box = CollideBox(make_float3(0, 0, 0), make_float3(0, 0, 0));
    virtual void dx(const float delta) { pos.x += delta; box.center.x += delta; }
    virtual void dy(const float delta) { pos.y += delta; box.center.y += delta; }
    virtual void dz(const float delta) { pos.z += delta; box.center.z += delta; }
    virtual void dX(const float3& vec) { pos += vec; box.center += vec; }
    virtual void dv(const float3& vec) { velocity += vec; }
    virtual void da(const float3& vec) { acceleration += vec; }
    virtual CollideBox& get_collideBox() { return box; }
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
    void dx(const float delta)  override{
        pos.x += delta;
        eye.x += delta;
        lookat.x += delta;
        box.center.x += delta;
    }
    void dy(const float delta)  override{
        pos.y += delta;
        eye.y += delta;
        lookat.y += delta;
        box.center.y += delta;
    }
    void dz(const float delta)  override{
        pos.z += delta;
        eye.z += delta;
        lookat.z += delta;
        box.center.z += delta;
    }
    void dX(const float3& vec)  override{
        pos += vec;
        eye += vec;
        lookat += vec;
        box.center += vec;
    }
    //@@todo: link with collidebox
};
std::vector<Creature*> crtList;//Creature list, the crtList[0] is our player.
Creature* player = nullptr;//refer to crtList[0]. Binding process located in void initEntitySystem()
Creature* control = nullptr;//refer to the creature you are controlling.

struct Particle : public Entity {
    static uint32_t OBJ_COUNT;
    int type = ENTITY_PARTICLE;
    ModelTexture texture_id = NONE;
    float beginTime = 0.f;
    float lifeLength = 0.f;
    Cube args;
    Particle() { OBJ_COUNT++; }
    ~Particle() { OBJ_COUNT--; }
    string get_type() {return "Cube";}
    void set_bound(float result[6])
    {
        auto* aabb = reinterpret_cast<OptixAabb*>(result);

        float3 m_min = args.center - args.size;
        float3 m_max = args.center + args.size;

        *aabb = {
                m_min.x, m_min.y, m_min.z,
                m_max.x, m_max.y, m_max.z
        };
    }
    uint32_t get_input_flag(){
        return OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
    }
    void set_hitgroup(WhittedState& state, HitGroupRecord* hgr, int idx) {
        set_hitgroup_cube_general(state, hgr, idx, this);
    }
    void dX(const float3& vec) override{
        pos += vec;
        args.center += vec;
        model_need_update = true;
    }

};
uint32_t Particle::OBJ_COUNT = 0;
std::vector<Particle*> ptcList;


void createParticle(float3& pos, float3& acceleration, float3& size, float timePtc, ModelTexture texture_id)
{
    Particle* tmp = new Particle;
    if (tmp != nullptr)
    {
        tmp->pos = pos;
        tmp->acceleration = acceleration;
        tmp->velocity = make_float3(0.f, 0.f, 0.f);
        tmp->beginTime = (float)glfwGetTime();
        tmp->lifeLength = timePtc;
        tmp->args.center = pos; tmp->args.size = size;
        tmp->texture_id = texture_id;//??????????????????
        ptcList.push_back(tmp);
    }
    else {
        throw sutil::Exception("Generating Particle Fault: gain nullptr when particling.");
    }
}//
void eraseParticle(Particle* pPar)
{
    if (pPar == nullptr) return;
    for (vector<Particle*>::iterator it = ptcList.begin(); it != ptcList.end();)
    {
        if (*it == pPar)
        {
            delete* it; // ??????????????????
            it = ptcList.erase(it);
            break;  //??????
        }
        it++;
    }
}

unsigned int jiangzemin = 19260817;
void createParticles_planeBounce(float3& place, float powery, float powerxz, float r, int number, float maxSize, ModelTexture texture_id)
{
    while (number--)
    {
        float theta = (float)fmod(rnd(jiangzemin), 2 * M_PI);
        float radiu = (float)fmod(rnd(jiangzemin), r);
        float randz = (float)fmod(rnd(jiangzemin), maxSize);
        createParticle(
            place + make_float3(radiu * cos(theta), 0.f, radiu * sin(theta)),
            make_float3(radiu * cos(theta) * powerxz, powery, radiu * sin(theta) * powerxz),
            make_float3(randz, randz, randz),
            0.3f,
            texture_id
        );
    }
}

void createParticles_Blockdestroy(float3& place, ModelTexture texture_id)
{
    int number = 5 + rand() % 7;
    while (number--)
    {
        float breakX = fmod(rnd(jiangzemin), 1.f) - 0.5f;
        float breakY = fmod(rnd(jiangzemin), 1.f) - 0.5f;
        float breakZ = fmod(rnd(jiangzemin), 1.f) - 0.5f;
        float randz = 0.05f + fmod(rnd(jiangzemin), 0.07f);
        createParticle(
            place + make_float3(breakX, breakY, breakZ),
            make_float3(breakX * 10.f, breakY * 10.f, breakZ * 10.f),
            make_float3(randz, randz, randz),
            0.25f,
            texture_id
        );
    }
}
// --------------------------------------------- Entity System ---------------------------------------------









//------------------------------------------------------------------------------
//
//  Geometry and Camera data
//
//------------------------------------------------------------------------------


const float PHYSICAL_SUN_RADIUS = 0.004675f;  // from Wikipedia
const float DEFAULT_SUN_RADIUS = 0.05f;  // Softer default to show off soft shadows
const float DEFAULT_SUN_THETA = 1.1f;
const float DEFAULT_SUN_PHI = 300.0f * M_PIf / 180.0f;

// data manipulation
void initData();
void saveData();

// to do: different pos in different time
DirectionalLight sun;

PreethamSunSky sky;
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
        if (button == GLFW_MOUSE_BUTTON_RIGHT )
        {
            if (istargeted)
            {
                float3 center = intersectBlock->get_collideBox().center;
                float3 tmp = f3abs(intersectPoint - center);
                float3 target;
                if (tmp.x >= tmp.y && tmp.x >= tmp.z)
                {
                    target = intersectBlock->get_collideBox().center + fsign(intersectPoint.x - center.x) * make_float3(1.f, 0.f, 0.f);
                }
                else if (tmp.y >= tmp.x && tmp.y >= tmp.z)
                {
                    target = intersectBlock->get_collideBox().center + fsign(intersectPoint.y - center.y) * make_float3(0.f, 1.f, 0.f);
                }
                else {
                    target = intersectBlock->get_collideBox().center + fsign(intersectPoint.z - center.z) * make_float3(0.f, 0.f, 1.f);
                }


                //??????????????????????????????????????????
                CollideBox tmpCLBOX = CollideBox(target, make_float3(0.5f,0.5f,0.5f));
                if (!CollideBox::collide_check(control->box, tmpCLBOX))
                {
                    float size = 0;
                    if (CurSize == NORM) size = 0.5;
                    else if (CurSize == THIN) size = 0.10;
                    modelLst.push_back(new cCube(target, {size, 0.5, size}, curTexture));
                    model_need_update = true;
                }


            }
        }
        else if (button == GLFW_MOUSE_BUTTON_LEFT)
        {
            if (istargeted)
            {
                if(isParticle) createParticles_Blockdestroy(intersectBlock->get_center(), intersectBlock->texture_id);
                for (vector<cModel*>::iterator it = modelLst.begin(); it != modelLst.end(); ++it)
                {
                    if (*it == intersectBlock)
                    {
                        if ((*it)->get_type() == "LightSphere") {
                            for (auto it2 = g_light.begin(); it2 != g_light.end(); ++it2) {
                                if ((*it2).id == ((cLightSphere*)(*it))->lightnum) {
                                    g_light.erase(it2);
                                    break;
                                }
                            }
                        }
                        delete *it; // ??????????????????
                        modelLst.erase(it);
                        break;  // ????????????????????????????????????
                    }
                }
                model_need_update = true;
                istargeted = false;
            }
        }
        
    }
    else
    {
        mouse_button = -1;
    }
}

static void mouseScrollCallback ( GLFWwindow* window, double xoffset, double yoffset ) {
    if(yoffset < 0) {
        curTexture = (ModelTexture)((curTexture + 1) % MT_SIZE);
    }
    if(yoffset > 0) {
        curTexture = (ModelTexture)((curTexture + MT_SIZE - 1) % MT_SIZE);
    }
}

static void cursorPosCallback( GLFWwindow* window, double xpos, double ypos )
{
    Params* params = static_cast<Params*>( glfwGetWindowUserPointer( window ) );

    if (mouse_button == GLFW_MOUSE_BUTTON_LEFT)
    {
    }
    else if (mouse_button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        
    }
    
    trackball.setViewMode( sutil::Trackball::EyeFixed );
    trackball.updateTracking( static_cast<int>( xpos ), static_cast<int>( ypos ), params->width, params->height );

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
        if (key == GLFW_KEY_LEFT_SHIFT)
        {
            camera_speed = 2.8f;
            sprint = true;
        }
        if (key == GLFW_KEY_I)
        {
            switchcam = true;// a test button. @@todo: change controlled entity.
        }
        if (key == GLFW_KEY_SPACE )
        {
            key_value['_'] = true, sccnt++;
            if (control->isOnGround && !control->isFlying)
            {
                control->isOnGround = false;
                control->acceleration += make_float3(0.f, 10.19804f + (camera_speed - 1.5f) * 2.134f, 0.f);
            }
        }
        if (key == GLFW_KEY_LEFT_CONTROL)
        {
            key_value['c'] = true, sccnt--;
        }

        if (key == GLFW_KEY_M)
        {
            if (control->isFlying)
            {
                control->isFlying = false;
                control->isOnGround = false;
            }
            else {
                control->isFlying = true;
                control->isOnGround = false;
            }
        }
        int curWidth = 0, curHeight = 0;
        glfwGetWindowSize(window, &curWidth, &curHeight);
        // make the window smaller
        if (key == GLFW_KEY_F10) {
            glfwSetWindowSize(window, (int)(curWidth/1.2f), (int)(curHeight/1.2f));
        }
        // make the window GREAT again
        if (key == GLFW_KEY_F11) {
            glfwSetWindowSize(window, (int)(curWidth*1.2f), (int)(curHeight*1.2f));
        }
        // save your file
        if (key == GLFW_KEY_F5) {
            saveData();
        }
        if (key == GLFW_KEY_F6) {
            renewShadowOnTime = !renewShadowOnTime;
            std::cout << "renewShadowOnTime = " << renewShadowOnTime << std::endl;
        }

        // ??????????????????
        if (key == GLFW_KEY_F) {
            if (istargeted) {
                curTexture = intersectBlock->texture_id;
            }
        }
        
        if (key == GLFW_KEY_T) {
            isParticle = !isParticle;
            std::cout << "isParticle = " << isParticle << std::endl;
        }
        if (key == GLFW_KEY_G)
        {
            control->dX(make_float3(8.0f, 19.7f, -4.0f) - control->pos);
        }

        if (key == GLFW_KEY_B) {
            CurSize = (BlockSize)((CurSize + 1) % BS_SIZE);
        }

        if (key == GLFW_KEY_K) {
            curLightColor = (LightColor)( (curLightColor + 1) % LC_SIZE );
        }

        if (key == GLFW_KEY_L) {
            if (istargeted)
            {
                float3 center = intersectBlock->get_collideBox().center;
                float3 tmp = f3abs(intersectPoint - center);
                float3 target;
                if (tmp.x >= tmp.y && tmp.x >= tmp.z)
                {
                    target = intersectBlock->get_collideBox().center + fsign(intersectPoint.x - center.x) * make_float3(1.f, 0.f, 0.f);
                }
                else if (tmp.y >= tmp.x && tmp.y >= tmp.z)
                {
                    target = intersectBlock->get_collideBox().center + fsign(intersectPoint.y - center.y) * make_float3(0.f, 1.f, 0.f);
                }
                else {
                    target = intersectBlock->get_collideBox().center + fsign(intersectPoint.z - center.z) * make_float3(0.f, 0.f, 1.f);
                }


                //??????????????????????????????????????????
                CollideBox tmpCLBOX = CollideBox(target, make_float3(0.5f, 0.5f, 0.5f));
                if (!CollideBox::collide_check(control->box, tmpCLBOX))
                {
                    modelLst.push_back(new cLightSphere(target, curLightColor, 0.1f, curTexture));
                    model_need_update = true;
                }


            }
        }

        if (key == GLFW_KEY_LEFT_ALT)
        {
            if (isNoCursor)
            {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                
            }
            else {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            }
            isNoCursor = !isNoCursor;
            
        }
    }
    else if (action == GLFW_RELEASE)
    {
        if (key == GLFW_KEY_W) key_value['w'] = false, wscnt--;
        if (key == GLFW_KEY_A) key_value['a'] = false, adcnt--;
        if (key == GLFW_KEY_S) key_value['s'] = false, wscnt++;
        if (key == GLFW_KEY_D) key_value['d'] = false, adcnt++;
        if (key == GLFW_KEY_LEFT_SHIFT)
        {
            camera_speed = 1.5f;
            sprint = false;
        }
        if (key == GLFW_KEY_SPACE)
        {
            key_value['_'] = false, sccnt--;
        }
        if (key == GLFW_KEY_LEFT_CONTROL)
        {
            key_value['c'] = false, sccnt++;
        }

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
    std::cerr << "         --dim=<width>x<height>      Set image dimensions; defaults to 1024x768\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    exit( 0 );
}

void displayHUD(float width, float height) {
    constexpr std::chrono::duration<double> display_update_min_interval_time( 0.5 );
    static int32_t                          total_subframe_count = 0;
    static int32_t                          last_update_frames   = 0;
    static auto                             last_update_time     = std::chrono::steady_clock::now();
    static char                             display_text[128];

    const auto cur_time = std::chrono::steady_clock::now();

    sutil::beginFrameImGui();

    last_update_frames++;

    typedef std::chrono::duration<double, std::milli> durationMs;

    // center

    const char* sCenter = "    |\n    |\n----+----\n    |\n    |";
    //todo imgui ????????????/??????
    float font_size_x = 80;
    float font_size_y = 80;

    sutil::displayText( sCenter,
                        width/2 - font_size_x / 2,
                        height/2 - font_size_y / 2 );

    sutil::endFrameImGui();

    // item

    sutil::beginFrameImGui();

    string sLeftDown = "CurBlock:\n" + get_texture_name(curTexture);
    if (CurSize == THIN) sLeftDown += "_T";
    sutil::displayText( sLeftDown.c_str(),
                        0,
                        height - font_size_y / 2 );

    sutil::endFrameImGui();

    ++total_subframe_count;
}

void initLaunchParams( WhittedState& state )
{
    if(!first_launch) {
        CUDA_CHECK(cudaFree((void*)state.params.accum_buffer));
    }
    CUDA_CHECK( cudaMalloc(
            reinterpret_cast<void**>( &state.params.accum_buffer ),
            state.params.width*state.params.height*sizeof(float4)
    ) );
    state.params.frame_buffer = nullptr; // Will be set when output buffer is mapped

    state.params.subframe_index = 0u;
    state.params.samples_per_launch = 3u;
    state.params.num_lights_sample = 1u;
    state.params.point_light_sum = static_cast<uint32_t>(g_light.size());
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&state.params.point_light.data),
        g_light.size() * sizeof(BasicLight)
    ));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(state.params.point_light.data),
        g_light.data(),
        g_light.size() * sizeof(BasicLight),
        cudaMemcpyHostToDevice
    ));
    state.params.sun = sun;
    state.params.sky = sky;
    state.params.circle = circle;
    state.params.game_time = game_time;
    state.params.ambient_light_color = make_float3( 0.1f, 0.1f, 0.1f );
    state.params.max_depth = max_trace;
    state.params.scene_epsilon = 1.e-4f;

    if(first_launch) {
        CUDA_CHECK( cudaStreamCreate( &state.stream ) );
    }
    if(first_launch) {
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_params ), sizeof( Params ) ) );
    }

    state.params.handle = state.gas_handle;

    first_launch = false;
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

void createTextures()
{
    int numTextures = (int)texture_list.size();

    textureArrays.resize(numTextures);

    for (int textureID = 0; textureID < numTextures; textureID++) {
        auto texture = texture_list[textureID];

        cudaResourceDesc res_desc = {};

        cudaChannelFormatDesc channel_desc;
        int32_t width = texture->resolution.x;
        int32_t height = texture->resolution.y;
        int32_t numComponents = 4;
        int32_t pitch = width * numComponents * sizeof(uint8_t);
        channel_desc = cudaCreateChannelDesc<uchar4>();

        cudaArray_t& pixelArray = textureArrays[textureID];
        CUDA_CHECK(cudaMallocArray(&pixelArray,
            &channel_desc,
            width, height));

        CUDA_CHECK(cudaMemcpy2DToArray(pixelArray,
            /* offset */0, 0,
            texture->pixel,
            pitch, pitch, height,
            cudaMemcpyHostToDevice));
        
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = pixelArray;

        cudaTextureDesc tex_desc = {};
        tex_desc.addressMode[0] = cudaAddressModeWrap;
        tex_desc.addressMode[1] = cudaAddressModeWrap;
        tex_desc.filterMode = cudaFilterModeLinear;
        tex_desc.readMode = cudaReadModeNormalizedFloat;
        tex_desc.normalizedCoords = 1;
        tex_desc.maxAnisotropy = 1;
        tex_desc.maxMipmapLevelClamp = 99;
        tex_desc.minMipmapLevelClamp = 0;
        tex_desc.mipmapFilterMode = cudaFilterModePoint;
        tex_desc.borderColor[0] = 1.0f;
        tex_desc.sRGB = 0;

        // Create texture object
        cudaTextureObject_t cuda_tex = 0;
        CUDA_CHECK(cudaCreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr));
        texture_list[textureID]->textureObject = cuda_tex;
    }
}

inline float calc_xz_distance(float3 delta) {
    return sqrt(delta.x * delta.x + delta.z + delta.z);
}

void createGeometry( WhittedState &state ) {
    //
    // Build Custom Primitives
    //

    // Load AABB into device memory
    int sumCOUNT = cModel::OBJ_COUNT + Particle::OBJ_COUNT;
    OptixAabb*  aabb = new OptixAabb[sumCOUNT];
    CUdeviceptr d_aabb;

    constexpr const float RENDER_DISTANT = 100000;

    for(size_t i=0; i<cModel::OBJ_COUNT; i++) {
        if(calc_xz_distance(control->pos - modelLst[i]->get_center()) > RENDER_DISTANT) {
            set_bound_zero(reinterpret_cast<float*>(&aabb[i]));
        } else {
            modelLst[i]->set_bound(reinterpret_cast<float*>(&aabb[i]));
        }
    }
    for (size_t i = 0; i < Particle::OBJ_COUNT; i++) {
        // ptc??????????????????????????????
        ptcList[i]->set_bound(reinterpret_cast<float*>(&aabb[i+cModel::OBJ_COUNT]));
    }

    // std::cerr << "[INFO] aabb size: " << cModel::OBJ_COUNT * sizeof( OptixAabb ) << std::endl;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_aabb
                            ), sumCOUNT * sizeof( OptixAabb ) ) );
    CUDA_CHECK( cudaMemcpy(
            reinterpret_cast<void*>( d_aabb ),
            aabb,                                       // notice: ???????????????&aabb, ???????????????????????????????????? //????????????????????????????????????magic number BUG //?????????????????????????????????
            sumCOUNT * sizeof( OptixAabb ),
            cudaMemcpyHostToDevice
    ) );

    // Setup AABB build input
    uint32_t* aabb_input_flags = new uint32_t[sumCOUNT];
    for(size_t i=0; i<cModel::OBJ_COUNT; i++) {
        aabb_input_flags[i] = modelLst[i]->get_input_flag();
    }
    for (size_t i = 0; i < Particle::OBJ_COUNT; i++) {
        aabb_input_flags[i+cModel::OBJ_COUNT] = ptcList[i]->get_input_flag();
    }

    /* TODO: This API cannot control flags for different ray type */

    // originally 0, 1, 2
    uint32_t* sbt_index = new uint32_t[sumCOUNT];
    for(int i=0; i<sumCOUNT; i++)
        sbt_index[i] = i;

    size_t size_sbt_index = sumCOUNT * sizeof(uint32_t);

    if(d_sbt_index_allocated) {
        CUDA_CHECK( cudaFree( (void*)d_sbt_index) );
        d_sbt_index_allocated = false;  // muda
    }

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_sbt_index ), size_sbt_index ) );
    CUDA_CHECK( cudaMemcpy(
            reinterpret_cast<void*>( d_sbt_index ),
            sbt_index,
            size_sbt_index,
            cudaMemcpyHostToDevice ) );

    d_sbt_index_allocated = true;

    OptixBuildInput aabb_input = {};
    aabb_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    aabb_input.customPrimitiveArray.aabbBuffers   = &d_aabb;
    aabb_input.customPrimitiveArray.flags         = aabb_input_flags;
    aabb_input.customPrimitiveArray.numSbtRecords = sumCOUNT;
    aabb_input.customPrimitiveArray.numPrimitives = sumCOUNT;
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

    delete aabb;
    delete sbt_index;
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
        const char* input = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "sunsky.cu", inputSize);
        OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
                state.context,
                &module_compile_options,
                &state.pipeline_compile_options,
                input,
                inputSize,
                log,
                &sizeof_log,
                &state.sunsky_module ) );
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

static void createTextureSphereProgram( WhittedState &state, std::vector<OptixProgramGroup> &program_groups )
{
    OptixProgramGroup           radiance_sphere_prog_group;
    OptixProgramGroupOptions    radiance_sphere_prog_group_options = {};
    OptixProgramGroupDesc       radiance_sphere_prog_group_desc = {};
    radiance_sphere_prog_group_desc.kind   = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
            radiance_sphere_prog_group_desc.hitgroup.moduleIS           = state.geometry_module;
    radiance_sphere_prog_group_desc.hitgroup.entryFunctionNameIS    = "__intersection__sphere";
    radiance_sphere_prog_group_desc.hitgroup.moduleCH               = state.shading_module;
    radiance_sphere_prog_group_desc.hitgroup.entryFunctionNameCH    = "__closesthit__texture_radiance";
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
    state.radiance_texture_sphere_prog_group = radiance_sphere_prog_group;

    OptixProgramGroup           occlusion_sphere_prog_group;
    OptixProgramGroupOptions    occlusion_sphere_prog_group_options = {};
    OptixProgramGroupDesc       occlusion_sphere_prog_group_desc = {};
    occlusion_sphere_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
        occlusion_sphere_prog_group_desc.hitgroup.moduleIS = state.geometry_module;
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
    state.occlusion_glass_sphere_prog_group = occlusion_sphere_prog_group;
}

static void createTextureCubeProgram(WhittedState& state, std::vector<OptixProgramGroup>& program_groups)
{
    OptixProgramGroup           radiance_cube_prog_group;
    OptixProgramGroupOptions    radiance_cube_prog_group_options = {};
    OptixProgramGroupDesc       radiance_cube_prog_group_desc = {};
    radiance_cube_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
        radiance_cube_prog_group_desc.hitgroup.moduleIS = state.geometry_module;
    radiance_cube_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__cube";
    radiance_cube_prog_group_desc.hitgroup.moduleCH = state.shading_module;
    radiance_cube_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__texture_radiance";
    radiance_cube_prog_group_desc.hitgroup.moduleAH = nullptr;
    radiance_cube_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;

    char    log[2048];
    size_t  sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &radiance_cube_prog_group_desc,
        1,
        &radiance_cube_prog_group_options,
        log,
        &sizeof_log,
        &radiance_cube_prog_group));

    program_groups.push_back(radiance_cube_prog_group);
    state.radiance_texture_cube_prog_group = radiance_cube_prog_group;

    OptixProgramGroup           occlusion_cube_prog_group;
    OptixProgramGroupOptions    occlusion_cube_prog_group_options = {};
    OptixProgramGroupDesc       occlusion_cube_prog_group_desc = {};
    occlusion_cube_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
        occlusion_cube_prog_group_desc.hitgroup.moduleIS = state.geometry_module;
    occlusion_cube_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__cube";
    occlusion_cube_prog_group_desc.hitgroup.moduleCH = state.shading_module;
    occlusion_cube_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__full_occlusion";
    occlusion_cube_prog_group_desc.hitgroup.moduleAH = nullptr;
    occlusion_cube_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &occlusion_cube_prog_group_desc,
        1,
        &occlusion_cube_prog_group_options,
        log,
        &sizeof_log,
        &occlusion_cube_prog_group));

    program_groups.push_back(occlusion_cube_prog_group);
    state.occlusion_texture_cube_prog_group = occlusion_cube_prog_group;
}

static void createGlassCubeProgram(WhittedState& state, std::vector<OptixProgramGroup>& program_groups)
{
    OptixProgramGroup           radiance_cube_prog_group;
    OptixProgramGroupOptions    radiance_cube_prog_group_options = {};
    OptixProgramGroupDesc       radiance_cube_prog_group_desc = {};
    radiance_cube_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
        radiance_cube_prog_group_desc.hitgroup.moduleIS = state.geometry_module;
    radiance_cube_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__cube";
    radiance_cube_prog_group_desc.hitgroup.moduleCH = state.shading_module;
    radiance_cube_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__transparency_radiance";
    radiance_cube_prog_group_desc.hitgroup.moduleAH = nullptr;
    radiance_cube_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;

    char    log[2048];
    size_t  sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &radiance_cube_prog_group_desc,
        1,
        &radiance_cube_prog_group_options,
        log,
        &sizeof_log,
        &radiance_cube_prog_group));

    program_groups.push_back(radiance_cube_prog_group);
    state.radiance_glass_cube_prog_group = radiance_cube_prog_group;

    OptixProgramGroup           occlusion_cube_prog_group;
    OptixProgramGroupOptions    occlusion_cube_prog_group_options = {};
    OptixProgramGroupDesc       occlusion_cube_prog_group_desc = {};
    occlusion_cube_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
        occlusion_cube_prog_group_desc.hitgroup.moduleIS = state.geometry_module;
    occlusion_cube_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__cube";
    occlusion_cube_prog_group_desc.hitgroup.moduleCH = nullptr;
    occlusion_cube_prog_group_desc.hitgroup.entryFunctionNameCH = nullptr;
    occlusion_cube_prog_group_desc.hitgroup.moduleAH = state.shading_module;
    occlusion_cube_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__glass_occlusion";

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &occlusion_cube_prog_group_desc,
        1,
        &occlusion_cube_prog_group_options,
        log,
        &sizeof_log,
        &occlusion_cube_prog_group));

    program_groups.push_back(occlusion_cube_prog_group);
    state.occlusion_glass_cube_prog_group = occlusion_cube_prog_group;
}

static void createWaterCubeProgram(WhittedState& state, std::vector<OptixProgramGroup>& program_groups)
{
    OptixProgramGroup           radiance_cube_prog_group;
    OptixProgramGroupOptions    radiance_cube_prog_group_options = {};
    OptixProgramGroupDesc       radiance_cube_prog_group_desc = {};
    radiance_cube_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
        radiance_cube_prog_group_desc.hitgroup.moduleIS = state.geometry_module;
    radiance_cube_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__cube";
    radiance_cube_prog_group_desc.hitgroup.moduleCH = state.shading_module;
    radiance_cube_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__water_radiance";
    radiance_cube_prog_group_desc.hitgroup.moduleAH = nullptr;
    radiance_cube_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;

    char    log[2048];
    size_t  sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &radiance_cube_prog_group_desc,
        1,
        &radiance_cube_prog_group_options,
        log,
        &sizeof_log,
        &radiance_cube_prog_group));

    program_groups.push_back(radiance_cube_prog_group);
    state.radiance_water_cube_prog_group = radiance_cube_prog_group;

    OptixProgramGroup           occlusion_cube_prog_group;
    OptixProgramGroupOptions    occlusion_cube_prog_group_options = {};
    OptixProgramGroupDesc       occlusion_cube_prog_group_desc = {};
    occlusion_cube_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
        occlusion_cube_prog_group_desc.hitgroup.moduleIS = state.geometry_module;
    occlusion_cube_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__cube";
    occlusion_cube_prog_group_desc.hitgroup.moduleCH = state.shading_module;
    occlusion_cube_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__full_occlusion";
    occlusion_cube_prog_group_desc.hitgroup.moduleAH = nullptr;
    occlusion_cube_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &occlusion_cube_prog_group_desc,
        1,
        &occlusion_cube_prog_group_options,
        log,
        &sizeof_log,
        &occlusion_cube_prog_group));

    program_groups.push_back(occlusion_cube_prog_group);
    state.occlusion_water_cube_prog_group = occlusion_cube_prog_group;
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
    miss_prog_group_desc.miss.module             = state.sunsky_module;
    miss_prog_group_desc.miss.entryFunctionName  = "__miss__bg";
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
            6,    /* RadiancePRD uses 5 payloads */                 // numPayloadValues
            6,    /* Parallelogram intersection uses 5 attrs */     // numAttributeValues
            OPTIX_EXCEPTION_FLAG_NONE,                              // exceptionFlags
            "params"                                                // pipelineLaunchParamsVariableName
    };

    // Prepare program groups
    createModules( state );
    createCameraProgram( state, program_groups );
    createGlassSphereProgram( state, program_groups );
    createTextureSphereProgram( state, program_groups );
    createTextureCubeProgram(state, program_groups);
    createGlassCubeProgram(state, program_groups);
    createWaterCubeProgram(state, program_groups);
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

void createSBT( WhittedState &state)
{
    static bool first_createSBT = true;
    if(first_createSBT) {
        // Raygen program record
        {
            size_t sizeof_raygen_record = sizeof( RayGenRecord );

            if(d_raygen_record_allocated) {
                CUDA_CHECK( cudaFree((void*)d_raygen_record) );
                d_raygen_record_allocated = false;
            }

            CUDA_CHECK( cudaMalloc(
                    reinterpret_cast<void**>( &d_raygen_record ),
                    sizeof_raygen_record ) );

            d_raygen_record_allocated = true;

            state.sbt.raygenRecord = d_raygen_record;
        }

        // Miss program record
        {
            size_t sizeof_miss_record = sizeof( MissRecord );
        
            if(d_miss_record_allocated) {
                CUDA_CHECK(cudaFree((void*)d_miss_record));
                d_miss_record_allocated = false;
            }

            CUDA_CHECK( cudaMalloc(
                    reinterpret_cast<void**>( &d_miss_record ),
                    sizeof_miss_record*RAY_TYPE_COUNT ) );

            d_miss_record_allocated = true;

            MissRecord ms_sbt[RAY_TYPE_COUNT];
            optixSbtRecordPackHeader( state.radiance_miss_prog_group, &ms_sbt[0] );
            optixSbtRecordPackHeader( state.occlusion_miss_prog_group, &ms_sbt[1] );
            ms_sbt[1].data.bg_color = ms_sbt[0].data.bg_color = { 0.34f, 0.55f, 0.85f };
            ms_sbt[1].data.morning_map = ms_sbt[0].data.morning_map = texture_list[textures["MORNING_MAP"]]->textureObject;
            ms_sbt[1].data.noon_map = ms_sbt[0].data.noon_map = texture_list[textures["NOON_MAP"]]->textureObject;
            ms_sbt[1].data.night_map = ms_sbt[0].data.night_map = texture_list[textures["NIGHT_MAP"]]->textureObject;
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
        first_createSBT = false;
    }

    // Hitgroup program record
    {
        size_t count_records = 2 * (cModel::OBJ_COUNT + Particle::OBJ_COUNT);
        HitGroupRecord* hitgroup_records = new HitGroupRecord[count_records];

        // Note: Fill SBT record array the same order like AS is built.
        //std::cout << modelLst.size() << std::endl;
        for(int i = 0; i < 2 * cModel::OBJ_COUNT ; i += 2) {
            modelLst[i / 2]->set_hitgroup(state, hitgroup_records, i);
        }
       for(int i = 0; i < 2 * Particle::OBJ_COUNT; i += 2) {
            ptcList[i / 2]->set_hitgroup(state, hitgroup_records, i + 2 * cModel::OBJ_COUNT);
        }

        size_t      sizeof_hitgroup_record = sizeof( HitGroupRecord );
        
        if(d_hitgroup_records_allocated) {
            CUDA_CHECK(cudaFree((void*)d_hitgroup_records));
            d_hitgroup_records_allocated = false;
        }
        
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

        d_hitgroup_records_allocated = true;

        state.sbt.hitgroupRecordBase            = d_hitgroup_records;
        state.sbt.hitgroupRecordCount           = count_records;
        state.sbt.hitgroupRecordStrideInBytes   = static_cast<uint32_t>( sizeof_hitgroup_record );

        delete hitgroup_records;
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

bool detectModelUpdate(WhittedState& state) {
    if(model_need_update) {
        createGeometry  ( state );
        createSBT      ( state );
        initLaunchParams( state );
        model_need_update = false;
        return true;
    }
    return false;
}

bool isCollide_creature_cModel(Creature*& ent)
{
    CollideBox entbox = ent->get_collideBox();
    cModel* possibleCollider = nullptr;
    for (register float cllx = entbox.center.x - entbox.size.x; cllx <= entbox.center.x + entbox.size.x; cllx += 1.0f)
    {
        for (register float clly = entbox.center.y - entbox.size.y; clly <= entbox.center.y + entbox.size.y; clly += 1.0f)
        {
            for (register float cllz = entbox.center.z - entbox.size.z; cllz <= entbox.center.z + entbox.size.z; cllz += 1.0f)
            {
                if (get_model_at(make_float3(cllx, clly, cllz), possibleCollider) && possibleCollider->collidable)
                {
                    return true;
                }
            }
        }
    }
    for (register float clly = entbox.center.y - entbox.size.y; clly <= entbox.center.y + entbox.size.y; clly += 1.0f)
    {
        for (register float cllz = entbox.center.z - entbox.size.z; cllz <= entbox.center.z + entbox.size.z; cllz += 1.0f)
        {
            if (get_model_at(make_float3(entbox.center.x + entbox.size.x, clly, cllz), possibleCollider) && possibleCollider->collidable)
            {
                return true;
            }
        }
    }
    for (register float cllx = entbox.center.x - entbox.size.x; cllx <= entbox.center.x + entbox.size.x; cllx += 1.0f)
    {
        for (register float cllz = entbox.center.z - entbox.size.z; cllz <= entbox.center.z + entbox.size.z; cllz += 1.0f)
        {
            if (get_model_at(make_float3(cllx, entbox.center.y + entbox.size.y, cllz), possibleCollider) && possibleCollider->collidable)
            {
                return true;
            }
        }
    }
    for (register float cllx = entbox.center.x - entbox.size.x; cllx <= entbox.center.x + entbox.size.x; cllx += 1.0f)
    {
        for (register float clly = entbox.center.y - entbox.size.y; clly <= entbox.center.y + entbox.size.y; clly += 1.0f)
        {
            if (get_model_at(make_float3(cllx, clly, entbox.center.z + entbox.size.z), possibleCollider) && possibleCollider->collidable)
            {
                return true;
            }
        }
    }
    for (register float clly = entbox.center.y - entbox.size.y; clly <= entbox.center.y + entbox.size.y; clly += 1.0f)
    {
        if (get_model_at(make_float3(entbox.center.x + entbox.size.x, clly, entbox.center.z + entbox.size.z), possibleCollider) && possibleCollider->collidable)
        {
            return true;
        }
    }
    for (register float cllz = entbox.center.z - entbox.size.z; cllz <= entbox.center.z + entbox.size.z; cllz += 1.0f)
    {
        if (get_model_at(make_float3(entbox.center.x + entbox.size.x, entbox.center.y + entbox.size.y, cllz), possibleCollider) && possibleCollider->collidable)
        {
            return true;
        }
    }
    for (register float cllx = entbox.center.x - entbox.size.x; cllx <= entbox.center.x + entbox.size.x; cllx += 1.0f)
    {
        if (get_model_at(make_float3(cllx, entbox.center.y + entbox.size.y, entbox.center.z + entbox.size.z), possibleCollider) && possibleCollider->collidable)
        {
            return true;
        }
    }
    if (get_model_at(make_float3(entbox.center.x + entbox.size.x, entbox.center.y + entbox.size.y, entbox.center.z + entbox.size.z), possibleCollider) && possibleCollider->collidable)
    {
        return true;
    }
    return false;
}
bool isCollide(Creature*& ent)
{
    if (isCollide_creature_cModel(ent)) return true;

    return false;
}
//
// Camera
//

bool axisLegal(int x, int y)
{
    if (x < -50 || x>50 || y < -50 || y>50)
    {
        return false;
    }
    return true;
}
const int deltaX[10] = { 0,1,1,1,0,0,-1,-1,-1,0 };
const int deltaY[10] = { 0,-1,0,1,-1,1,-1,0,1,0 };
const int ddeltaX[20] = { 0,2,2,2,2,2,    1,1,0,0,-1,-1, -2,-2,-2,-2,-2 };
const int ddeltaY[20] = { 0,2,1,0,-1,-2,  2,-2,2,-2,2,-2, 2,1,0,-1,-2 };
void createBuildTree(float x, float y, float z, float height)
{
    cModel* tmpModelcheck = nullptr;
    if (height == 2)
    {
        for (register int i = 0; i <= 4; i++)
        {
            if(!get_model_at(make_float3(x, y + i, z), tmpModelcheck)) modelLst.push_back(new cCube({ x, y+i, z }, 0.5f, BARK));
        }
        for (register float h = 0; h >= -2.1f; h -= 1.0f)
        {
            for (register int i = 1; i <= 8; i++)
            {
                float tx = x + deltaX[i];
                float tz = z + deltaY[i];
                if (!get_model_at(make_float3(tx, y + 4.f + h, tz), tmpModelcheck)) modelLst.push_back(new cCube({ tx , y + 4.f + h, tz }, 0.5f, LEAF));
            }
        }
        for (register float h = 0; h >= -1.1f; h -= 1.0f)
        {
            for (register int i = 1; i <= 16; i++)
            {
                float tx = x + ddeltaX[i];
                float tz = z + ddeltaY[i];
                if (!get_model_at(make_float3(tx, y + 3.f + h, tz), tmpModelcheck)) modelLst.push_back(new cCube({ tx , y + 3.f + h, tz }, 0.5f, LEAF));
            }
        }
    }
    else if (height == 3)
    {
        for (register float i = 0.f; i <= 4.f; i+=1.f)
        {
            if (!get_model_at(make_float3(x, y + i, z), tmpModelcheck)) modelLst.push_back(new cCube({ x, y + i, z }, 0.5f, BARK));
        }
        for (register float h = 0; h >= -2.1f; h -= 1.0f)
        {
            for (register int i = 1; i <= 8; i++)
            {
                float tx = x + deltaX[i];
                float tz = z + deltaY[i];
                if (!get_model_at(make_float3(tx, y + 5.f + h, tz), tmpModelcheck)) modelLst.push_back(new cCube({ tx , y + 5.f + h, tz }, 0.5f, LEAF));
            }
        }
        if (!get_model_at(make_float3(x, y + 5.f, z), tmpModelcheck)) modelLst.push_back(new cCube({ x , y + 5.f, z }, 0.5f, LEAF));
        for (register float h = 0; h >= -1.1f; h -= 1.0f)
        {
            for (register int i = 1; i <= 16; i++)
            {
                float tx = x + ddeltaX[i];
                float tz = z + ddeltaY[i];
                if (!get_model_at(make_float3(tx, y + 4.f + h, tz), tmpModelcheck)) modelLst.push_back(new cCube({ tx , y + 4.f + h, tz }, 0.5f, LEAF));
            }
        }
    }
    else if (height == 4)
    {
        for (register float i = 0.f; i <= 5.f; i += 1.f)
        {
            if (!get_model_at(make_float3(x, y + i, z), tmpModelcheck)) modelLst.push_back(new cCube({ x, y + i, z }, 0.5f, BARK));
        }
        for (register float h = 0; h >= -3.1f; h -= 1.0f)
        {
            for (register int i = 1; i <= 8; i++)
            {
                float tx = x + deltaX[i];
                float tz = z + deltaY[i];
                if (!get_model_at(make_float3(tx, y + 6.f + h, tz), tmpModelcheck)) modelLst.push_back(new cCube({ tx , y + 6.f + h, tz }, 0.5f, LEAF));
            }
        }
        if (!get_model_at(make_float3(x, y + 6.f, z), tmpModelcheck)) modelLst.push_back(new cCube({ x , y + 6.f, z }, 0.5f, LEAF));
        if (!get_model_at(make_float3(x, y + 7.f, z), tmpModelcheck)) modelLst.push_back(new cCube({ x , y + 7.f, z }, 0.5f, LEAF));
        for (register float h = 0; h >= -1.1f; h -= 1.0f)
        {
            for (register int i = 1; i <= 16; i++)
            {
                float tx = x + ddeltaX[i];
                float tz = z + ddeltaY[i];
                if (!get_model_at(make_float3(tx, y + 5.f + h, tz), tmpModelcheck))   modelLst.push_back(new cCube({ tx , y + 5.f + h, tz }, 0.5f, LEAF));
            }
        }
    }
}
int worldIntList[110][110] = { 0 };
int worldHeightList[110][110] = { 0 };
bool isTree[110][110] = { 0 };
void initData()
{
    /*std::ifstream savefile(sutil::sampleDataFilePath("Saves/SAVE0.txt"), std::ios::in);
    if (!savefile) std::cout << "failed to load savefiles" << std::endl;
    else {
        int savecnt = 0;
        float savedx, savedy, savedz, saveds;
        int savedTexture;
        while (!savefile.eof())
        {
            savecnt++;
            savefile >> savedx >> savedy >> savedz >> saveds >> savedTexture;
            modelLst.push_back(new cCube({ savedx, savedy, savedz }, saveds, (ModelTexture)savedTexture));
        }
        std::cerr << "Successfully loaded " << savecnt << " Cubes" << std::endl;
        savefile.close();
    }*/

    //Part1:??????0-1???
    int yeshouxianbei = 114514;
    srand(jiangzemin);

    for (register int i = -50; i <= 50; i++)
    {
        for (register int j = -50; j <= 50; j++)
        {
            worldIntList[i+50][j+50] = generateIntScaling(rand() % 4);
        }
    }
    //?????????
    for (register int epoch = 1; epoch <= 3; epoch++)
    {
        for (register int i = -49; i <= 49; i++)
        {
            for (register int j = -49; j <= 49; j++)
            {
                int cnt = 0;
                for (register int k = 1; k <= 9; k++)
                {
                    int tx = i + deltaX[k];
                    int ty = j + deltaY[k];
                    if (worldIntList[tx + 50][ty + 50])
                    {
                        cnt++;
                    }
                }
                if (cnt > 3)
                {
                    worldIntList[i + 50][j + 50] = 1;
                }
                else {
                    worldIntList[i + 50][j + 50] = 0;
                }
            }
        }
    }
    //???????????????
    for (register int i = -49; i <= 49; i++)
    {
        for (register int j = -49; j <= 49; j++)
        {
            int cnt = 0;
            for (register int k = 1; k <= 9; k++)
            {
                int tx = i + deltaX[k];
                int ty = j + deltaY[k];
                if (worldIntList[tx + 50][ty + 50])
                {
                    cnt++;
                }
            }
            if (cnt > 4)
            {
                worldIntList[i + 50][j + 50] = 1;
            }
            else {
                worldIntList[i + 50][j + 50] = 0;
            }
        }
    }

    

    for (register int i = -50; i <= 50; i++)
    {
        for (register int j = -50; j <= 50; j++)
        {
            int Worlddepth = 0;
            for (Worlddepth = 0; Worlddepth <= 20; Worlddepth++)
            {
                bool isBreak = false;
                for (register int k = i - Worlddepth; k <= i + Worlddepth; k++)
                {
                    if (axisLegal(k,j+Worlddepth) && worldIntList[k+50][j + Worlddepth+50] == 0)
                    {
                        isBreak = true;
                        break;
                    }
                }
                if (isBreak) break;

                for (register int k = i - Worlddepth; k <= i + Worlddepth; k++)
                {
                    if (axisLegal(k, j - Worlddepth) && worldIntList[k + 50][j - Worlddepth + 50] == 0)
                    {
                        isBreak = true;
                        break;
                    }
                }
                if (isBreak) break;

                for (register int k = j - Worlddepth; k <= j + Worlddepth; k++)
                {
                    if (axisLegal(i + Worlddepth,k) && worldIntList[i + Worlddepth + 50][k + 50] == 0)
                    {
                        isBreak = true;
                        break;
                    }
                }
                if (isBreak) break;

                for (register int k = j - Worlddepth; k <= j + Worlddepth; k++)
                {
                    if (axisLegal(i - Worlddepth, k) && worldIntList[i - Worlddepth + 50][k + 50] == 0)
                    {
                        isBreak = true;
                        break;
                    }
                }
                if (isBreak) break;
            }
            worldHeightList[i + 50][j + 50] = Worlddepth;
        }
    }

    //???????????????????????????
    srand(yeshouxianbei);
    for (register int i = -50; i <= 50; i++)
    {
        for (register int j = -50; j <= 50; j++)
        {
            worldIntList[i + 50][j + 50] = generateIntScaling(rand() % 4);
        }
    }
    for (register int epoch = 1; epoch <= 3; epoch++)
    {
        for (register int i = -49; i <= 49; i++)
        {
            for (register int j = -49; j <= 49; j++)
            {
                int cnt = 0;
                for (register int k = 1; k <= 9; k++)
                {
                    int tx = i + deltaX[k];
                    int ty = j + deltaY[k];
                    if (worldIntList[tx + 50][ty + 50])
                    {
                        cnt++;
                    }
                }
                if (cnt > 3)
                {
                    worldIntList[i + 50][j + 50] = 1;
                }
                else {
                    worldIntList[i + 50][j + 50] = 0;
                }
            }
        }
    }
    for (register int i = -50; i <= 50; i++)
    {
        for (register int j = -50; j <= 50; j++)
        {
            for (register int k = 1; k <= 8; k++)
            {
                int tx = i + deltaX[k];
                int ty = j + deltaY[k];
                if (axisLegal(tx,ty) && worldIntList[tx + 50][ty + 50])
                {
                    worldIntList[i + 50][j + 50] = 0;
                    break;
                }
            }
        }
    }
    for (register int i = -49; i <= 49; i++)
    {
        for (register int j = -49; j <= 49; j++)
        {
            if (worldIntList[i + 50][j + 50]) isTree[i + 50][j + 50] = true;
        }
    }


    for (register int i = -50; i <= 50; i++)
    {
        for (register int j = -50; j <= 50; j++)
        {
            modelLst.push_back(new cCube({ i + 0.5f,  0.5f, j + 0.5f }, 0.5f, DIRT));
            for (register int k = 0; k <= worldHeightList[i + 50][j + 50] - 1; k++)
            {
                modelLst.push_back(new cCube({ i + 0.5f, k+ 1.f + 0.5f, j + 0.5f }, 0.5f, DIRT));
            }
            modelLst.push_back(new cCube({ i + 0.5f,  worldHeightList[i + 50][j + 50] + 1.f + 0.5f, j + 0.5f }, 0.5f, GRASS));
            if (isTree[i + 50][j + 50])
            {
                int height = 2 + rand() % 3;
                createBuildTree(i + 0.5f, worldHeightList[i + 50][j + 50] + 2.f + 0.5f, j + 0.5f, height);
            }
        }
    }
    

}
int readData()
{
    std::cout << "Select your saves:" << std::endl;
    std::cout << "       0 : Auto generate" << std::endl;
    std::cout << "       1 : Creator" << std::endl;
    std::cout << "       2 : Test" << std::endl;
    std::cout << "       3 : Presentor1" << std::endl;
    int savefilenum = -1;
    while (scanf_s("%d",&savefilenum) != 1)
    {
        std::cout << "input_error : please imput an int" << std::endl;
    }
    if (!(savefilenum >= 0 && savefilenum <= 3))
    {
        std::cout << "intput_error : number error" << std::endl;
        return 1;
    }
    if (savefilenum == 0)
    {
        initData();
    }
    else if (savefilenum == 1)
    {
        std::ifstream savefile(sutil::sampleDataFilePath("Saves/SAVE0.txt"), std::ios::in);
        if (!savefile)
        {
            std::cout << "failed to load savefiles" << std::endl;
            return 1;
        }
        else {
            int savecnt = 0;
            float savedx, savedy, savedz, saveds;
            int savedTexture;
            while (!savefile.eof())
            {
                savecnt++;
                savefile >> savedx >> savedy >> savedz >> saveds >> savedTexture;
                modelLst.push_back(new cCube({ savedx, savedy, savedz }, saveds, (ModelTexture)savedTexture));
            }
            std::cerr << "Successfully loaded " << savecnt << " Cubes" << std::endl;
            savefile.close();
        }
    }
    else if (savefilenum == 2)
    {
        for(int i=0; i<100; i++) 
        {
            for(int j=0; j<100; j++) 
            {
                    modelLst.push_back(new cCube({1.f*i + 0.5f, 0.5f, 1.f*j + 0.5f}, 0.5f, GRASS));
            }
        }
    }
}
void saveData()
{
    //save Blocks
    std::ofstream savefile(sutil::sampleDataFilePath("Saves/SAVE0.txt"), std::ios::out);
    if (!savefile) std::cout << "failed to load savefiles" << std::endl;
    else {
        int savecnt = 0;
        for (auto& mp : modelLst)
        {
            if (mp->get_type() == "Cube")
            {
                savecnt++;
                savefile << " " << mp->get_center().x << " " << mp->get_center().y << " " << mp->get_center().z <<" "<< mp->get_collideBox().size.x << " " << mp->texture_id;
            }
        }
        std::cerr << "Successfully saved " << savecnt << " Cubes" << std::endl;
        savefile.close();
    }
    
}
void initCreature()
{
    //Create a Player
    switchcam = true;
    Creature* enttmp = new Creature;
    crtList.push_back(enttmp);
    control = player = (Creature*)crtList[0];
    player->pos = make_float3(8.0f, 19.7f, -4.0f);
    player->eye = make_float3(8.0f, 21.0f, -4.0f);
    player->lookat = make_float3(4.0f, 2.3f, -4.0f);
    player->up = make_float3(0.0f, 1.0f, 0.0f);
    player->box = CollideBox(make_float3(8.0f, 20.55f, -4.0f), make_float3(0.3f, 0.85f, 0.3f));
}
void initEntitySystem()
{
    initCreature();


    //@@todo: Create other entities 
}
void initCameraState()
{
    camera.setEye( make_float3( 8.0f, 21.0f, -4.0f ) );
    camera.setLookat( make_float3( 4.0f, 2.3f, -4.0f ) );
    camera.setUp( make_float3( 0.0f, 1.0f, 0.0f ) );
    camera.setFovY( 60.0f );

    trackball.setCamera( &camera );
    trackball.setMoveSpeed( 10.0f );
    trackball.setReferenceFrame( make_float3( 1.0f, 0.0f, 0.0f ), make_float3( 0.0f, 0.0f, 1.0f ), make_float3( 0.0f, 1.0f, 0.0f ) );
    trackball.setGimbalLock(true);
}
void recycleParticles()//??????????????????????????? only used when program is end. 
{
    for (auto& ptc : ptcList)
    {

    }
}//??????????????????????????????
void updateParticle(float dt)//the motion of particles in dt time
{
    float nowTime = glfwGetTime();/////
    if (!ptcList.empty())
    {
        model_need_update = true;
        for (vector<Particle*>::iterator it = ptcList.begin(); it != ptcList.end();)
        {
            Particle* ptc = *it;
            if (nowTime - ptc->beginTime >= ptc->lifeLength)//??????????????????
            {
                delete* it;
                it = ptcList.erase(it);
                
            }
            else {
                //???????????????????????????
                ptc->velocity += ptc->acceleration;
                ptc->velocity = ptc->velocity + make_float3(0.f, -40.f * dt, 0.f); //?????????????????????????????????????????????????????????????????????????????????
                ptc->acceleration = make_float3(0.f, 0.f, 0.f);
                ptc->dX(ptc->velocity * dt);
                ptc->velocity.x *= 0.9f;
                ptc->velocity.z *= 0.9f;
                ++it;
            }
            
        }
    }
    
}
void updateCreature(float dt)//the motion of entities in dt time
{
    //std::cout << control->isOnGround << std::endl;
    if (!switchcam)
    {
        control->lookat = camera.lookat();
    }
    cModel* tmp = nullptr;
    for (auto& ent : crtList)
    {
        ent->velocity += ent->acceleration;
        if (!ent->isOnGround && !control->isFlying) ent->velocity = ent->velocity + make_float3(0.f, -40.f*dt, 0.f);//???????????????
        ent->acceleration = make_float3(0.f, 0.f, 0.f);
        if (ent->isOnGround)
        {
            ent->dx(ent->velocity.x * dt);
            if (isCollide(ent)) ent->dx(-ent->velocity.x * dt), ent->velocity.x = 0;
            ent->dz(ent->velocity.z * dt);
            if (isCollide(ent)) ent->dz(-ent->velocity.z * dt), ent->velocity.z = 0;
            ent->dy(ent->velocity.y * dt);
            if (isCollide(ent)) ent->dy(-ent->velocity.y * dt), ent->velocity.y = 0;
            if (!get_model_at(ent->box.center + make_float3(0.f,-ent->box.size.y-0.05f,0.f),tmp) 
                && !get_model_at(ent->box.center + make_float3(ent->box.size.x, -ent->box.size.y - 0.05f, ent->box.size.z), tmp)
                && !get_model_at(ent->box.center + make_float3(-ent->box.size.x,-ent->box.size.y - 0.05f, ent->box.size.z), tmp)
                && !get_model_at(ent->box.center + make_float3(ent->box.size.x, -ent->box.size.y - 0.05f, -ent->box.size.z), tmp)
                && !get_model_at(ent->box.center + make_float3(-ent->box.size.x, -ent->box.size.y - 0.05f, -ent->box.size.z), tmp)
                )
                
            {
                ent->isOnGround = false;
            }
        }
        else {
            ent->dx(ent->velocity.x * dt);
            if (isCollide(ent)) ent->dx(-ent->velocity.x * dt),ent->velocity.x = 0;
            ent->dz(ent->velocity.z * dt);
            if (isCollide(ent)) ent->dz(-ent->velocity.z * dt),ent->velocity.z = 0;
            ent->dy(ent->velocity.y * dt);
            if (isCollide(ent))
            {
                if (ent->velocity.y <= 0)
                {
                    cModel* entCollideATBlockhere = nullptr;
                    if (isParticle && get_model_at(ent->box.center - make_float3(0.f, ent->box.size.y + 0.1f, 0.f), entCollideATBlockhere))
                    {
                        createParticles_planeBounce(ent->box.center - ent->box.size - make_float3(0.2f,0.f,0.2f), -0.6 * ent->velocity.y, 4.f, 2.f, 10, 0.01f, entCollideATBlockhere->texture_id);
                    }
                    ent->isOnGround = true;
                }  
                ent->dy(-ent->velocity.y * dt);
                ent->velocity.y = 0;
            }
        }
        

        ent->velocity.x *= 0.7f;
        ent->velocity.z *= 0.7f;
        if (control->isFlying) ent->velocity *= 0.7;
        
        
        if (ent->pos.y <= 0.f)
        {
            float delta = 0 - ent->pos.y;
            ent->dX(make_float3(0, delta, 0));
            switchcam = true;
            ent->velocity.y = 0.f;
            ent->isOnGround = true;
        }
    }

}
void updateControl(float dt)//from keyboard to *contol
{

    float3 direction(make_float3(0.0f));
    if (!control->isFlying)
    {
        if (wscnt != 0 || adcnt != 0)
        {
            float3 camera_target_vector = camera.direction();
            float3 camera_normal_vector = cross(camera.up(), camera.direction());
            if (key_value['w']) direction += normalize(make_float3(camera_target_vector.x, 0, camera_target_vector.z));
            if (key_value['s']) direction -= normalize(make_float3(camera_target_vector.x, 0, camera_target_vector.z));
            if (key_value['a']) direction += normalize(make_float3(camera_normal_vector.x, 0, camera_normal_vector.z));
            if (key_value['d']) direction -= normalize(make_float3(camera_normal_vector.x, 0, camera_normal_vector.z));
            direction = normalize(direction);

            //model_need_update = true;
        }   
    }
    else {
        if (wscnt != 0 || adcnt != 0 || sccnt !=0)
        {
            float3 camera_target_vector = camera.direction();
            float3 camera_normal_vector = cross(camera.up(), camera.direction());
            if (key_value['w']) direction += normalize(make_float3(camera_target_vector.x, 0, camera_target_vector.z));
            if (key_value['s']) direction -= normalize(make_float3(camera_target_vector.x, 0, camera_target_vector.z));
            if (key_value['a']) direction += normalize(make_float3(camera_normal_vector.x, 0, camera_normal_vector.z));
            if (key_value['d']) direction -= normalize(make_float3(camera_normal_vector.x, 0, camera_normal_vector.z));
            if (key_value['_']) direction += normalize(make_float3(0, 1, 0));
            if (key_value['c']) direction -= normalize(make_float3(0, 1, 0));
            direction = normalize(direction);
            
            //model_need_update = true;
        }
    }
    
    control->da(camera_speed * direction);

    if (sprint)
    {
        if (camera.fovY() < 70.f) camera.setFovY(camera.fovY() + dt * 120.f);
    }
    else {
        if (camera.fovY() > 60.f) camera.setFovY(camera.fovY() - dt * 120.f);
    }
}

void handleCameraUpdate( WhittedState &state)
{
    //modifing camera through entity data
    if (switchcam)
    {
        camera.setEye(control->eye);
        camera.setLookat(control->lookat);
        switchcam = false;
    }
    else {
        camera.setLookat(control->lookat);
        camera.setEye(control->eye);
    }
    //end modifing

    camera.setAspectRatio( static_cast<float>( state.params.width ) / static_cast<float>( state.params.height ) );
    

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

void handleTimeUpdate( WhittedState& state )
{
    
}

void updateState( sutil::CUDAOutputBuffer<uchar4>& output_buffer, WhittedState &state)
{
    // Update params on device

    state.params.subframe_index = 0;
    handleTimeUpdate( state );
    handleCameraUpdate( state );
    handleResize( output_buffer, state.params );

    // if we place a new model, then update
    if(detectModelUpdate(state)) {
        output_buffer.setStream( state.stream );
    }
}

void updateInteration()
{
    //update MODEL
    float3 vec = camera.lookat() - camera.eye();
    bool isintersect = false;
    cModel* mp = nullptr;
    float3 startp = camera.eye();
    if (get_model_at(startp, mp))//?????????????????????????????????????????????
    {
        istargeted = false;//?????????????????????
        
    }
    else {
        float3 nextp = nearCeil(startp, vec);
        int findBlockcnt = 10;//???25?????????
        while (findBlockcnt--)
        {
            startp = nextp;
            nextp = nearCeil(startp, vec);
            if (get_model_at((startp + nextp) / 2, mp))
            {
                istargeted = true;
                isintersect = true;
                break;
            }
        }
        if (!isintersect)
        {
            istargeted = false;
        }
        else {
            istargeted = true;
            intersectBlock = mp;
            intersectPoint = startp;
        }
    }


    //updateCreature
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
    OPTIX_CHECK( optixProgramGroupDestroy ( state.radiance_texture_sphere_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.occlusion_texture_sphere_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.radiance_glass_sphere_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.occlusion_glass_sphere_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy  ( state.radiance_texture_cube_prog_group   ) );
    OPTIX_CHECK( optixProgramGroupDestroy  ( state.occlusion_texture_cube_prog_group  ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.radiance_miss_prog_group         ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.radiance_floor_prog_group        ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.occlusion_floor_prog_group       ) );
    OPTIX_CHECK( optixModuleDestroy       ( state.shading_module          ) );
    OPTIX_CHECK( optixModuleDestroy       ( state.geometry_module         ) );
    OPTIX_CHECK( optixModuleDestroy       ( state.camera_module           ) );
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
    state.params.width  = 1024;
    state.params.height = 768;
    sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;
    sky.init();
    sky.setSunTheta(0);  // 0: noon, pi/2: sunset
    sky.setSunPhi(DEFAULT_SUN_PHI);
    sky.setTurbidity(2.2f);
    //Split out sun for direct sampling
    sun.direction = sky.getSunDir();
    Onb onb(sun.direction);
    sun.radius = DEFAULT_SUN_RADIUS/15;
    sun.v0 = onb.m_tangent;
    sun.v1 = onb.m_binormal;
    //float3 dir({ 1.0f,0.0f,0.0f });
    //sun.v0 = cross(sun.direction, dir);
    //sun.v1 = cross(sun.v0, sun.direction);
    const float sqrt_sun_scale = PHYSICAL_SUN_RADIUS / sun.radius;
    sun.color = sky.sunColor() * sqrt_sun_scale * sqrt_sun_scale;
    sun.casts_shadow = 1;

    // Image credit: CC0Textures.com (https://cc0textures.com/)
    // Licensed under the Creative Commons CC0 License.
    load_texture_integrated("Wood049", WOOD);
    load_texture_integrated("Planks021", PLANK);
    load_texture_integrated("Bricks059", BRICK);
    load_texture_integrated("Ground048", DIRT);
    load_texture_integrated("Ground037", GRASS);
    load_texture("Ground786.jpg", "GRASS_side_diffuse");
    load_texture_integrated("Metal003", IRON);
    load_texture_integrated("bark1", BARK);
    load_texture("stripped_oak_log_top.png", "BARK_top_diffuse");
    load_texture_integrated("Leaves002", LEAF);
    load_texture_integrated("Gravel022", GRAVEL);
    load_texture_integrated("Metal034", GOLD);
    load_texture_integrated("Tiles101", CHINA);
    load_texture_integrated("MetalPlates006", STEEL);
    load_texture_integrated("Rock030", ROCK);
    load_texture("clear_sky.jpg", "NOON_MAP");
    load_texture("sky1.jpg", "MORNING_MAP");
    load_texture("night.jpg", "NIGHT_MAP");
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
        readData();
        initEntitySystem();

        initCameraState();

        //
        // Set up OptiX state
        //
        createContext  ( state );
        createGeometry  ( state );
        createPipeline ( state );
        createTextures(  );
        createSBT      ( state);

        initLaunchParams( state );


        //
        // Render loop
        //
        if( outfile.empty() )
        {
            GLFWwindow* window = sutil::initUI( "OptiX7Craft", state.params.width, state.params.height );
            glfwSetMouseButtonCallback  ( window, mouseButtonCallback   );
            glfwSetCursorPosCallback    ( window, cursorPosCallback     );
            glfwSetScrollCallback       ( window, mouseScrollCallback   );
            glfwSetWindowSizeCallback   ( window, windowSizeCallback    );
            glfwSetWindowIconifyCallback( window, windowIconifyCallback );
            glfwSetKeyCallback          ( window, keyCallback           );
            glfwSetWindowUserPointer    ( window, &state.params         );
           glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
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
                    float currentframe = glfwGetTime();
                    deltatime = currentframe - lastframe;
                    lastframe = currentframe;

                    auto t0 = std::chrono::steady_clock::now();
                    glfwPollEvents();

                    



                    //----------------------------sun updating----------------------------
                    game_time = fmod(circle / 4.f + glfwGetTime() , circle);
                    float sunAngle = sunAngleScaling((game_time - circle / 4.f) / (circle / 4.f) * ( M_PI / 2.f) );
                    //std::cout << sunAngle << std::endl;
                    sky.setSunTheta(sunAngle);
                    sun.direction = sky.getSunDir();
                    Onb t_onb(sun.direction);
                    sun.radius = DEFAULT_SUN_RADIUS;
                    sun.v0 = t_onb.m_tangent;
                    sun.v1 = t_onb.m_binormal;
                    const float sqrt_sun_scale_time = PHYSICAL_SUN_RADIUS / sun.radius;
                    sun.color = sky.sunColor() * sqrt_sun_scale_time * sqrt_sun_scale_time;
                    sun.casts_shadow = 1;
                    //state.params.sun = sun;
                    state.params.sky = sky;
                    state.params.game_time = game_time;
                    if (renewShadowOnTime)
                    {
                        model_need_update = true;
                    }
                    





                    updateControl(deltatime);
                    updateParticle(deltatime);
                    updateCreature(deltatime);
                    updateState( output_buffer, state);
                    
                    updateInteration(); // ?????????????????????????????????

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

                    
                    displayHUD(state.params.width, state.params.height);
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
            float currentframe = glfwGetTime();
            deltatime = currentframe - lastframe;
            lastframe = currentframe;
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
