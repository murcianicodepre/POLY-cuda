#ifndef __POLYRENDERER_H__
#define __POLYRENDERER_H__

/*
    PolyRenderer ~ main renderer header
    Diego Párraga Nicolás ~ diegojose.parragan@um.es
*/

#include "POLY-cuda.h"
#include "Vec.h"
#include "RGBA.h"
// #include "Light.h"
#include "Camera.h"
// #include "Poly.h"
#include "PolyShaders.h"

// BVH Acceleration struct node
class BVHNode {
public:
    Vec3 aabbMin, aabbMax;
    uint32_t n, leftOrFirst;
    BVHNode();
    __device__ bool intersectAABB(Ray&);
};

// Main renderer class
class PolyRenderer{
public:
    RGBA* _frame;
    cudaDeviceProp _gpu;
    uint16_t _global = 0x0000u;

    PolyRenderer();
    ~PolyRenderer();

    /*Camera* _cam;
    std::vector<Tri> _tris;
    std::vector<Material> _mats;
    std::vector<std::unique_ptr<Light>> _lights;
    BVHNode* _bvh;*/

    // Cuda related functions
    static void cudaerr(cudaError_t);

    // Main program functions
    bool loadScene(const char* scene);
    bool render();
    void save(const char* path);


    // Acceleration struct
    /*uint32_t _nextNode = 1, * _triIdx = nullptr;
    __host__ void buildBVH();
    __host__ void updateNodeBounds(uint32_t nodeId);
    __host__ void subdivide(uint32_t nodeId);
    __device__ void intersectBVH(Ray& ray, Hit& hit, uint32_t nodeId, uint16_t discard = 0x0000u);

    // Rendering pipeline functions
    __device__ RGBA compute_pixel(uint16_t, uint16_t);
    __device__ bool intersection_shader(Ray&, Hit&, uint16_t discard = 0x0000u);
    __device__ Fragment blinn_phong_shading(Hit&, uint8_t flags = 0x00u);
    __device__ Fragment flat_shading(Hit&);
    __device__ Fragment fragment_shader(Hit&, uint8_t flags = 0x00u);
    __device__ Fragment texture_mapping(Hit&);
    __device__ Vec3 bump_mapping(Hit&);
    __device__ Fragment raytracing_shader(Hit&, uint8_t, uint8_t);
    __device__ Fragment reflection_shader(Hit&, uint8_t, uint8_t);
    __device__ Fragment refraction_shader(Hit&, uint8_t, uint8_t);*/

    // Other renderer functions
    static RGBA* loadPNG(const char* path);
    static void savePNG(const char* path, RGBA* texture);
    static const char* getCpu();
    static const char* getGpu(cudaDeviceProp&);
    static void printIntro();
    static void polyMsg(std::string msg);
    static Vec3 parseVec3(YAML::Node node);
    static RGBA parseColor(YAML::Node node);
    static uint16_t parseFlags(YAML::Node node);
}; 

#endif