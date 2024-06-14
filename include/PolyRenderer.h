#ifndef __POLYRENDERER_H__
#define __POLYRENDERER_H__

/*
    PolyRenderer ~ main renderer header
    Diego Párraga Nicolás ~ diegojose.parragan@um.es
*/

#include "POLY-cuda.h"
#include "Vec.h"
#include "RGBA.h"
#include "Light.h"
#include "Camera.h"
#include "Poly.h"
#include "Material.h"
#include "PolyShaders.h"

// BVH Acceleration struct node
class BVHNode {
public:
    Vec3 aabbMin, aabbMax;
    uint32_t n, leftOrFirst;
    BVHNode();
    __device__ bool intersectAABB(Ray&);
};

// Fixed size arrays
template<class T> class PolyArray {
public:
    T data;
    uint32_t size;
    PolyArray(T, uint32_t);
    PolyArray() : data(nullptr), size(0) {} 
};

// Contains all the scene data for the GPU
struct Scene {
    Camera* cam;
    PolyArray<Tri*> tris;
    PolyArray<Material*> mats;
    PolyArray<Light*> lights;
    PolyArray<cudaTextureObject_t*> textures;
    BVHNode* bvh;
    uint32_t* triIdx;
    uint16_t global;

    Scene() : cam(nullptr), tris(nullptr, 0), mats(nullptr, 0), lights(nullptr, 0), textures(nullptr, 0), bvh(nullptr), triIdx(nullptr), global(0x0000u) {}
};

// Main renderer class
class PolyRenderer{
public:
    RGBA* _frame;
    cudaDeviceProp _gpu;
    uint16_t _global = 0x0000u;

    vector<Material> _mats;
    vector<Tri> _tris;
    vector<Light> _lights;
    Camera* _cam;
    BVHNode* _bvh;

    vector<cudaArray_t> _cuArrays;

    PolyRenderer();
    ~PolyRenderer();

    // Main program functions
    bool loadScene(const char* scene);
    bool render();
    void save(const char* path);

    // Acceleration struct
    uint32_t _nextNode = 1, * _triIdx = nullptr, _splits = 128u;
    __host__ void buildBVH();
    __host__ void updateNodeBounds(uint32_t nodeId);
    __host__ void subdivide(uint32_t nodeId);
    __host__ float EvaluateSAH(BVHNode&, uint8_t, float);
    __host__ float getBestSplit(BVHNode&, uint8_t&, float&);
    __host__ float getNodeCost(BVHNode&);

    // Other renderer functions
    static RGBA* loadPNG(const char* path);
    static void savePNG(const char* path, RGBA* texture);
    static const char* getCpu();
    static const char* getGpu(cudaDeviceProp&);
    static void printIntro();
    static void polyMsg(std::string msg);
    static Vec3 parseVec3(YAML::Node node);
    static RGBA parseColor(YAML::Node node);
    static uint16_t parseFlags(YAML::Node node, bool print = false);
}; 

// Cuda related functions
void cudaerr(cudaError_t);

#endif