#ifndef __POLY_CUDA_H__
#define __POLY_CUDA_H__

/*
    POLY cuda ~ main header
    Diego Párraga Nicolás ~ diegojose.parragan@um.es
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cmath>
#include <vector>
#include <string>
#include <cstring>
#include "png.h"
#include "yaml-cpp/yaml.h"
#include "yaml-cpp/exceptions.h"
#include "cuda_runtime.h"
#include "cuda_texture_types.h"
#include "omp.h"

using namespace std;

// Forward class declaration
class Vec3;
class Vec4;
class RGBA;
class Ray;
struct Hit;
class Vertex;
class Tri;
class Material;
class Camera;
class BVHNode;
class Light;
struct Scene;
template<class T> class PolyArray;

// Render pipeline defs
constexpr uint16_t WIDTH = 1280;
constexpr uint16_t HEIGHT = 960;
constexpr uint16_t TEXTURE_SIZE = 1024;
constexpr uint8_t TILE_SIZE = 8u;
constexpr float AR = 1.33333f;
constexpr uint8_t MAX_RAY_BOUNCES = 64u;
constexpr float RAY_BOUNCE_ATT = 1.0f / static_cast<float>(MAX_RAY_BOUNCES);
constexpr uint16_t BVH_STACK_SIZE = 33u; // log2(MAX_TRIS * 2)
constexpr uint8_t SPLIT_PLANES = 64u;
constexpr uint8_t FRAGMENT_COLOR_STEP = 0x1u;

// Individual tri rendering flags
constexpr uint8_t DISABLE_RENDERING = 0x01u;
constexpr uint8_t DISABLE_TEXTURES = 0x02u;
constexpr uint8_t DISABLE_SHADING = 0x04u;
constexpr uint8_t DISABLE_BUMP = 0x08u;
constexpr uint8_t DISABLE_TRANSPARENCY = 0x10u;
constexpr uint8_t DISABLE_SHADOWS = 0x20u;
constexpr uint8_t DISABLE_REFLECTIONS = 0x40u;
constexpr uint8_t DISABLE_REFRACTIONS = 0x80u;

// Rendering pipeline flags
constexpr uint8_t FLAT_SHADING = 0x01u;
constexpr uint8_t DISABLE_SAH = 0x02u;

// Math defs
constexpr float PI = 3.14159264f;
constexpr float ALPHA = 0.017453292f;
constexpr float EPSILON = 1e-6f;

#endif