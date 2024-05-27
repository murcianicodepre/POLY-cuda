#ifndef __MATERIAL_H__
#define __MATERIAL_H__

/*
    Material ~ Material class header
    Diego Párraga Nicolás ~ diegojose.parragan@um.es
*/

#include "POLY-cuda.h"
#include "RGBA.h"

class Texture {
public:
    cudaArray_t _array;
    cudaTextureObject_t _obj;
    Texture();
    Texture(const char*);
    __device__ Vec4 tex2d(float, float);
};

class Material{
public:
    Texture tex;
    RGBA* texture, * bump;
    RGBA color;
    float diff, spec, reflective, refractive;
    Material(float diff, float spec, float reflective, float refractive);
    void loadTexture(const char* path);
    void loadBump(const char* path);
};

#endif