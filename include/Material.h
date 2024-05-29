#ifndef __MATERIAL_H__
#define __MATERIAL_H__

/*
    Material ~ Material class header
    Diego Párraga Nicolás ~ diegojose.parragan@um.es
*/

#include "POLY-cuda.h"
#include "RGBA.h"

class Material{
public:
    cudaTextureObject_t texture, bump;
    RGBA color;
    float diff, spec, reflective, refractive;
    Material(float diff, float spec, float reflective, float refractive);
    cudaArray_t loadTexture(const char* path);
    cudaArray_t loadBump(const char* path);
};

#endif