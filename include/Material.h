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
    RGBA* texture, * bump;
    RGBA color;
    float diff, spec, reflective, refractive;
    __host__ __device__ Material(float diff, float spec, float reflective, float refractive);
    void loadTexture(const char* path);
    void loadBump(const char* path);
};

#endif