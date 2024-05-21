#ifndef __RGBA_H__
#define __RGBA_H__

/*
    RGBA ~ rgba header for PolyRenderer
    Diego Párraga Nicolás ~ diegojose.parragan@um.es
*/

#include "POLY-cuda.h"

class RGBA{
public:
    uint8_t r,g,b,a;
    __host__ __device__ RGBA();
    __host__ __device__ RGBA(uint8_t r, uint8_t g, uint8_t b, uint8_t a);
    __host__ __device__ RGBA(Vec3 vec, float a = 1.0f);
    __host__ __device__ RGBA(Vec4 frag);
};

#endif