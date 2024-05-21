#ifndef __POLY_SHADERS_H__
#define __POLY_SHADERS_H__

#include "POLY-cuda.h"

/*
    PolyShaders ~ GPU shaders for poly-cuda
    Diego Párraga Nicolás ~ diegojose.parragan@um.es
*/

__global__ void compute_pixel(RGBA*, uint16_t flags16);
__device__ bool intersection_shader(Ray&, Hit&, uint8_t discard = 0x00u);
__device__ Vec3 blinn_phong_shading(Hit&, uint8_t flags = 0x00u);
__device__ Vec3 flat_shading(Hit&);
__device__ Vec4 fragment_shader(Hit&, uint8_t flags = 0x00u);
__device__ Vec4 texture_mapping(Hit&);
__device__ Vec3 bump_mapping(Hit&);

#endif