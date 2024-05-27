#ifndef __POLY_SHADERS_H__
#define __POLY_SHADERS_H__

#include "POLY-cuda.h"

/*
    PolyShaders ~ GPU shaders for poly-cuda
    Diego Párraga Nicolás ~ diegojose.parragan@um.es
*/

__global__ void compute_pixel(RGBA*, Scene*);
__device__ bool intersection_shader(Ray&, Hit&, Scene*, uint8_t discard = 0x00u);
__device__ Vec3 blinn_phong_shading(Hit&, Scene*, uint8_t flags = 0x00u);
__device__ Vec3 flat_shading(Hit&);
__device__ Vec4 fragment_shader(Hit&, Scene*, uint8_t flags = 0x00u);
__device__ Vec4 texture_mapping(Hit&, Material&);
__device__ Vec3 bump_mapping(Hit&, Tri&, Material&);

#endif