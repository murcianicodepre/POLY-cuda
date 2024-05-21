#include "PolyShaders.h"
#include "Vec.h"
#include "RGBA.h"

/*
    PolyShaders ~ GPU shaders for poly-cuda
    Diego Párraga Nicolás ~ diegojose.parragan@um.es
*/

// Main pixel rendering entrypoint
__global__ void compute_pixel(RGBA* frame, uint16_t flags16){

    // Get global pixel coordinates
    uint2 pixelCoord = make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

    // Compute pixel color
    Vec4 out = Vec4(1.0f * threadIdx.x / blockDim.x, 0.0f, 1.0f * threadIdx.y / blockDim.y, 1.0f);

    // Store pixel color in global memory buffer
    frame[pixelCoord.x + pixelCoord.y * WIDTH] = RGBA(out);
    
    // Wait in a barrier until all threads have finished
    __syncthreads();
}

// Computes closest tri intersection
__device__ bool intersection_shader(Ray&, Hit&, uint8_t discard){

}

// Computes blinn-phong shading for a given hit point
__device__ Vec3 blinn_phong_shading(Hit&, uint8_t flags){

}

// Computes flat shading for a given hit point
__device__ Vec3 flat_shading(Hit&){

}

// Computes fragment color for a given hit
__device__ Vec4 fragment_shader(Hit&, uint8_t flags){

}

// Maps a texture pixel to a hit
__device__ Vec4 texture_mapping(Hit&){

}

// Computes hit surface normal using a bump texture
__device__ Vec3 bump_mapping(Hit&){

}