#ifndef __CAMERA_H__
#define __CAMERA_H__

/*
    Camera ~ world camera class header
    Diego Párraga Nicolás ~ diegojose.parragan@um.es
*/

#include "POLY-cuda.h"
#include "Vec.h"

class Ray {
public:
    Vec3 ori, dir;
    Vec4 color;
    uint8_t medium = 0u;
    __device__ Ray();
    __device__ Ray(Vec3, Vec3);
    __device__ Vec3 point(float);
};

class Camera{
public:
    Vec3 ori, lookAt;
    float fov;
    __host__ __device__ Camera(Vec3 ori, Vec3 lookAt, float fov);
    __device__ Ray rayTo(uint16_t x, uint16_t y);
};

#endif