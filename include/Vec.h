#ifndef __VEC_H__
#define __VEC_H__

/*
    Vec ~ Vector lib for Poly cuda
    Diego Párraga Nicolás ~ diegojose.parragan@um.es
*/

#include "POLY-cuda.h"

class Vec3 {
public:
    float x,y,z;
    
    __host__ __device__ Vec3();
    __host__ __device__ Vec3(float);
    __host__ __device__ Vec3(float, float, float);
    __host__ __device__ Vec3(RGBA rgb);
    __host__ __device__ Vec3 operator+(float);
    __host__ __device__ Vec3 operator+(Vec3);
    __host__ __device__ Vec3 operator-(float);
    __host__ __device__ Vec3 operator-(Vec3);
    __host__ __device__ Vec3 operator*(float);
    __host__ __device__ Vec3 operator*(Vec3);
    __host__ __device__ float operator[](uint8_t i);
    __host__ __device__ bool operator==(float);
    __host__ __device__ float length();
    __host__ __device__ Vec3 normalize();
    __host__ __device__ void rotateX(float);
    __host__ __device__ void rotateY(float);
    __host__ __device__ void rotateZ(float);
    __host__ __device__ void rotate(Vec3);
    __host__ __device__ static float dot(Vec3, Vec3);
    __host__ __device__ static Vec3 cross(Vec3, Vec3);
    __host__ __device__ static Vec3 max(Vec3, Vec3);
    __host__ __device__ static Vec3 min(Vec3, Vec3);
};

class Vec4 {
public:
    float x,y,z,w;
    __host__ __device__ Vec4();
    __host__ __device__ Vec4(float);
    __host__ __device__ Vec4(float, float, float, float);
    __host__ __device__ Vec4(Vec3, float w = 1.0f);
    __host__ __device__ Vec4(RGBA rgba);
    __host__ __device__ Vec4 operator+(float);
    __host__ __device__ Vec4 operator+(Vec4);
    __host__ __device__ Vec4 operator+(Vec3);
    __host__ __device__ Vec4 operator-(float);
    __host__ __device__ Vec4 operator-(Vec4);
    __host__ __device__ Vec4 operator-(Vec3);
    __host__ __device__ Vec4 operator*(float);
    __host__ __device__ Vec4 operator*(Vec4);
    __host__ __device__ Vec4 operator*(Vec3);
};

#endif