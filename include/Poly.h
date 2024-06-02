#ifndef __POLY_H__
#define __POLY_H__

/*
    Poly ~ Vertex, Hit, Tri and Poly class header
    Diego Párraga Nicolás ~ diegojose.parragan@um.es
*/

#include "POLY-cuda.h"
#include "Vec.h"
#include "Camera.h"
#include "PolyRenderer.h"

// Vertex class
class Vertex {
public:
    Vec3 xyz, normal;
    float u, v;
    Vertex() : xyz(), normal(), u(), v() {}
    Vertex(Vec3 xyz, Vec3 normal, float u, float v);
    void move(Vec3 m);
    void scale(Vec3 s);
    void scale(float s);
    void rotate(Vec3 r);
    void rotateX(float r);
    void rotateY(float r);
    void rotateZ(float r);
};

/*
    Hit struct, stores:
        - triId (uint32_t)
        - Surface and Phong normals (Vec3)
        - Texture coordinates and t parameter (float32_t)
*/
struct Hit{
    uint32_t triId = 0u;
    Vec3 normal, phong, bump;
    float u = 0.0f, v = 0.0f, t = __FLT_MAX__;
    Ray ray;
    __device__ Hit() : normal(), phong(), bump(), u(0.0f), v(0.0f), t(__FLT_MAX__), ray() {}
    __device__ Vec3 point(){ return ray.point(t); }
};

// Tri class
class Tri {
public:
    Vertex a, b, c;
    uint8_t flags, matId;
    Tri(Vertex a, Vertex b, Vertex c, uint8_t matId, uint8_t flags);
    __device__ bool intersect(Ray ray, Hit& hit);
    void move(Vec3 m);
    void scale(float s);
    void scale(Vec3 s);
    void rotate(Vec3 r);
    void rotateX(float r);
    void rotateY(float r);
    void rotateZ(float r);
    float min(uint8_t axis);
    float max(uint8_t axis);
    Vec3 centroid();
};

// Poly class
class Poly {
public:
    std::vector<Tri> tris;
    Poly(const char* path, uint8_t mat, uint8_t flags);
    void move(Vec3 m);
    void scale(Vec3 s);
    void scale(float s);
    void rotate(Vec3 r);
    void rotateX(float r);
    void rotateY(float r);
    void rotateZ(float r);
};

#endif