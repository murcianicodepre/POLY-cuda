#include "Camera.h"

/*
    Camera ~ world camera class
    Diego Párraga Nicolás ~ diegojose.parragan@um.es
*/

__device__ Ray::Ray() : ori(), dir(), color() {}
__device__ Ray::Ray(Vec3 ori, Vec3 dir) : ori(ori), dir(dir), color() {}
__device__ Vec3 Ray::point(float t){ return dir * t + ori; }

Camera::Camera(Vec3 ori, Vec3 lookAt, float fov) : ori(ori), lookAt(lookAt), fov(fov*ALPHA) {}
__device__ Ray Camera::rayTo(uint16_t x, uint16_t y){
    float aux = tanf(fov / 2.0f);
    float px = (2.0f * ((x + 0.5f)/WIDTH) - 1.0f) * aux * AR;
    float py = (1.0f - (2.0f * (y + 0.5f)/HEIGHT)) * aux;

    Vec3 dir = Vec3(px, py, 1.0f);

    Vec3 f = (lookAt - ori).normalize(), r = Vec3::cross(f, Vec3(0.0f, 1.0f, 0.0f)).normalize(), u = Vec3::cross(r, f);

    dir = (r * dir.x) * -1.0f + u * dir. y + f * dir.z;

    return Ray(ori, dir.normalize());
}