#include "RGBA.h"
#include "Vec.h"

using namespace std;

/*
    RGBA ~ rgba class for PolyRenderer
    Diego Párraga Nicolás ~ diegojose.parragan@um.es
*/

__host__ __device__ RGBA::RGBA() : r(0u), g(0u), b(0u), a(0u) {}
__host__ __device__ RGBA::RGBA(uint8_t r, uint8_t g, uint8_t b, uint8_t a) : r(r), g(g), b(b), a(a) {}
__host__ __device__ RGBA::RGBA(Vec3 vec, float w) {
    Vec4 aux = vec;
    aux.x = (aux.x>1.0f) ? 1.0f : ((aux.x<0.0f) ? 0.0f : aux.x);
    aux.y = (aux.y>1.0f) ? 1.0f : ((aux.y<0.0f) ? 0.0f : aux.y);
    aux.z = (aux.z>1.0f) ? 1.0f : ((aux.z<0.0f) ? 0.0f : aux.z);
    aux.w = (w > 1.0f) ? 1.0f : ((w < 0.0f) ? 0.0f : w);

    r = static_cast<uint8_t>(aux.x*255.0f);
    g = static_cast<uint8_t>(aux.y*255.0f);
    b = static_cast<uint8_t>(aux.z*255.0f);
    a = static_cast<uint8_t>(aux.w*255.0f);
}
__host__ __device__ RGBA::RGBA(Vec4 frag){

    Vec4 aux = frag;
    aux.x = (aux.x>1.0f) ? 1.0f : ((aux.x<0.0f) ? 0.0f : aux.x);
    aux.y = (aux.y>1.0f) ? 1.0f : ((aux.y<0.0f) ? 0.0f : aux.y);
    aux.z = (aux.z>1.0f) ? 1.0f : ((aux.z<0.0f) ? 0.0f : aux.z);
    aux.w = (aux.w>1.0f) ? 1.0f : ((aux.w<0.0f) ? 0.0f : aux.w);

    r = static_cast<uint8_t>(aux.x*255.0f);
    g = static_cast<uint8_t>(aux.y*255.0f);
    b = static_cast<uint8_t>(aux.z*255.0f);
    a = static_cast<uint8_t>(aux.w*255.0f);
}
