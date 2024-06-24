/*
    Vec.cu ~ Vector lib for Poly cuda
    Diego Párraga Nicolás ~ diegojose.parragan@um.es
*/

#include "Vec.h"
#include "RGBA.h"

Vec3::Vec3() : x(0.0f), y(0.0f), z(0.0f) {}
Vec3::Vec3(float f) : x(f), y(f), z(f) {}
Vec3::Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
Vec3::Vec3(RGBA rgb) :
    x(static_cast<float>(rgb.r)/255.0f),
    y(static_cast<float>(rgb.g)/255.0f),
    z(static_cast<float>(rgb.b)/255.0f) {}
Vec3::Vec3(float4 xyzw) : x(xyzw.x), y(xyzw.y), z(xyzw.z) {}
Vec3 Vec3::operator+(float f) { return Vec3(x+f, y+f, z+f); }
Vec3 Vec3::operator+(Vec3 v) { return Vec3(x+v.x, y+v.y, z+v.z); }
Vec3 Vec3::operator-(float f) { return Vec3(x-f, y-f, z-f); }
Vec3 Vec3::operator-(Vec3 v) { return Vec3(x-v.x, y-v.y, z-v.z); }
Vec3 Vec3::operator*(float f) { return Vec3(x*f, y*f, z*f); }
Vec3 Vec3::operator*(Vec3 v) { return Vec3(x*v.x, y*v.y, z*v.z); }
float Vec3::operator[](uint8_t i) { return i==0 ? x : (i==1 ? y : z); }
bool Vec3::operator==(float f) { return x==f && y==f && z==f; }
float Vec3::length(){ return sqrtf(x*x + y*y + z*z); }
Vec3 Vec3::normalize(){ float m = length(); return (m>0.0f) ? Vec3(x/m, y/m, z/m) : Vec3(x,y,z); }
void Vec3::rotateX(float r){
    float a = r*ALPHA, y0 = y, z0 = z, cosa = cosf(a), sina = sinf(a);
    y = y0 * cosa - z0 * sina;
    z = y0 * sina + z0 * cosa;
}
void Vec3::rotateY(float r){
    float a = r*ALPHA, x0 = x, z0 = z, cosa = cosf(a), sina = sinf(a);
    x = x0 * cosa + z0 * sina;
    z = z0 * cosa - x0 * sina;
}
void Vec3::rotateZ(float r){
    float a = r*ALPHA, x0 = x, y0 = y, cosa = cosf(a), sina = sinf(a);
    x = x0 * cosa - y0 * sina;
    y = x0 * sina + y0 * cosa;
}
void Vec3::rotate(Vec3 rot){ rotateY(rot.y); rotateX(rot.x); rotateZ(rot.z); }
float Vec3::dot(Vec3 a, Vec3 b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
Vec3 Vec3::cross(Vec3 a, Vec3 b){ return Vec3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); }
Vec3 Vec3::max(Vec3 a, Vec3 b){ return Vec3(a.x >= b.x ? a.x : b.x, a.y >= b.y ? a.y : b.y, a.z >= b.z ? a.z : b.z); }
Vec3 Vec3::min(Vec3 a, Vec3 b){ return Vec3(a.x <= b.x ? a.x : b.x, a.y <= b.y ? a.y : b.y, a.z <= b.z ? a.z : b.z); }
Vec3 Vec3::clamp(float max, float min){
    x = (x>max) ? max : ((x<min) ? min : x);
    y = (y>max) ? max : ((y<min) ? min : y);
    z = (z>max) ? max : ((z<min) ? min : z);
    return Vec3(x,y,z);
}

Vec4::Vec4() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
Vec4::Vec4(float f) : x(f), y(f), z(f), w(f) {}
Vec4::Vec4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
Vec4::Vec4(Vec3 xyz, float w) : x(xyz.x), y(xyz.y), z(xyz.z), w(w) {}
Vec4::Vec4(RGBA rgba) :
    x(static_cast<float>(rgba.r)/255.0f),
    y(static_cast<float>(rgba.g)/255.0f),
    z(static_cast<float>(rgba.b)/255.0f),
    w(static_cast<float>(rgba.a)/255.0f) {}
Vec4::Vec4(float4 xyzw) : x(xyzw.x), y(xyzw.y), z(xyzw.z), w(xyzw.w) {} 
Vec4 Vec4::operator+(float f){ return Vec4(x+f, y+f, z+f, w+f); }
Vec4 Vec4::operator+(Vec4 v){ return Vec4(x+v.x, y+v.y, z+v.z, w+v.w); }
Vec4 Vec4::operator+(Vec3 v){ return Vec4(x+v.x, y+v.y, z+v.z, w); }
Vec4 Vec4::operator-(float f){ return Vec4(x-f, y-f, z-f, w-f); }
Vec4 Vec4::operator-(Vec4 v){ return Vec4(x-v.x, y-v.y, z-v.z, w-v.w); }
Vec4 Vec4::operator-(Vec3 v){ return Vec4(x-v.x, y-v.y, z-v.z, w); }
Vec4 Vec4::operator*(float f){ return Vec4(x*f, y*f, z*f, w*f); }
Vec4 Vec4::operator*(Vec4 v){ return Vec4(x*v.x, y*v.y, z*v.z, w*v.w); }
Vec4 Vec4::operator*(Vec3 v){ return Vec4(x*v.x, y*v.y, z*v.z, w); }
bool Vec4::operator==(float f) { return x==f && y==f && z==f && w==f; }
Vec4 Vec4::clamp(float max, float min){
    x = (x>max) ? max : ((x<min) ? min : x);
    y = (y>max) ? max : ((y<min) ? min : y);
    z = (z>max) ? max : ((z<min) ? min : z);
    w = (w>max) ? max : ((w<min) ? min : w);
    return Vec4(x,y,z,w);
}