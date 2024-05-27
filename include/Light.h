#ifndef __LIGHT_H__
#define __LIGHT_H__

/*
    Light ~ light class header
    Diego Párraga Nicolás ~ diegojose.parragan@um.es
*/

#include "POLY-cuda.h"
#include "Vec.h"
#include "RGBA.h"

enum class LightType { Ambient, Point, Directional };

class Light {
public:
    RGBA color;
    float intensity;
    Vec3 dirPos;
    LightType type;
    Light(RGBA, float, Vec3, LightType);
};

#endif 