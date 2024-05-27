#include "Light.h"

/*
    Light ~ light class
    Diego Párraga Nicolás ~ diegojose.parragan@um.es
*/

Light::Light(RGBA color, float intensity, Vec3 dirPos, LightType type) : color(color), intensity(intensity), dirPos(dirPos), type(type) {}