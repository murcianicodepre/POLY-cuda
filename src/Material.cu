#include "Material.h"
#include "PolyRenderer.h"

/*
    Material ~ Material class
    Diego Párraga Nicolás ~ diegojose.parragan@um.es
*/

Texture::Texture() : _obj(), _array() {}
Texture::Texture(const char* path){
    RGBA* tex = PolyRenderer::loadPNG(path, false);

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
    cudaerr(cudaMallocArray(&_array, &desc, TEXTURE_SIZE, TEXTURE_SIZE));
    cudaerr(cudaMemcpyToArray(_array, 0,0, (void*) tex, sizeof(RGBA) * TEXTURE_SIZE * TEXTURE_SIZE, cudaMemcpyHostToDevice));

    struct cudaResourceDesc res;
    memset(&res, 0, sizeof(res));
    res.resType = cudaResourceTypeArray;
    res.res.array.array = _array;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap; texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 1;

    _obj = 0;
    cudaerr(cudaCreateTextureObject(&_obj, &res, &texDesc, NULL));

    free(tex);
}
__device__ Vec4 Texture::tex2d(float u, float v){
    uchar4 tex = tex2D<uchar4>(_obj, u, v);
    return Vec4(tex.x / 255.0f, tex.y / 255.0f, tex.z / 255.0f, tex.w / 255.0f);
}

Material::Material(float diff, float spec, float reflective, float refractive) : texture(NULL), bump(NULL), diff(diff), spec(spec), reflective(reflective), refractive(refractive) {}
void Material::loadTexture(const char* path){ 
    texture = PolyRenderer::loadPNG(path); 
    // tex = Texture(path);
}
void Material::loadBump(const char* path){ bump = PolyRenderer::loadPNG(path); }