#include "Material.h"
#include "PolyRenderer.h"

/*
    Material ~ Material class
    Diego Párraga Nicolás ~ diegojose.parragan@um.es
*/

cudaArray_t createTexture(const char* path, cudaTextureObject_t& texObj){
    // Load texture into host memory
    uchar4* texData = reinterpret_cast<uchar4*>(PolyRenderer::loadPNG(path));

    // Channel descriptor and cuda Array
    cudaArray_t texArray;
    cudaChannelFormatDesc chanDesc = cudaCreateChannelDesc<uchar4>();
    cudaMallocArray(&texArray, &chanDesc, TEXTURE_SIZE, TEXTURE_SIZE);
    cudaMemcpy2DToArray(texArray, 0,0, texData, TEXTURE_SIZE*sizeof(uchar4), TEXTURE_SIZE*sizeof(uchar4), TEXTURE_SIZE, cudaMemcpyHostToDevice);
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = texArray;
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = true;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    
    free(texData);

    return texArray;
}

Material::Material(float diff, float spec, float reflective, float refractive) : texture(0), bump(0), diff(diff), spec(spec), reflective(reflective), refractive(refractive) {}
cudaArray_t Material::loadTexture(const char* path){ return createTexture(path, texture); }
cudaArray_t Material::loadBump(const char* path){ return createTexture(path, bump); }