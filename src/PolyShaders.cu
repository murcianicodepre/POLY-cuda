#include "PolyShaders.h"
#include "PolyRenderer.h"
#include "Vec.h"
#include "RGBA.h"

/*
    PolyShaders ~ GPU shaders for poly-cuda
    Diego Párraga Nicolás ~ diegojose.parragan@um.es
*/

struct Entry {
    Ray ray;
    Hit hit;
    Vec4 color;
    bool hasIntersection = false, hasReflection = false, hasRefraction = false;
};

// Main pixel rendering entrypoint
__global__ void compute_pixel(RGBA* frame, Scene* scene){

    // Get global pixel coordinates
    uint2 pixelCoord = make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

    // Rendering loop
    Entry stack[MAX_RAY_BOUNCES + 1];
    uint8_t i=0;
    stack[i++] = { .ray = scene->cam->rayTo(pixelCoord.x, pixelCoord.y) };
    while(i>0){
        Entry& entry = stack[i-1];

        if(!entry.hasIntersection)
            entry.hasIntersection = intersection_shader(entry.ray, entry.hit, scene);

        if(entry.hasIntersection){
            Hit& hit = entry.hit;
            Tri& tri = scene->tris.data[hit.triId];
            Material& mat = scene->mats.data[tri.matId];
            uint16_t flags16 = scene->global | tri.flags;
            uint8_t cont = i-1;

            // REFLECTION STEP
            if((i<=MAX_RAY_BOUNCES) && (!entry.hasReflection) && (mat.reflective>0.0f) && !(flags16 & DISABLE_REFLECTIONS)){
                entry.hasReflection = true;
                Vec3 rdir = hit.ray.dir - hit.phong * (Vec3::dot(hit.ray.dir, hit.phong) * 2.0f);
                Ray rray = Ray(hit.point(), rdir); rray.medium = tri.matId;
                stack[i++] = { .ray = rray};
            }

            // REFRACTION STEP
            if((i<=MAX_RAY_BOUNCES) && (!entry.hasRefraction) && (mat.refractive<1.0f || mat.refractive>1.0f) && !(flags16 & DISABLE_REFRACTIONS)){
                entry.hasRefraction = true;

                Material& oldMat = scene->mats.data[entry.hit.ray.medium];
                
                float cosTheta1 = Vec3::dot(hit.ray.dir, hit.phong) * -1.0f, theta1 = acosf(cosTheta1);
                float sinTheta2 = (oldMat.refractive / mat.refractive) * sinf(theta1);

                Ray rray;

                if(sinTheta2<1.0f){
                    // Refraction
                    Vec3 rdir = (hit.ray.dir + hit.phong * cosTheta1) * (oldMat.refractive / mat.refractive) - hit.phong * cosTheta1;
                    rray = Ray(hit.point(), rdir); rray.medium = tri.matId;
                } else {
                    // Internal reflection
                    Vec3 rdir = hit.ray.dir - hit.phong * (Vec3::dot(hit.ray.dir, hit.phong) * 2.0f);
                    rray = Ray(hit.point(), rdir); rray.medium = tri.matId;
                }

                stack[i++] = { .ray = rray };
            }

            // Check if all child entries have been resolved
            if(cont == (i-1)){
                Vec4 frag = (entry.hasRefraction) ? fragment_shader(entry.hit, scene, DISABLE_SHADING) : fragment_shader(hit, scene);
                Vec4 color = ((entry.hasRefraction) ? stack[i].color * frag * Vec3(0.9f) : frag) * Vec3(1.0f - mat.reflective) + stack[i].color * Vec3(mat.reflective);
                
                stack[--i].color = color;

            }
        } else i--;
    }
    frame[pixelCoord.x + pixelCoord.y * WIDTH] = RGBA(stack[0].color);
    
    // Wait in a barrier until all threads have finished
    __syncthreads();
}

// Computes closest tri intersection
__device__ bool intersection_shader(Ray& ray, Hit& hit, Scene* scene, uint8_t discard){
    Tri* tris = scene->tris.data;
    Material* mats = scene->mats.data;
    BVHNode* bvh = scene->bvh;
    uint32_t* triIdx = scene->triIdx;

    uint32_t stack[BVH_STACK_SIZE];
    uint8_t i = 0;
    stack[i++] = 0;
    while(i>0){
        BVHNode& node = bvh[stack[--i]];
        if(!node.intersectAABB(ray)) continue;
        if(node.n > 0){
            Hit aux;
            for(uint32_t t=node.leftOrFirst; t<(node.n + node.leftOrFirst); t++){
                Tri& tri = tris[triIdx[t]];
                Material& mat = mats[tri.matId];
                uint16_t flags16 = scene->global | tri.flags;

                // Discard tri if matches with discard flag
                if(tri.flags & discard) continue;

                // Intersection and depth tests
                if(tri.intersect(ray, aux) && (aux.t < hit.t)){
                    hit = aux; hit.triId = triIdx[t]; hit.ray = ray;
                    
                    // BUMP MAPPING step
                    hit.phong = (!(flags16 & DISABLE_BUMP) && !((flags16>>8) & FLAT_SHADING)) ?
                                bump_mapping(hit, tri, mat) :
                                hit.phong;
                }
            }
        } else {
            stack[i++] = node.leftOrFirst;
            stack[i++] = node.leftOrFirst+1;
        }
    }

    return hit.t < __FLT_MAX__;
}

// Computes blinn-phong shading for a given hit point
__device__ Vec3 blinn_phong_shading(Hit& hit, Scene* scene, uint8_t flags){
    Tri* tris = scene->tris.data;
    Material* mats = scene->mats.data;

    Tri& tri = tris[hit.triId];
    Material& mat = mats[tri.matId];
    uint16_t flags16 = (flags | tri.flags | scene->global);

    Vec3 shading, view = (scene->cam->ori - hit.point()).normalize();
    if(!(flags16 & DISABLE_SHADING)){
        for(uint8_t i=0; i<scene->lights.size; i++){
            Light l = scene->lights.data[i];
            if(l.intensity < 1e-3f) continue;
            Vec3 ldir, lpos, half;
            float att, dist;

            // Compute ldir and att depending on the light type
            if(l.type==LightType::Point){
                lpos = l.dirPos;
                ldir = ((lpos - hit.point())).normalize();
                dist = (lpos - hit.point()).length();
                att = 1.0f / (1.0f + 0.14f * dist + 0.07f * (dist * dist));
            } else if(l.type==LightType::Directional){
                lpos = Vec3(1000.0f);
                ldir = l.dirPos.normalize();
                dist = __FLT_MAX__;
                att = 1.0f;
            } else { shading = shading + Vec3(l.color) * l.intensity; continue; }

            // Compute if geometry exists between the fragment and the light origin
            if(!(flags16 & DISABLE_SHADOWS)){
                Vec3 sori = Vec3::dot(ldir, hit.normal) < 0.0f ? hit.point() + hit.phong * EPSILON : hit.point() - hit.phong * EPSILON;
                Ray lray = Ray(sori, ldir);
                Hit aux;
                float t = (lpos - sori).x / ldir.x;
                if(intersection_shader(lray, aux, scene, DISABLE_SHADING | DISABLE_SHADOWS) && (aux.t < t)) continue;
            }

            // Compute specular and diffuse components
            float diff = max(0.0f, Vec3::dot(hit.phong, ldir));
            Vec3 diffuse = (Vec3(l.color) * l.intensity) * (diff * mat.diff);
            
            half = (ldir + view).normalize();
            float spec = powf(max(0.0f, Vec3::dot(hit.phong, half)), mat.spec);
            Vec3 specular = (Vec3(l.color) * l.intensity) * spec;

            shading = shading + (diffuse + specular) * att;
        }
    } else shading = Vec3(1.0f);

    return shading;
}

// Computes flat shading for a given hit point
__device__ Vec3 flat_shading(Hit& hit){
    return Vec3(1.0f) * max(0.0f, Vec3::dot(hit.normal, (hit.ray.ori - hit.point()).normalize()));
}

// Computes fragment color for a given hit
__device__ Vec4 fragment_shader(Hit& hit, Scene* scene, uint8_t flags){
    Tri& tri = scene->tris.data[hit.triId];
    Material& mat = scene->mats.data[tri.matId];
    uint16_t flags16 = (tri.flags | flags | scene->global);

    // TEXTURE MAPPING step
    Vec4 tex = (!(flags16 & DISABLE_TEXTURES) && !((flags16>>8) & FLAT_SHADING)) ?
                texture_mapping(hit, mat) :
                Vec4(mat.color);

    // SHADING step
    Vec3 shading = ((flags16 >> 8) & FLAT_SHADING) ? flat_shading(hit) : blinn_phong_shading(hit, scene, flags);

    return tex * shading;
}

// Maps a texture pixel to a hit
__device__ Vec4 texture_mapping(Hit& hit, Material& mat){
    uint2 pixelCoord = make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    return (mat.texture) ? Vec4(tex2D<float4>(mat.texture, hit.u, hit.v)) : Vec4(mat.color);
}

// Computes hit surface normal using a bump texture
__device__ Vec3 bump_mapping(Hit& hit, Tri& tri, Material& mat){
    if(mat.bump){
        // Compute tangent and bitangent
        Vec3 e1 = tri.b.xyz - tri.a.xyz, e2 = tri.c.xyz - tri.a.xyz;
        float du1 = tri.b.u - tri.a.u, dv1 = tri.b.v - tri.a.v;
        float du2 = tri.c.u - tri.a.u, dv2 = tri.c.v - tri.a.v;
        float f = 1.0f / (du1 * dv2 - du2 * dv1);

        Vec3 tangent = Vec3(f * (dv2 * e1.x - dv1 * e2.x), f * (dv2 * e1.y - dv1 * e2.y), f * (dv2 * e1.z - dv1 * e2.z)).normalize();
        Vec3 bitangent = Vec3(f * (du2 * e1.x - du1 * e2.x), f * (du2 * e1.y - du1 * e2.y), f * (du2 * e1.z - du1 * e2.z)).normalize();

        Vec3 bumpMap = Vec3(tex2D<float4>(mat.bump, hit.u, hit.v)) * 2.0f - 1.0f;

        return (tangent * bumpMap.x + bitangent * bumpMap.y + hit.phong * bumpMap.z).normalize();
    } else return hit.phong;
}