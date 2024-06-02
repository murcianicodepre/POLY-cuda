#include "PolyRenderer.h"

std::vector<std::string> msgvector;

template<class T> PolyArray<T>::PolyArray(T data, uint32_t size) : data(data), size(size) {}

BVHNode::BVHNode() : aabbMin(), aabbMax(), leftOrFirst(0u), n(0u) {}
__device__ bool BVHNode::intersectAABB(Ray& ray){
    // Slab test for volume-ray intersection
    float tx1 = (aabbMin.x - ray.ori.x) / ray.dir.x, tx2 = (aabbMax.x - ray.ori.x) / ray.dir.x;
    float tmin = fmin(tx1, tx2), tmax = fmax(tx1, tx2);
    float ty1 = (aabbMin.y - ray.ori.y) / ray.dir.y, ty2 = (aabbMax.y - ray.ori.y) / ray.dir.y;
    tmin = fmax(fmin(ty1, ty2), tmin), tmax = fmin(fmax(ty1, ty2), tmax);
    float tz1 = (aabbMin.z - ray.ori.z) / ray.dir.z, tz2 = (aabbMax.z - ray.ori.z) / ray.dir.z;
    tmin = fmax(fmin(tz1, tz2), tmin), tmax = fmin(fmax(tz1, tz2), tmax);
    return tmax >= tmin && tmax > 0.0f;
}

void PolyRenderer::buildBVH(){
    _bvh = (BVHNode*) malloc(sizeof(BVHNode) * 2 * _tris.size() - 1);
    _triIdx = (uint32_t*) malloc(sizeof(uint32_t) * _tris.size());
    for(uint32_t i=0; i<_tris.size(); i++) 
        _triIdx[i] = i;

    // Assign all tris to root node
    BVHNode& root = _bvh[0];
    root.n = _tris.size(); root.leftOrFirst = 0u;

    updateNodeBounds(0);    // Update bounds of root node
    subdivide(0);           // Start subdivision

    // Resize with only the leaf nodes
    std::vector<BVHNode> leaf;
    for(uint32_t i=0; i<_nextNode; i++)
        if(_bvh[i].n > 0){ leaf.push_back(_bvh[i]); }

    // Print size of _bvh structure
    PolyRenderer::polyMsg("\e[1;93m compiling bvh: \e[95m" + to_string(_nextNode) + " \e[93mnodes (\e[95m" + to_string(_nextNode*sizeof(BVHNode)) + " bytes\e[93m) \e[92mOK\e[0m\n");
}

void PolyRenderer::updateNodeBounds(uint32_t nodeId){
    BVHNode& node = _bvh[nodeId];
    node.aabbMin = Vec3(1e30f), node.aabbMax = Vec3(1e-30f);
    for(uint32_t i=0; i<node.n; i++){
        Tri& tri = _tris[_triIdx[node.leftOrFirst+i]];
        node.aabbMin = Vec3::min(node.aabbMin, tri.a.xyz);
        node.aabbMin = Vec3::min(node.aabbMin, tri.b.xyz);
        node.aabbMin = Vec3::min(node.aabbMin, tri.c.xyz);
        node.aabbMax = Vec3::max(node.aabbMax, tri.a.xyz);
        node.aabbMax = Vec3::max(node.aabbMax, tri.b.xyz);
        node.aabbMax = Vec3::max(node.aabbMax, tri.c.xyz);
    }
}

void PolyRenderer::subdivide(uint32_t nodeId){
    BVHNode& node = _bvh[nodeId];
    if(node.n<=2) return;  // Terminate recursive subdivision

    // Get split axis
    Vec3 ext = node.aabbMax - node.aabbMin;
    uint8_t axis = 0u;
    if(ext.y > ext.x) axis = 1;
    if(ext.z > ext[axis]) axis = 2;

    float split = node.aabbMin[axis] + ext[axis] * 0.5f;

    // Split the geometry in two parts
    uint32_t i = node.leftOrFirst, j = i + node.n - 1;
    while(i<=j){
        Tri& tri = _tris[_triIdx[i]];
        if(tri.centroid()[axis]<split) i++;
        else std::swap(_triIdx[i], _triIdx[j--]);
    }

    // Terminate if one of the sides is empty
    uint32_t leftCount = i - node.leftOrFirst;
    if (leftCount==0 || leftCount==node.n) return;

    // Create child nodes
    uint32_t leftIdx = _nextNode++, rightIdx = _nextNode++;
    _bvh[leftIdx].leftOrFirst = node.leftOrFirst; _bvh[leftIdx].n = leftCount;
    _bvh[rightIdx].leftOrFirst = i; _bvh[rightIdx].n = node.n - leftCount;
    node.leftOrFirst = leftIdx; node.n = 0;
    updateNodeBounds(leftIdx); updateNodeBounds(rightIdx);

    // Continue recursion
    subdivide(leftIdx); subdivide(rightIdx);
}

PolyRenderer::PolyRenderer() : _tris(), _mats(), _lights(), _cam(nullptr), _cuArrays() {
    _frame = (RGBA*) malloc(sizeof(RGBA) * WIDTH * HEIGHT);
    memset((void*) _frame, 0, sizeof(RGBA) * WIDTH * HEIGHT);
}

PolyRenderer::~PolyRenderer(){
    free(_frame);
    if(_bvh) 
        free(_bvh);
    if(_triIdx) 
        free(_triIdx);
}

void cudaerr(cudaError_t err){
    if(err != cudaSuccess){ printf("\e[1;91mcudaerr! %s\e[0m\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }
}

// Other renderer functions
__host__ RGBA* PolyRenderer::loadPNG(const char* path){
    FILE* input = fopen(path, "rb");
        if(!input){ PolyRenderer::polyMsg("\e[1;91m  err loading '" + string(path) + "': file could not be opened!\n\e[0m"); exit(EXIT_FAILURE); }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
        if(!png){ 
            fclose(input);
            PolyRenderer::polyMsg("\e[1;91m  err texture '" + string(path) + "' read struct could not be loaded!\n\e[0m"); exit(EXIT_FAILURE);
        }

    png_infop info = png_create_info_struct(png);
        if(!info){
            fclose(input);
            png_destroy_read_struct(&png, NULL, NULL);
            PolyRenderer::polyMsg("\e[1;91m  err texture '" + string(path) + "' info struct could not be loaded!\e[0m\n"); exit(EXIT_FAILURE);
        }

    png_init_io(png, input);
    png_read_info(png, info);

    int w = png_get_image_width(png, info), h = png_get_image_height(png, info);
    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth = png_get_bit_depth(png, info);

    if(bit_depth==16) png_set_strip_16(png);
    if(color_type==PNG_COLOR_TYPE_RGB) png_set_palette_to_rgb(png);
    if(color_type==PNG_COLOR_TYPE_GRAY && bit_depth < 8) png_set_expand_gray_1_2_4_to_8(png);
    if(png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
    if (color_type == PNG_COLOR_TYPE_RGB || color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_PALETTE) 
        png_set_filler(png, 0xff, PNG_FILLER_AFTER);
    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA) png_set_gray_to_rgb(png);

    png_read_update_info(png, info);

    RGBA* texture = (RGBA*) malloc(sizeof(RGBA) * TEXTURE_SIZE * TEXTURE_SIZE);
    vector<png_bytep> row_p(h);
    for(int y=0; y<h; y++)
        row_p[y] = new png_byte[png_get_rowbytes(png, info)];

    png_read_image(png, row_p.data());

    for(int y=0; y<h; y++)
        for(int x=0; x<w; x++){
            png_bytep pixel = &(row_p[y][x * 4]); 
            texture[y * w + x].r = pixel[0];
            texture[y * w + x].g = pixel[1];
            texture[y * w + x].b = pixel[2];
            texture[y * w + x].a = pixel[3];
        }

    for(int y=0; y<h; y++)
        delete[] row_p[y];

    png_destroy_read_struct(&png, &info, NULL);

    if(fclose(input)==-1){ PolyRenderer::polyMsg("\e[1;91m  err closing '" + string(path) + "'\e[0m\n"); exit(EXIT_FAILURE); }
    PolyRenderer::polyMsg("\e[1;93m  loaded texture '" + string(path) + "'\e[0m\n");
    
    return texture;
}
__host__ void PolyRenderer::savePNG(const char* path, RGBA* texture){
    FILE* output = fopen(path, "wb");
        if(!output){ PolyRenderer::polyMsg("\e[1;91m  err saving to '" + string(path) + "': file could not be opened!\n\e[0m"); exit(EXIT_FAILURE); }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
        if(!png){ 
            fclose(output);
            PolyRenderer::polyMsg("\e[1;91m  err saving render\n\e[0m");
            exit(EXIT_FAILURE);
        }

    png_infop info = png_create_info_struct(png);
        if(!info){ 
            fclose(output);
            png_destroy_write_struct(&png, NULL);
            PolyRenderer::polyMsg("\e[1;91m  err saving render\e[0m\n");
            exit(EXIT_FAILURE);
        }
    
    png_init_io(png, output);
    png_set_IHDR(png, info, WIDTH, HEIGHT, 8, PNG_COLOR_TYPE_RGBA, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    vector<png_byte> row(WIDTH * 4);
    for(int y=0; y<HEIGHT; y++){
        for(int x=0; x<WIDTH; x++){
            row[x * 4] = static_cast<png_byte>(texture[y * WIDTH + x].r);               // r
            row[x * 4 + 1] = static_cast<png_byte>(texture[y * WIDTH + x].g);          // g
            row[x * 4 + 2] = static_cast<png_byte>(texture[y * WIDTH + x].b);         // b    
            row[x * 4 + 3] = static_cast<png_byte>(texture[y * WIDTH + x].a);        // a
        }
        png_write_row(png, row.data());
    }
    png_write_end(png, NULL);
    png_destroy_write_struct(&png, &info);

    if(fclose(output)==-1){ PolyRenderer::polyMsg("\e[1;91m  err closing '" + string(path) + "'\e[0m\n"); exit(EXIT_FAILURE); }
}
__host__ const char* PolyRenderer::getCpu(){
    char vendor[13];
    uint eax = 0, ebx, ecx, edx;

    __asm__ __volatile__(
        "cpuid"
        : "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "a"(eax)
    );

    memcpy(vendor, &ebx, 4);
    memcpy(vendor + 4, &edx, 4);
    memcpy(vendor + 8, &ecx, 4);
    vendor[12] = '\0';
    return vendor[0] == 'G' ? "\e[1;94mx86-64" : "\e[1;91mamd64";
}
__host__ const char* PolyRenderer::getGpu(cudaDeviceProp& gpu){
    switch(gpu.major){
        case 2 : { return "fermi"; break; }
        case 3 : { return "kepler"; break; }
        case 5 : { return "maxwell"; break; }
        case 6 : { return "pascal"; break; }
        case 7 : {
            switch(gpu.minor){
                case 5 : { return "turing"; break; }
                default : { return "volta"; break; }
            }
            break;
        }
        case 8 : {
            switch(gpu.minor){
                case 9 : { return "ada"; break; }
                default : { return "ampere"; break; }
            }
            break;
        }
        case 9 : { return "hopper"; break; }
        default: { return "3d device"; break; }
    }
}
__host__ void PolyRenderer::printIntro(){
    if(system("clear")<0){ printf("\e[1;91m err clear failed\e[0m\n"); exit(EXIT_FAILURE); }
    printf(" \e[1;91m▄▄▄   \e[92m▄▄   \e[94m▄  \e[95m▄   ▄ \n");
    printf(" \e[91m█  █ \e[92m█  █  \e[94m█   \e[95m█ █  \e[92m ▄▄ ▄  ▄ ▄▄   ▄\n");
    printf(" \e[91m█▀▀  \e[92m█  █  \e[94m█    \e[95m█  \e[92m █   █  █ █ █ █▄█\n");
    printf(" \e[91m█    \e[92m▀▄▄▀  \e[94m█▄▄  \e[95m█   \e[92m▀▄▄ ▀▄▄▀ █▄▀ █ █\n");
    printf("\e[91m     - diegojose.parragan@um.es -\n\e[0m\n");
}
__host__ void PolyRenderer::polyMsg(std::string msg){
    printf("%s", msg.c_str());
    msgvector.push_back(msg);
}
__host__ Vec3 PolyRenderer::parseVec3(YAML::Node node){
    if(node.IsSequence() && node.size()==3){
        return Vec3(node[0].as<float>(), node[1].as<float>(), node[2].as<float>());
    } else throw runtime_error("\e[1;91m  err parsing Vec3\e[0m\n");
}
__host__ RGBA PolyRenderer::parseColor(YAML::Node node){
    if(node.IsSequence() && node.size()>2){
        return RGBA(node[0].as<uint8_t>(), node[1].as<uint8_t>(), node[2].as<uint8_t>(), node[3] ? node[3].as<uint8_t>() : 255u);
    } else throw runtime_error("\e[1;91m  err parsing RGBA\e[0m\n");
}
__host__ uint16_t PolyRenderer::parseFlags(YAML::Node node){
    uint16_t flags = 0x0000u;
    for(auto f : node){
        string flag = f.as<string>();
        if(flag=="DISABLE_RENDERING")
            flags |= DISABLE_RENDERING;
        else if(flag=="DISABLE_SHADING")
            flags |= DISABLE_SHADING;
        else if(flag=="DISABLE_TEXTURES")
            flags |= DISABLE_TEXTURES;
        else if(flag=="DISABLE_BUMP")
            flags |= DISABLE_BUMP;
        else if(flag=="DISABLE_TRANSPARENCY")
            flags |= DISABLE_TRANSPARENCY;
        else if(flag=="DISABLE_SHADOWS")
            flags |= DISABLE_SHADOWS;
        else if(flag=="DISABLE_REFLECTIONS")
            flags |= DISABLE_REFLECTIONS;
        else if(flag=="DISABLE_REFRACTIONS")
            flags |= DISABLE_REFRACTIONS;
        else if(flag=="FLAT_SHADING"){
            PolyRenderer::polyMsg("\e[1;96m  FLAT_SHADING\e[0m\n");
            flags |= (FLAT_SHADING<<8);
        }
        else PolyRenderer::polyMsg("\e[1;96m  err unknown flag '" + string(flag) + "'\e[0m\n");
        
    }
    return flags;
}

// Polyscript v2 scene loader
bool PolyRenderer::loadScene(const char* scene){
    printf("\e[1;93m compiling '\e[95m%s\e[93m'\e[0m\n", scene);

    _tris.clear(); _mats.clear(); _lights.clear();
    vector<Poly> polyvec;

    string script_path = filesystem::path(scene).parent_path(); if(!script_path.empty()) script_path += "/";
    try{
        YAML::Node file = YAML::LoadFile(scene);
        vector<string> mats_names, objs_names;

        // Parse global flags
        if(file["global"])
            _global = (parseFlags(file["global"]) & 0xffffu);
        
        // Parse camera
        if(file["camera"]){
            YAML::Node camera = file["camera"];
            Vec3 pos = parseVec3(camera["position"]);
            float fov = camera["fov"].as<float>();
            Vec3 lookAt = camera["lookAt"] ? parseVec3(camera["lookAt"]) : Vec3(pos.x, pos.y, 1.0f);
            _cam = new Camera(pos, lookAt, fov);
        } else { PolyRenderer::polyMsg("\e[1;91m  err parsing scene: camera missing\e[0m\n"); return false; }

        // Parse materials and store material name for object declaration
        if(file["materials"]){
            YAML::Node mats = file["materials"];

            // Insert dummy material for void
            _mats.push_back(Material(0.0f, 0.0f, 0.0f, 1.0f));
            mats_names.push_back("VOID_MAT");

            for(const auto& m : mats){
                if(m["name"] && m["diffuse"] && m["specular"]){
                    string mat_name = m["name"].as<string>();
                    float diff = m["diffuse"].as<float>(), spec = m["specular"].as<float>();
                    float reflect = m["reflective"] ? m["reflective"].as<float>() : 0.0f, refract = m["refractive"] ? m["refractive"].as<float>() : 1.0f;
                    reflect = (reflect>1.0f) ? 1.0f : (reflect<0.0f ? 0.0f : reflect);
                    Material mat = Material(diff, spec, reflect, refract);
                    if(m["texture"]) _cuArrays.push_back(mat.loadTexture((script_path + m["texture"].as<string>()).c_str())); 
                    mat.color = (m["color"]) ? parseColor(m["color"]) : RGBA(Vec3(0.8f, 0.8f, 0.8f), 1.0f);
                    if(m["bump"]) _cuArrays.push_back(mat.loadBump((script_path + m["bump"].as<string>()).c_str()));

                    // Push material and name at same index
                    _mats.push_back(mat);
                    mats_names.push_back(mat_name);
                } else { PolyRenderer::polyMsg("\e[1;91m  err parsing material: attributes missing!\e[0m\n"); return false; }
            }
        } else { PolyRenderer::polyMsg("\e[1;91m  err parsing scene: materials missing\e[0m\n"); return false; }

        // Parse objects and store object name
        if(file["objects"]){
            YAML::Node objs = file["objects"];
            for(const auto& obj : objs){
                if(obj["name"] && obj["file"] && obj["material"]){
                    string obj_name = obj["name"].as<string>(), obj_file = script_path + obj["file"].as<string>(), obj_mat = obj["material"].as<string>();

                    // Get material index from name
                    auto it = find_if(mats_names.begin(), mats_names.end(),
                        [&obj_mat](const string& s){ return s==obj_mat; }
                    ); 
                    if(it==mats_names.end()){ PolyRenderer::polyMsg("\e[1;91m  err parsing object: undeclared material!\e[0m\n"); return false; }
                    
                    // Parse rendering flags
                    uint8_t obj_flags = 0u;
                    if(obj["flags"])
                        obj_flags = static_cast<uint8_t>(parseFlags(obj["flags"]));

                    Poly poly = Poly(obj_file.c_str(), static_cast<uint8_t>(distance(mats_names.begin(), it)), (obj_flags|(_global & 0xffu)));
                    if(obj["transforms"])
                        for(const auto& t : obj["transforms"]){
                            string op = t.first.as<string>();
                            if(op=="scale"){
                                    if(t.second.IsSequence())
                                        poly.scale(parseVec3(t.second));
                                    else poly.scale(t.second.as<float>());
                                } 
                                else if(op=="rotate") poly.rotate(parseVec3(t.second));
                                else if(op=="rotateX") poly.rotateX(t.second.as<float>());
                                else if(op=="rotateY") poly.rotateY(t.second.as<float>());
                                else if(op=="rotateZ") poly.rotateZ(t.second.as<float>());
                                else if(op=="move")
                                    poly.move(parseVec3(t.second));
                                else PolyRenderer::polyMsg("\e[1;94m  unknown transform '" + op + "'\e[0m\n");
                        }

                    // Insert both object and object name in vector
                    polyvec.push_back(poly);
                    objs_names.push_back(obj_name);

                } else { PolyRenderer::polyMsg("\e[1;91m  err parsing object: attributes missing!\e[0m\n"); return false; }
            }
        } else { PolyRenderer::polyMsg("\e[1;91m  err parsing scene: objects missing!\e[0m\n"); return false; }

        for(const auto& p : polyvec)
            _tris.insert(_tris.end(), p.tris.begin(), p.tris.end()); 

        if(file["lights"]){
            YAML::Node ls = file["lights"];
            for(const auto& l : ls){
                if(l["type"] && l["color"] && l["intensity"]){
                    string ltype = l["type"].as<string>();
                    if(ltype=="ambient"){
                        _lights.push_back(Light(parseColor(l["color"]), l["intensity"].as<float>(), Vec3(), LightType::Ambient));
                    } else if(ltype=="point"){
                        if(!l["position"]){ PolyRenderer::polyMsg("\e[1;91m  err parsing point light: position missing!\e[0m\n"); return false; }
                        _lights.push_back(Light(parseColor(l["color"]), l["intensity"].as<float>(), parseVec3(l["position"]), LightType::Point));
                    } else if(ltype=="directional"){
                        if(!l["direction"]){ PolyRenderer::polyMsg("\e[1;91m  err parsing directional light: direction missing!\e[0m\n"); return false; }
                        _lights.push_back(Light(parseColor(l["color"]), l["intensity"].as<float>(), parseVec3(l["direction"]), LightType::Directional));
                    } else { PolyRenderer::polyMsg("\e[1;91m  err parsing light: unknown light type!\e[0m\n"); return false; }
                } else { PolyRenderer::polyMsg("\e[1;91m  err parsing light: attributes missing!\e[0m\n"); return false; }
            }
        } else { PolyRenderer::polyMsg("\e[1;91m  err parsing scene: lights missing!\e[0m\n"); return false; }

    } catch (const YAML::ParserException& pe){ printf("\e[1;91m exception while parsing '\e[95m%s\e[93m': %s\e[0m\n", scene, pe.msg.c_str()); return false; }

    // Nice printing
    if(system("clear")<0){ printf("\e[1;91m err clear failed\e[0m\n"); exit(EXIT_FAILURE); }
    printIntro();
    printf("\e[1;93m compiling '\e[95m%s\e[93m' \e[92mOK\e[0m\n", scene);
    for(auto s : msgvector) { printf("%s", s.c_str());}

    return true;
}

bool PolyRenderer::render(){
    constexpr uint32_t FRAME_SIZE = sizeof(RGBA) * WIDTH * HEIGHT;

    // Abort rendering if global DISABLE_RENDERING is set
    if(static_cast<uint8_t>(_global) & DISABLE_RENDERING){
        printf("\e[1;91m fatal: rendering disabled!\e[0m\n");
        return false;
    }

    // Compile BVH after scene is loaded
    buildBVH();

    // Initialize device
    cudaError_t shaderStatus = cudaSuccess;
    cudaerr(cudaSetDevice(0));
    cudaerr(cudaGetDeviceProperties(&_gpu, 0));

    // Allocate frame in GPU global memory
    RGBA* frame_d;
    cudaerr(cudaMalloc((void**) &frame_d, FRAME_SIZE));
    cudaerr(cudaMemcpy((void*) frame_d, (void*) _frame, FRAME_SIZE, cudaMemcpyHostToDevice));

    // Set shader settings
    dim3 grid(WIDTH/TILE_SIZE, HEIGHT/TILE_SIZE), block(TILE_SIZE, TILE_SIZE);
    cudaEvent_t tIni, tEnd;
    cudaerr(cudaEventCreate(&tIni)); cudaerr(cudaEventCreate(&tEnd));
    float tGpu;

    // Copy scene data into GPU memory
    Scene scene, * scene_d; 

    cudaerr(cudaMalloc((void**) &scene.cam, sizeof(Camera)));
    cudaerr(cudaMemcpy((void*) scene.cam, (void*) _cam, sizeof(Camera), cudaMemcpyHostToDevice));

    Tri* tris_d;
    cudaerr(cudaMalloc((void**) &tris_d, sizeof(Tri) * _tris.size()));
    cudaerr(cudaMemcpy((void*) tris_d, (void*) _tris.data(), sizeof(Tri) * _tris.size(), cudaMemcpyHostToDevice));
    scene.tris = PolyArray<Tri*>(tris_d, _tris.size());

    Material* mats_d;
    cudaerr(cudaMalloc((void**) &mats_d, sizeof(Material) * _mats.size()));
    cudaerr(cudaMemcpy((void*) mats_d, (void*) _mats.data(), sizeof(Material) * _mats.size(), cudaMemcpyHostToDevice));
    scene.mats = PolyArray<Material*>(mats_d, _mats.size());

    Light* lights_d;
    cudaerr(cudaMalloc((void**) &lights_d, sizeof(Light) * _lights.size()));
    cudaerr(cudaMemcpy((void*) lights_d, (void*) _lights.data(), sizeof(Light) * _lights.size(), cudaMemcpyHostToDevice));
    scene.lights = PolyArray<Light*>(lights_d, _lights.size());

    BVHNode* bvh_d;
    cudaerr(cudaMalloc((void**) &bvh_d, sizeof(BVHNode) * _nextNode));
    cudaerr(cudaMemcpy((void*) bvh_d, (void*) _bvh, sizeof(BVHNode) * _nextNode, cudaMemcpyHostToDevice));
    scene.bvh = bvh_d;

    uint32_t* triIdx_d;
    cudaerr(cudaMalloc((void**) &triIdx_d, sizeof(uint32_t) * _tris.size()));
    cudaerr(cudaMemcpy((void*) triIdx_d, (void*) _triIdx, sizeof(uint32_t) * _tris.size(), cudaMemcpyHostToDevice));
    scene.triIdx = triIdx_d;

    cudaerr(cudaMalloc((void**) &scene_d, sizeof(Scene)));
    cudaerr(cudaMemcpy((void*) scene_d, (void*) &scene, sizeof(Scene), cudaMemcpyHostToDevice));

    // Start timer and launch compute_pixel shader
    printf("\e[1;93m rendering in \e[92m%s ", getGpu(_gpu));
    fflush(stdout);
    cudaerr(cudaEventRecord(tIni, 0));

    // Launch rendering kernel
    compute_pixel<<<grid, block>>> (frame_d, scene_d);
    shaderStatus = cudaGetLastError();

    // If shaderStatus OK, copy rendering to local memory
    if(shaderStatus==cudaSuccess){
        cudaerr(cudaEventRecord(tEnd, 0));
        cudaerr(cudaEventSynchronize(tEnd));
        cudaerr(cudaEventElapsedTime(&tGpu, tIni, tEnd));
        cudaerr(cudaMemcpy((void*) _frame, (void*) frame_d, FRAME_SIZE, cudaMemcpyDeviceToHost));
        printf("\e[1;95m%.3fs \e[92mOK\e[0m\n", tGpu * 1e-3f);
    } else {
        printf("\e[1;91mERR! %s\e[0m\n", cudaGetErrorString(shaderStatus));
    }

    // Free resources and reset GPU
    for(Material mat : _mats){
        if(mat.texture) cudaerr(cudaDestroyTextureObject(mat.texture));
        if(mat.bump) cudaerr(cudaDestroyTextureObject(mat.bump));
    }
    for(cudaArray_t array : _cuArrays)
        cudaerr(cudaFreeArray(array));
    cudaFree(frame_d); cudaFree(scene_d); 
    cudaFree(scene.cam); cudaFree(tris_d); cudaFree(mats_d); cudaFree(lights_d); cudaFree(bvh_d); cudaFree(triIdx_d);
    cudaerr(cudaDeviceReset());

    return (shaderStatus!=cudaSuccess) ? false : true;
}

void PolyRenderer::save(const char* path){
    savePNG(path, _frame);
    printf("\e[1;93m saved in '\e[95m%s\e[93m'\e[0m\n", path);
}
