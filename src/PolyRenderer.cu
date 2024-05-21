#include "PolyRenderer.h"

std::vector<std::string> msgvector;

PolyRenderer::PolyRenderer(){
    _frame = (RGBA*) malloc(sizeof(RGBA) * WIDTH * HEIGHT);
    memset((void*) _frame, 0, sizeof(RGBA) * WIDTH * HEIGHT);
}

PolyRenderer::~PolyRenderer(){
    free(_frame);
}

void PolyRenderer::cudaerr(cudaError_t err){
    if(err != cudaSuccess){ printf("\e[1;91mcudaerr! %s\e[0m\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }
}

// TODO bvh struct functions

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
    
    // Copy texture into GPU global memory
    RGBA* texture_d;
    PolyRenderer::cudaerr(cudaMalloc((void**) &texture_d, sizeof(RGBA) * TEXTURE_SIZE * TEXTURE_SIZE));
    PolyRenderer::cudaerr(cudaMemcpy((void**) &texture_d, (void**) texture, sizeof(RGBA) * TEXTURE_SIZE * TEXTURE_SIZE, cudaMemcpyHostToDevice));

    // Free host memory texture
    free(texture);

    return texture_d;
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
        else if(flag=="DISABLE_FAST_INTERSECTION_SHADER"){
            PolyRenderer::polyMsg("\e[1;96m  DISABLE_FAST_INTERSECTION_SHADER\e[0m\n");
            flags |= (DISABLE_FAST_INTERSECTION_SHADER<<8);
        }
        else if(flag=="FLAT_SHADING"){
            PolyRenderer::polyMsg("\e[1;96m  FLAT_SHADING\e[0m\n");
            flags |= (FLAT_SHADING<<8);
        }
        else PolyRenderer::polyMsg("\e[1;96m  err unknown flag '" + string(flag) + "'\e[0m\n");
        
    }
    return flags;
}

// TODO polyscript scene loader
bool PolyRenderer::loadScene(const char* scene){
    return true;
}

bool PolyRenderer::render(){
    constexpr uint32_t RENDERING_TOTAL_SIZE = sizeof(RGBA) * WIDTH * HEIGHT;

    // Abort rendering if global DISABLE_RENDERING is set
    if(static_cast<uint8_t>(_global) & DISABLE_RENDERING){
        printf("\e[1;91m fatal: rendering disabled!\e[0m\n");
        return false;
    }

    // Initialize device
    cudaError_t shaderStatus = cudaSuccess;
    cudaerr(cudaSetDevice(0));
    cudaerr(cudaGetDeviceProperties(&_gpu, 0));

    // Allocate frame in GPU global memory
    RGBA* frame_d;
    cudaerr(cudaMalloc((void**) &frame_d, RENDERING_TOTAL_SIZE));
    cudaerr(cudaMemcpy((void*) frame_d, (void*) _frame, RENDERING_TOTAL_SIZE, cudaMemcpyHostToDevice));

    // Set shader settings
    dim3 grid(WIDTH/TILE_SIZE, HEIGHT/TILE_SIZE), block(TILE_SIZE, TILE_SIZE);
    cudaEvent_t tIni, tEnd;
    cudaerr(cudaEventCreate(&tIni)); cudaerr(cudaEventCreate(&tEnd));
    float tGpu;

    // Start timer and launch compute_pixel shader
    printf("\e[1;93m rendering in \e[92m%s ", getGpu(_gpu));
    cudaerr(cudaEventRecord(tIni, 0));

    compute_pixel<<<grid, block>>> (frame_d, _global);
    shaderStatus = cudaGetLastError();

    // If shaderStatus OK, copy rendering to local memory
    if(shaderStatus==cudaSuccess){
        cudaerr(cudaEventRecord(tEnd, 0));
        cudaerr(cudaEventSynchronize(tEnd));
        cudaerr(cudaEventElapsedTime(&tGpu, tIni, tEnd));
        cudaerr(cudaMemcpy((void*) _frame, (void*) frame_d, RENDERING_TOTAL_SIZE, cudaMemcpyDeviceToHost));
        printf("\e[1;95m%.3fs \e[92mOK\e[0m\n", tGpu * 1e-3f);
    } else {
        printf("\e[1;91mERR! %s\e[0m\n", cudaGetErrorString(shaderStatus));
    }

    // Free resources and reset GPU
    cudaFree(frame_d);
    cudaerr(cudaDeviceReset());

    return (shaderStatus!=cudaSuccess) ? false : true;
}

void PolyRenderer::save(const char* path){
    savePNG(path, _frame);
    printf("\e[1;93m saved in '\e[95m%s\e[93m'\e[0m\n", path);
}
