/*
    POLY cuda ~ main
    Diego Párraga Nicolás ~ diegojose.parragan@um.es
*/

#include "PolyRenderer.h"

// Print usage
void printUsage(){
    printf("\e[1;93m Usage: \e[91mp\e[92mo\e[94ml\e[95my \e[92mcuda \e[95m-i=SCENE.POLY \e[96m [-o=OUTPUT_PATH]\e[0m\n");
}

// Application entrypoint
int main(int argc, char** argv){
    
    // Parse arguments
    if(argc<2 || argc>3){ printUsage(); exit(EXIT_FAILURE); }
    string input, output, outName;
    for(uint8_t i=1; i<argc; i++){
        string arg = string(argv[i]);
        if(arg.length()>3 && arg[0]=='-' && arg[2]=='='){
            string value = arg.substr(3, arg.length() - 3);
            switch(arg[1]){
                case 'o' :  // Output path
                    output = value + "/";
                    break;  
                case 'i' :  // Input scene script
                    input = value;
                    filesystem::path scene(value);
                    if(!filesystem::exists(scene)){ printf("\e[1;91m err loading scene '%s': file does not exist!\e[0m\n", value.c_str()); return EXIT_FAILURE; }
                    else if(scene.extension().string() != ".poly"){ printf("\e[1;91m err loading scene '%s': not a .poly file!\e[0m\n", value.c_str()); return EXIT_FAILURE; }
                    outName = scene.stem(); outName += ".png";
                    break;
            }
        } else { printUsage(); return EXIT_FAILURE; }
    }
    output += outName;

    // Print poly intro
    PolyRenderer::printIntro();

    // Create rendering object, load scene and start the rendering
    PolyRenderer renderer = PolyRenderer();
    if(renderer.loadScene(input.c_str()) && renderer.render())
        renderer.save(output.c_str());
    
    return EXIT_SUCCESS;
}