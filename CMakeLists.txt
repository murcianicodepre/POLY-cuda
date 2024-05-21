# POLY cuda ~ CMAKE file
# Diego Párraga Nicolás ~ diegojose.parragan@um.es

cmake_minimum_required(VERSION 3.16)
project(poly-cuda LANGUAGES CXX CUDA)

# Search packages
find_package(PNG REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(CUDA REQUIRED)

# Some configuration
set(CMAKE_INSTALL_PREFIX "/usr/local/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "..")
set(CMAKE_CXX_STANDARD 17)
set(CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(SRC_DIR "src")
set(INC_DIR "include")

# Build options
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")

# Cuda source files
set(CU_SOURCES
    ${SRC_DIR}/Vec.cu
    ${SRC_DIR}/RGBA.cu
    ${SRC_DIR}/Material.cu
    ${SRC_DIR}/PolyRenderer.cu
    ${SRC_DIR}/PolyShaders.cu
)
include_directories(${INC_DIR} ${CUDA_INCLUDE_DIRS})

# Compile CUDA objects
add_library(poly-cuda_objs OBJECT ${CU_SOURCES})
set_target_properties(poly-cuda_objs PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Compile main.cpp and link with poly-cuda_objs
add_executable(poly-cuda ${SRC_DIR}/main.cpp $<TARGET_OBJECTS:poly-cuda_objs>)
set_target_properties(poly-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link libraries
target_link_libraries(poly-cuda 
    PNG::PNG
    ${YAML_CPP_LIBRARIES}
    ${CUDA_LIBRARIES}
)
target_include_directories(poly-cuda PRIVATE ${INC_DIR} ${CUDA_INCLUDE_DIRS})

# Install target
install(TARGETS poly-cuda RUNTIME DESTINATION "/usr/local/bin")