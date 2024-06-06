#include "Poly.h"

/*
    Poly ~ Vertex, Hit, Tri and Poly class
    Diego Párraga Nicolás ~ diegojose.parragan@um.es
*/

Vertex::Vertex(Vec3 xyz, Vec3 normal, float u, float v) : xyz(xyz), normal(normal), u(u), v(v) {}
void Vertex::move(Vec3 m){ xyz = xyz + m; }
void Vertex::scale(Vec3 s){ xyz = Vec3(s.x * xyz.x, s.y * xyz.y, s.z * xyz.z); }
void Vertex::scale(float s){ xyz = xyz * s; normal = (normal*s).normalize();  }
void Vertex::rotate(Vec3 r){ xyz.rotate(r); normal.rotate(r); }
void Vertex::rotateX(float r){ xyz.rotateX(r); normal.rotateX(r); }
void Vertex::rotateY(float r){ xyz.rotateY(r); normal.rotateY(r); }
void Vertex::rotateZ(float r){ xyz.rotateZ(r); normal.rotateZ(r); }

Tri::Tri(Vertex a, Vertex b, Vertex c, uint8_t matId, uint8_t flags) : a(a), b(b), c(c), matId(matId), flags(flags) {}
__device__ bool Tri::intersect(Ray ray, Hit& hit){
    Vec3 edge1 = b.xyz - a.xyz, edge2 = c.xyz - a.xyz, h = Vec3::cross(ray.dir, edge2);

    float A = Vec3::dot(edge1, h);
    if(A>-EPSILON && A<EPSILON) return false;   // Ray parallel

    Vec3 s = ray.ori - a.xyz;
    float inv = 1.0f/A, U = inv * Vec3::dot(s, h);
    if(U<0 || U>1.0f) return false;

    Vec3 q = Vec3::cross(s, edge1);
    float V = inv * Vec3::dot(ray.dir, q);
    if(V<0 || (U+V)>1.0f) return false;

    float t = inv * Vec3::dot(edge2, q);
    Vec3 normal = Vec3::cross(edge1, edge2).normalize();
    if(t>EPSILON && Vec3::dot(ray.dir, normal)<0.0f){  // Hit!

        hit.t = t;
        hit.normal = normal;

        hit.ray = ray;
        hit.u = (1.0f-U-V) * a.u + U * b.u + V * c.u;
        hit.v = (1.0f-U-V) * a.v + U * b.v + V * c.v;

        // Compute interpolated normal
        hit.phong = !(a.normal == 0.0f && b.normal == 0.0f && c.normal == 0.0f) ? (a.normal*(1.0f-U-V) + b.normal*U + c.normal*V).normalize() : hit.normal;

        return true;
    } else return false;
}
void Tri::move(Vec3 m){ a.move(m); b.move(m); c.move(m); }
void Tri::scale(Vec3 s){ a.scale(s); b.scale(s), c.scale(s); }
void Tri::scale(float s){ a.scale(s); b.scale(s), c.scale(s); }
void Tri::rotate(Vec3 r){ a.rotate(r); b.rotate(r); c.rotate(r); }
void Tri::rotateX(float r){ a.rotateX(r); b.rotateX(r); c.rotateX(r); }
void Tri::rotateY(float r){ a.rotateY(r); b.rotateY(r); c.rotateY(r); }
void Tri::rotateZ(float r){ a.rotateZ(r); b.rotateZ(r); c.rotateZ(r); }
float Tri::min(uint8_t axis){ float m = a.xyz[axis]<b.xyz[axis] ? a.xyz[axis] : b.xyz[axis]; return m<c.xyz[axis] ? m : c.xyz[axis]; }
float Tri::max(uint8_t axis){ float m = a.xyz[axis]>=b.xyz[axis] ? a.xyz[axis] : b.xyz[axis]; return m>=c.xyz[axis] ? m : c.xyz[axis]; }
Vec3 Tri::centroid(){  return (a.xyz + b.xyz + c.xyz) * 0.33333f; }

Poly::Poly(const char* path, uint8_t matId, uint8_t flags){
    if(flags & DISABLE_RENDERING) return;
    char buff[128u];
    float x, y, z, nx, ny, nz, u, v;
    Vertex a, b, c;
    uint nvertex, nfaces;

    ifstream input; input.open(path, fstream::in);
    if(!input.is_open()){ PolyRenderer::polyMsg("\e[1;91m  err opening '" + string(path) + "'\e[0m\n"); exit(EXIT_FAILURE); }

    // From the header we expect to get the number of faces and vertices, and also check if the vertex data contains the UVs and the normals
    string element;
    bool foundXyz = false, foundNormals = false, foundUVs = false;
    while(element!="end_header" && input.getline(buff, 128u)){
        stringstream iss(buff); iss >> element;
        if(element=="property"){
            iss >> element; iss>>element;
            if(element=="x" || element=="y" || element=="z") foundXyz = true;
            else if(element=="nx" || element=="ny" || element=="nz") foundNormals = true;
            else if (element=="s" || element=="t") foundUVs = true;
        } else if(element=="element"){
            iss >> element;
            if(element=="vertex"){ iss>>element; nvertex = stoi(element); }
            else { iss>>element; nfaces = stoi(element); }
        }
    }

    // Check if at least the vertex are defined in the file. UVs and normals are optional
    if(!foundXyz){ PolyRenderer::polyMsg("\e[1;91m  err parsing '" + string(path) + "': vertex property missing!\e[0m\n"); exit(EXIT_FAILURE); }


    // Next we get nvertex lines with the vertex data, followed by nfaces lines with the tri data
    vector<Vertex> vertices;
    for(uint i=0; i<nvertex+nfaces; i++){
        input.getline(buff, 128u);
        istringstream iss(buff);
        if(i<nvertex){  // Read vertex data: coordinates, normals and UVs, in that order
            iss >> element; x = stof(element); iss >> element; y = stof(element); iss >> element; z = stof(element);
            if(foundNormals){
                iss >> element; nx = stof(element); iss >> element; ny = stof(element); iss >> element; nz = stof(element);
            } else { nx = ny = nz = 0.0f; }
            if(foundUVs){
                iss >> element; u = stof(element); iss >> element; v = stof(element);
            } else { u = v = 0.0f; }

            Vertex vertex = Vertex(Vec3(x,y,z), Vec3(nx,ny,nz), u, v);
            vertices.push_back(vertex);

        } else {        // Read tri data. Check if the line starts whith a 3
            iss >> element;
            if(element!="3") { PolyRenderer::polyMsg("\e[1;91m  err parsing '" + string(path) + "': expected tri faces!\e[0m\n"); exit(EXIT_FAILURE); }
            iss >> element; a = vertices[stoi(element)];
            iss >> element; b = vertices[stoi(element)];
            iss >> element; c = vertices[stoi(element)];

            Tri tri = Tri(a, b, c, matId, flags);
            tris.push_back(tri);
        }
    }

    input.close();
    if(input.is_open()){ PolyRenderer::polyMsg("\e[1;91m  err closing '" + string(path) + "'\e[0m\n"); exit(EXIT_FAILURE); }
    PolyRenderer::polyMsg("\e[1;93m  loaded " + to_string(tris.size()) + " tris from '" + string(path) + "'\e[0m\n");
}
void Poly::move(Vec3 m){
    #pragma omp parallel for num_threads(omp_get_max_threads()) schedule(static)
    for(Tri& tri : tris) tri.move(m);
}
void Poly::scale(Vec3 s){
    #pragma omp parallel for num_threads(omp_get_max_threads()) schedule(static)
    for(Tri& tri : tris) tri.scale(s);
}
void Poly::scale(float s){
    #pragma omp parallel for num_threads(omp_get_max_threads()) schedule(static)
    for(Tri& tri : tris) tri.scale(s);
}
void Poly::rotate(Vec3 r){
    #pragma omp parallel for num_threads(omp_get_max_threads()) schedule(static)
    for(Tri& tri : tris) tri.rotate(r);
}
void Poly::rotateX(float r){
    #pragma omp parallel for num_threads(omp_get_max_threads()) schedule(static)
    for(Tri& tri : tris) tri.rotateX(r);
}
void Poly::rotateY(float r){
    #pragma omp parallel for num_threads(omp_get_max_threads()) schedule(static)
    for(Tri& tri : tris) tri.rotateY(r);
}
void Poly::rotateZ(float r){
    #pragma omp parallel for num_threads(omp_get_max_threads()) schedule(static)
    for(Tri& tri : tris) tri.rotateZ(r);
}