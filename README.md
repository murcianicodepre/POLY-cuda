# POLY-cuda
90s inspired polygon raytracer now enchanced with GPU acceleration

* 1280×960 4:3 rendering output to <code>.png</code>
* <code>.poly</code> scripts v2 describing objects, materials and light sources
* <code>.ply</code> 3D models
* GPU accelerated raytracing pipeline with _mind-boggling_ effects:
    * <code>.png</code> 1024×1024 textures with bump mapping
    * Reflections and refractions
    * _Blinn-Phong_ shading with _hard_ shadows

## <code>.poly</code> script example
This example script renders the text "poly" with transparent background:
```yaml
# Define camera
camera:
  position: [0.0, 0.0, -0.2]
  fov: 50.0
# Define materials
materials:
  - name: "mat1"
    texture: "textures/poly_texture.png"
    diffuse: 2.0
    specular: 32.0
# Define objects
objects:
  - name: "poly"
    file: "models/poly.ply"
    material: "mat1"
    transforms:
      scale: 0.4
      move: [0.0, -0.1, 1.0]
# Finally, define scene lights
lights:
  - type: point
    position: [0.5, 1.0, -1.0]
    color: [255,255,255]
    intensity: 0.8
```


![demo](/klein.png "render demo")

## How to get <code>.ply</code> 3D models
Using Blender, import any 3D model and then **File > Export > Standford PLY(.ply)**. In the window, select "ASCII", "UV Coordinates", "Vertex Normals" and "Triangulated Mesh" and set "Forward Axis" to "Z" and "Up Axis" to "Y". Remenber that all models will be flipped in the X axis.