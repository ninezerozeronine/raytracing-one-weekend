def gen_row_of_spheres_world():
    grey_mat = materials.PointOnHemiSphereMaterial(numpy.array([0.5, 0.5, 0.5]))

    # World setup
    world = World()

    # Row of spheres front to back
    world.renderables.append(Sphere(numpy.array([-3.0, 0.0, -7.0]), 3.0, grey_mat))
    world.renderables.append(Sphere(numpy.array([0.0, 0.0, -10.0]), 3.0, grey_mat))
    world.renderables.append(Sphere(numpy.array([3.0, 0.0, -13.0]), 3.0, grey_mat))
    world.renderables.append(Sphere(numpy.array([6.0, 0.0, -17.0]), 3.0, grey_mat))

    return world


def gen_glass_experiment_world():
    ground_mat = materials.PointOnHemiSphereMaterial(numpy.array([(148/256), (116/256), (105/256)]))
    blue_mat = materials.PointOnHemiSphereMaterial(numpy.array([0.1, 0.2, 0.5]))
    discrete_normal_mat = materials.NormalToDiscreteRGBMaterial()
    metal_mat = materials.MetalMaterial(numpy.array([0.8, 0.8, 0.8]), 0.0)
    glass_mat = materials.DielectricMaterial(1.5)

    # World setup
    world = World()

    # Row of spheres front to back
    # world.renderables.append(Sphere(numpy.array([-3.0, 0.0, -7.0]), 3.0, grey_mat))
    # world.renderables.append(Sphere(numpy.array([0.0, 0.0, -10.0]), 3.0, grey_mat))
    # world.renderables.append(Sphere(numpy.array([3.0, 0.0, -13.0]), 3.0, grey_mat))
    # world.renderables.append(Sphere(numpy.array([6.0, 0.0, -17.0]), 3.0, grey_mat))

    # Line of shperes left to right
    world.renderables.append(Sphere(numpy.array([-6.0, 0.0, -10.0]), 3.0, glass_mat))
    world.renderables.append(Sphere(numpy.array([0.0, 0.0, -10.0]), 3.0, blue_mat))
    world.renderables.append(Sphere(numpy.array([6.0, 0.0, -10.0]), 3.0, discrete_normal_mat))

    # Floating sphere above and to the right of the left/right line.
    world.renderables.append(Sphere(numpy.array([5.0, 6.0, -16.0]), 3.0, metal_mat))

    # Sphere embedded in the ground behind the glass sphere
    world.renderables.append(Sphere(numpy.array([-9.0, -3.0, -16.0]), 3.0, discrete_normal_mat))

    for x in range(3):
        for y in range(3):
            world.renderables.append(Sphere(numpy.array([(x*1.3)-12.0, (y*2.0)+1.5, -16.0]), 0.3, discrete_normal_mat))

    # Ground Sphere
    world.renderables.append(Sphere(numpy.array([0.0, -503.0, -10.0]), 500.0, ground_mat))

    return world


def gen_simple_world():
    ground_mat = materials.PointOnHemiSphereMaterial(numpy.array([(148/256), (116/256), (105/256)]))
    blue_mat = materials.PointOnHemiSphereMaterial(numpy.array([0.1, 0.2, 0.5]))
    discrete_normal_mat = materials.NormalToDiscreteRGBMaterial()
    metal_mat = materials.MetalMaterial(numpy.array([0.8, 0.8, 0.8]), 0.0)
    glass_mat = materials.DielectricMaterial(1.5)

    world = World()

    # Line of shperes left to right
    world.renderables.append(Sphere(numpy.array([-6.0, 0.0, -10.0]), 3.0, glass_mat))
    world.renderables.append(Sphere(numpy.array([0.0, 0.0, -10.0]), 3.0, blue_mat))
    world.renderables.append(Sphere(numpy.array([6.0, 0.0, -10.0]), 3.0, discrete_normal_mat))

    # Floating sphere above and to the right of the left/right line.
    world.renderables.append(Sphere(numpy.array([5.0, 6.0, -16.0]), 3.0, metal_mat))

    # Ground Sphere
    world.renderables.append(Sphere(numpy.array([0.0, -503.0, -10.0]), 500.0, ground_mat))

    return world


def focal_length_world():
    blue_mat = materials.PointOnHemiSphereMaterial(numpy.array([0.1, 0.2, 0.5]))
    normal_mat = materials.NormalToRGBMaterial()

    world = World()

    world.renderables.append(Sphere(numpy.array([-2.0, 0.0, -10.0]), 2.0, normal_mat))
    world.renderables.append(Sphere(numpy.array([2.0, 0.0, -10.0]), 2.0, blue_mat))

    return world


def positionable_cam_scene():

    cam_pos = numpy.array([-2.0, 2.0, 1.0])
    cam_lookat = numpy.array([0.0, 0.0, -1.0])
    camera = Camera(cam_pos, cam_lookat, ASPECT_RATIO, 45.0)

    blue_mat = materials.PointOnHemiSphereMaterial(numpy.array([0.1, 0.2, 0.5]))
    ground_mat = materials.PointOnHemiSphereMaterial(numpy.array([0.8, 0.8, 0.0]))
    metal_mat = materials.MetalMaterial(numpy.array([0.8, 0.6, 0.2]), 0.0)
    glass_mat = materials.DielectricMaterial(1.5)

    world = World()

    # Ground
    world.renderables.append(Sphere(numpy.array([0.0, -100.5, -1.0]), 100.0, ground_mat))

    # Glass, blue, metal
    world.renderables.append(Sphere(numpy.array([-1.0, 0.0, -1.0]), 0.5, glass_mat))
    world.renderables.append(Sphere(numpy.array([-1.0, 0.0, -1.0]), -0.45, glass_mat))
    world.renderables.append(Sphere(numpy.array([0.0, 0.0, -1.0]), 0.5, blue_mat))
    world.renderables.append(Sphere(numpy.array([1.0, 0.0, -1.0]), 0.5, metal_mat))

    return world, camera


def dof_cam_scene():

    cam_pos = numpy.array([3.0, 3.0, 2.0])
    cam_lookat = numpy.array([0.0, 0.0, -1.0])
    pos_to_lookat = cam_lookat - cam_pos
    focus_dist = numpy.sqrt(pos_to_lookat.dot(pos_to_lookat))
    aperture = 2.0
    camera = Camera(cam_pos, cam_lookat, focus_dist, aperture, ASPECT_RATIO, 45.0)

    blue_mat = materials.PointOnHemiSphereMaterial(numpy.array([0.1, 0.2, 0.5]))
    ground_mat = materials.PointOnHemiSphereMaterial(numpy.array([0.8, 0.8, 0.0]))
    metal_mat = materials.MetalMaterial(numpy.array([0.8, 0.6, 0.2]), 0.0)
    glass_mat = materials.DielectricMaterial(1.5)

    world = World()

    # Ground
    world.renderables.append(Sphere(numpy.array([0.0, -100.5, -1.0]), 100.0, ground_mat))

    # Glass, blue, metal
    world.renderables.append(Sphere(numpy.array([-1.0, 0.0, -1.0]), 0.5, glass_mat))
    world.renderables.append(Sphere(numpy.array([-1.0, 0.0, -1.0]), -0.45, glass_mat))
    world.renderables.append(Sphere(numpy.array([0.0, 0.0, -1.0]), 0.5, blue_mat))
    world.renderables.append(Sphere(numpy.array([1.0, 0.0, -1.0]), 0.5, metal_mat))

    return world, camera


def many_spheres_scene():
    cam_pos = numpy.array([13.0, 2.0, 3.0])
    cam_lookat = numpy.array([0.0, 0.5, 0.0])
    focus_dist = 10
    aperture = 0.1
    camera = Camera(cam_pos, cam_lookat, focus_dist, aperture, ASPECT_RATIO, 30.0)

    ground_mat = materials.PointOnHemiSphereMaterial(numpy.array([0.5, 0.5, 0.5]))
    brown_mat = materials.PointOnHemiSphereMaterial(numpy.array([0.4, 0.2, 0.1]))
    glass_mat = materials.DielectricMaterial(1.5)
    metal_mat = materials.MetalMaterial(numpy.array([0.7, 0.6, 0.5]), 0.0)

    world = World()

    # Ground
    world.renderables.append(Sphere(numpy.array([0.0, -1000.0, 0.0]), 1000.0, ground_mat))

    # Brown, Glass, Metal
    world.renderables.append(Sphere(numpy.array([-4.0, 1.0, 0.0]), 1.0, brown_mat))
    world.renderables.append(Sphere(numpy.array([0.0, 1.0, 0.0]), 1.0, glass_mat))
    world.renderables.append(Sphere(numpy.array([4.0, 1.0, 0.0]), 1.0, metal_mat))

    with open("sphere_data.json") as file_handle:
        sphere_data = json.load(file_handle)

    for sphere in sphere_data:
        material = materials.NormalToDiscreteRGBMaterial()
        if sphere["material"] == "diffuse":
            material = materials.PointOnHemiSphereMaterial(numpy.array(sphere["colour"]))
        if sphere["material"] == "glass":
            material = materials.DielectricMaterial(sphere["ior"])
        if sphere["material"] == "metal":
            material = materials.MetalMaterial(numpy.array(sphere["colour"]), sphere["fuzziness"])
        world.renderables.append(
            Sphere(numpy.array(sphere["pos"]), sphere["radius"], material)
        )

    return world, camera


def many_spheres_scene_accelerated():
    cam_pos = numpy.array([13.0, 2.0, 3.0])
    cam_lookat = numpy.array([0.0, 0.5, 0.0])
    focus_dist = 10
    aperture = 0.1
    camera = Camera(cam_pos, cam_lookat, focus_dist, aperture, ASPECT_RATIO, 30.0)

    ground_mat = materials.PointOnHemiSphereMaterial(numpy.array([0.5, 0.5, 0.5]))
    brown_mat = materials.PointOnHemiSphereMaterial(numpy.array([0.4, 0.2, 0.1]))
    glass_mat = materials.DielectricMaterial(1.5)
    metal_mat = materials.MetalMaterial(numpy.array([0.7, 0.6, 0.5]), 0.0)

    world = World()

    # Ground
    world.renderables.append(Sphere(numpy.array([0.0, -1000.0, 0.0]), 1000.0, ground_mat))

    # Brown, Glass, Metal
    world.renderables.append(Sphere(numpy.array([-4.0, 1.0, 0.0]), 1.0, brown_mat))
    world.renderables.append(Sphere(numpy.array([0.0, 1.0, 0.0]), 1.0, glass_mat))
    world.renderables.append(Sphere(numpy.array([4.0, 1.0, 0.0]), 1.0, metal_mat))

    with open("sphere_data.json") as file_handle:
        sphere_data = json.load(file_handle)

    all_spheres = SphereGroup()
    print(len(sphere_data))
    for sphere in sphere_data:
        material = materials.NormalToDiscreteRGBMaterial()
        if sphere["material"] == "diffuse":
            material = materials.PointOnHemiSphereMaterial(numpy.array(sphere["colour"]))
        if sphere["material"] == "glass":
            material = materials.DielectricMaterial(sphere["ior"])
        if sphere["material"] == "metal":
            material = materials.MetalMaterial(numpy.array(sphere["colour"]), sphere["fuzziness"])
        all_spheres.add_sphere(sphere["pos"], sphere["radius"], material)

    world.renderables.append(all_spheres)

    return world, camera


def mttriangles_scene():
    cam_pos = numpy.array([0.0, 1.0, 6.0])
    cam_lookat = numpy.array([0.0, 0.5, 0.0])
    # cam_pos = numpy.array([5.0, 5.0, 5.0])
    # cam_lookat = numpy.array([0.0, 0.5, 0.0])
    focus_dist = 10
    aperture = 0.0
    horizontal_fov = 50.0
    camera = Camera(cam_pos, cam_lookat, focus_dist, aperture, ASPECT_RATIO, horizontal_fov)

    ground_mat = materials.PointOnHemiSphereMaterial(numpy.array([0.5, 0.5, 0.5]))
    brown_mat = materials.PointOnHemiSphereMaterial(numpy.array([0.4, 0.2, 0.1]))
    blue_mat = materials.PointOnHemiSphereMaterial(numpy.array([0.1, 0.2, 0.5]))
    green_mat = materials.PointOnHemiSphereMaterial(numpy.array([0.1, 0.6, 0.15]))
    red_mat = materials.PointOnHemiSphereMaterial(numpy.array([0.8, 0.1, 0.1]))
    metal_mat = materials.MetalMaterial(numpy.array([0.8, 0.8, 0.8]), 0.0)

    world = World()

    # Ground
    world.renderables.append(Sphere(numpy.array([0.0, -1000.0, 0.0]), 1000.0, ground_mat))

    # Brown sphere
    world.renderables.append(Sphere(numpy.array([-1.0, 0.5, 0.0]), 0.5, brown_mat))

    # Red sphere
    world.renderables.append(Sphere(numpy.array([-2.3, 0.3, -0.4]), 0.3, red_mat))

    # Blue triangle
    world.renderables.append(MTTriangle(
        numpy.array([1.0, 0.0, 0.0]),
        numpy.array([2.0, 0.0, 0.0]),
        numpy.array([1.0, 2.0, 0.0]),
        blue_mat
    ))

    # Green triangle
    world.renderables.append(MTTriangle(
        numpy.array([-2.5, 0.0, 0.0]),
        numpy.array([-1.5, 0.0, 0.0]),
        numpy.array([-2.0, 0.75, 0.0]),
        green_mat
    ))

    # Mirror triangle
    offset = numpy.array([0.0, 0.0, -2.0])
    world.renderables.append(MTTriangle(
        numpy.array([-2.0, 0.0, -1.0]) + offset,
        numpy.array([2.0, 0.0, 1.0]) + offset,
        numpy.array([0.0, 2.0, 0.0]) + offset,
        metal_mat
    ))

    return world, camera


def mttriangles_scene_accelerated():
    cam_pos = numpy.array([6.0, 3.0, 6.0])
    cam_lookat = numpy.array([0.0, 0.3, 0.0])
    # cam_pos = numpy.array([5.0, 5.0, 5.0])
    # cam_lookat = numpy.array([0.0, 0.5, 0.0])
    focus_dist = 10
    aperture = 0.0
    horizontal_fov = 25.0
    camera = Camera(cam_pos, cam_lookat, focus_dist, aperture, ASPECT_RATIO, horizontal_fov)

    ground_mat = materials.PointOnHemiSphereMaterial(numpy.array([0.5, 0.5, 0.5]))
    brown_mat = materials.PointOnHemiSphereMaterial(numpy.array([0.4, 0.2, 0.1]))
    blue_mat = materials.PointOnHemiSphereMaterial(numpy.array([0.1, 0.2, 0.5]))
    green_mat = materials.PointOnHemiSphereMaterial(numpy.array([0.1, 0.6, 0.15]))
    red_mat = materials.PointOnHemiSphereMaterial(numpy.array([0.8, 0.1, 0.1]))
    metal_mat = materials.MetalMaterial(numpy.array([0.8, 0.8, 0.8]), 0.0)

    world = World()

    # Ground
    world.renderables.append(Sphere(numpy.array([0.0, -1000.0, 0.0]), 1000.0, ground_mat))

    # Brown sphere
    world.renderables.append(Sphere(numpy.array([-1.0, 0.5, 0.0]), 0.5, brown_mat))

    # Red sphere
    world.renderables.append(Sphere(numpy.array([-2.3, 0.3, -0.4]), 0.3, red_mat))

    tri_grp = MTTriangleGroup()

    for x in range(10):
        for y in range(10):
            for z in range(10):
                offset = numpy.array([x/10, y/10, z/10])
                tri_grp.add_triangle(
                    numpy.array([-0.1, 0.0, 0.0]) + offset,
                    numpy.array([0.1, 0.0, 0.0]) + offset,
                    numpy.array([0.0, 0.1, 0.0]) + offset,
                    blue_mat
                )

    world.renderables.append(tri_grp)

    return world, camera


def bunny_scene():
    cam_pos = numpy.array([-2.0, 3.5, 8.0])
    cam_lookat = numpy.array([-0.5, 1.7, 0.0])
    # cam_pos = numpy.array([5.0, 5.0, 5.0])
    # cam_lookat = numpy.array([0.0, 0.5, 0.0])
    focus_dist = 10
    aperture = 0.0
    horizontal_fov = 53.0
    camera = Camera(cam_pos, cam_lookat, focus_dist, aperture, ASPECT_RATIO, horizontal_fov)

    ground_mat = materials.PointOnHemiSphereMaterial(numpy.array([0.5, 0.5, 0.5]))
    brown_mat = materials.PointOnHemiSphereMaterial(numpy.array([0.4, 0.2, 0.1]))
    blue_mat = materials.PointOnHemiSphereMaterial(numpy.array([0.1, 0.2, 0.5]))
    green_mat = materials.PointOnHemiSphereMaterial(numpy.array([0.1, 0.6, 0.15]))
    red_mat = materials.PointOnHemiSphereMaterial(numpy.array([0.8, 0.1, 0.1]))
    metal_mat = materials.MetalMaterial(numpy.array([0.8, 0.8, 0.8]), 0.0)
    glass_mat = materials.DielectricMaterial(1.5)
    normal_mat = materials.NormalToRGBMaterial()

    world = World()

    # Ground
    world.renderables.append(Sphere(numpy.array([0.0, -1000.0, 0.0]), 1000.0, ground_mat))

    # # Metal sphere
    # world.renderables.append(Sphere(numpy.array([-3.5, 1.0, -1.5]), 1.0, metal_mat))

    # # Glass sphere
    # world.renderables.append(Sphere(numpy.array([-0.7, 3.0, 5.5]), 0.4, glass_mat))

    # # Normal sphere
    # world.renderables.append(Sphere(numpy.array([-2.6, 0.4, 0.5]), 0.4, normal_mat))

    tri_grp = MTTriangleGroup()

    obj_mesh = OBJTriMesh()
    obj_mesh.read("bunny.obj")

    smallest_y = min([vertex[1] for vertex in obj_mesh.vertices])

    for triangle in obj_mesh.faces:
        tri_grp.add_triangle(
            numpy.array([
                obj_mesh.vertices[triangle[0][0]][0],
                obj_mesh.vertices[triangle[0][0]][1] - smallest_y,
                obj_mesh.vertices[triangle[0][0]][2],
            ]),
            numpy.array([
                obj_mesh.vertices[triangle[1][0]][0],
                obj_mesh.vertices[triangle[1][0]][1] - smallest_y,
                obj_mesh.vertices[triangle[1][0]][2],
            ]),
            numpy.array([
                obj_mesh.vertices[triangle[2][0]][0],
                obj_mesh.vertices[triangle[2][0]][1] - smallest_y,
                obj_mesh.vertices[triangle[2][0]][2],
            ]),
            blue_mat
        )

    world.renderables.append(tri_grp)

    return world, camera


def bunnies_scene():
    cam_pos = numpy.array([3.0, 5.0, 10.0])
    cam_lookat = numpy.array([-1.0, 1.2, 0.0])
    # cam_pos = numpy.array([5.0, 5.0, 5.0])
    # cam_lookat = numpy.array([0.0, 0.5, 0.0])
    focus_dist = 10
    aperture = 0.0
    horizontal_fov = 60.0
    camera = Camera(cam_pos, cam_lookat, focus_dist, aperture, ASPECT_RATIO, horizontal_fov)

    ground_mat = materials.PointOnHemiSphereCheckerboardMaterial(
        numpy.array([1.0, 1.0, 1.0]),
        numpy.array([0.0, 0.0, 0.0]),
        numpy.array([0.5, 0.5, 0.5]),
        numpy.array([0.3, 0.3, 0.3]),
    )
    red_blue_mat = materials.PointOnHemiSphereCheckerboardMaterial(
        numpy.array([2.0, 2.0, 2.0]),
        numpy.array([0.2, 0.2, 0.2]),
        numpy.array([0.7, 0.3, 0.2]),
        numpy.array([0.1, 0.2, 0.5]),
    )
    metal_mat = materials.MetalMaterial(numpy.array([0.8, 0.8, 0.8]), 0.0)
    glass_mat = materials.DielectricMaterial(1.5)
    normal_mat = materials.NormalToRGBMaterial()

    world = World()

    # Ground
    world.renderables.append(Sphere(numpy.array([0.0, -1000.0, 0.0]), 1000.0, ground_mat))

    obj_mesh = OBJTriMesh()
    obj_mesh.read("bunny.obj")

    smallest_y = min([vertex[1] for vertex in obj_mesh.vertices])

    spacing = 2.0

    # Metal bunny
    metal_grp = MTTriangleGroup()
    offset_x = -spacing
    offset_z = -spacing
    for triangle in obj_mesh.faces:
        metal_grp.add_triangle(
            numpy.array([
                obj_mesh.vertices[triangle[0][0]][0] + offset_x,
                obj_mesh.vertices[triangle[0][0]][1] - smallest_y,
                obj_mesh.vertices[triangle[0][0]][2] + offset_z,
            ]),
            numpy.array([
                obj_mesh.vertices[triangle[1][0]][0] + offset_x,
                obj_mesh.vertices[triangle[1][0]][1] - smallest_y,
                obj_mesh.vertices[triangle[1][0]][2] + offset_z,
            ]),
            numpy.array([
                obj_mesh.vertices[triangle[2][0]][0] + offset_x,
                obj_mesh.vertices[triangle[2][0]][1] - smallest_y,
                obj_mesh.vertices[triangle[2][0]][2] + offset_z,
            ]),
            metal_mat
        )
    world.renderables.append(metal_grp)

    # Glass bunny
    glass_grp = MTTriangleGroup()
    offset_x = spacing
    offset_z = spacing
    for triangle in obj_mesh.faces:
        glass_grp.add_triangle(
            numpy.array([
                obj_mesh.vertices[triangle[0][0]][0] + offset_x,
                obj_mesh.vertices[triangle[0][0]][1] - smallest_y,
                obj_mesh.vertices[triangle[0][0]][2] + offset_z,
            ]),
            numpy.array([
                obj_mesh.vertices[triangle[1][0]][0] + offset_x,
                obj_mesh.vertices[triangle[1][0]][1] - smallest_y,
                obj_mesh.vertices[triangle[1][0]][2] + offset_z,
            ]),
            numpy.array([
                obj_mesh.vertices[triangle[2][0]][0] + offset_x,
                obj_mesh.vertices[triangle[2][0]][1] - smallest_y,
                obj_mesh.vertices[triangle[2][0]][2] + offset_z,
            ]),
            glass_mat
        )
    world.renderables.append(glass_grp)

    # Red/blue bunny
    blue_grp = MTTriangleGroup()
    offset_x = -spacing
    offset_z = spacing
    for triangle in obj_mesh.faces:
        blue_grp.add_triangle(
            numpy.array([
                obj_mesh.vertices[triangle[0][0]][0] + offset_x,
                obj_mesh.vertices[triangle[0][0]][1] - smallest_y,
                obj_mesh.vertices[triangle[0][0]][2] + offset_z,
            ]),
            numpy.array([
                obj_mesh.vertices[triangle[1][0]][0] + offset_x,
                obj_mesh.vertices[triangle[1][0]][1] - smallest_y,
                obj_mesh.vertices[triangle[1][0]][2] + offset_z,
            ]),
            numpy.array([
                obj_mesh.vertices[triangle[2][0]][0] + offset_x,
                obj_mesh.vertices[triangle[2][0]][1] - smallest_y,
                obj_mesh.vertices[triangle[2][0]][2] + offset_z,
            ]),
            red_blue_mat
        )
    world.renderables.append(blue_grp)


    # Normal bunny
    normal_grp = MTTriangleGroup()
    offset_x = spacing
    offset_z = -spacing
    for triangle in obj_mesh.faces:
        normal_grp.add_triangle(
            numpy.array([
                obj_mesh.vertices[triangle[0][0]][0] + offset_x,
                obj_mesh.vertices[triangle[0][0]][1] - smallest_y,
                obj_mesh.vertices[triangle[0][0]][2] + offset_z,
            ]),
            numpy.array([
                obj_mesh.vertices[triangle[1][0]][0] + offset_x,
                obj_mesh.vertices[triangle[1][0]][1] - smallest_y,
                obj_mesh.vertices[triangle[1][0]][2] + offset_z,
            ]),
            numpy.array([
                obj_mesh.vertices[triangle[2][0]][0] + offset_x,
                obj_mesh.vertices[triangle[2][0]][1] - smallest_y,
                obj_mesh.vertices[triangle[2][0]][2] + offset_z,
            ]),
            normal_mat
        )
    world.renderables.append(normal_grp)


    return world, camera


def checkerboard_scene():
    cam_pos = numpy.array([10.0, 10.0, 10.0])
    cam_lookat = numpy.array([0.0, 0.0, 0.0])
    focus_dist = 10
    aperture = 0.0
    horizontal_fov = 50.0
    camera = Camera(cam_pos, cam_lookat, focus_dist, aperture, ASPECT_RATIO, horizontal_fov)

    ground_mat = materials.PointOnHemiSphereCheckerboardMaterial(
        numpy.array([1.0, 1.0, 1.0]),
        numpy.array([0.0, 0.0, 0.0]),
        numpy.array([0.5, 0.5, 0.5]),
        numpy.array([0.3, 0.3, 0.3]),
    )
    sphere_mat = materials.PointOnHemiSphereCheckerboardMaterial(
        numpy.array([1.0, 1.0, 1.0]),
        numpy.array([0.0, 0.0, 0.0]),
        numpy.array([0.7, 0.3, 0.2]),
        numpy.array([0.1, 0.2, 0.5]),
    )

    world = World()

    # Ground
    world.renderables.append(Sphere(numpy.array([0.0, -1000.0, 0.0]), 1000.0, ground_mat))

    # Sphere
    world.renderables.append(Sphere(numpy.array([0.0, 2.0, 0.0]), 2.0, sphere_mat))

    return world, camera