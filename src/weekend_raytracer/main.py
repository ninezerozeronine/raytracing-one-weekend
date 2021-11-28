"""
Main entry point for renderer functionality
"""
import math
from random import random
import time
import datetime
import json

from PIL import Image, ImageDraw
import numpy
import humanize

from .ray import Ray
from .renderable import World
from .sphere_group import SphereGroup
from .triangle_group import TriangleGroup
from .disk import Disk
from .obj_tri_mesh import OBJTriMesh
from .camera import Camera
from . import materials


IMG_WIDTH = 160
IMG_HEIGHT = 90
ASPECT_RATIO = IMG_WIDTH/IMG_HEIGHT
PIXEL_SAMPLES = 10
MAX_BOUNCES = 10
HORIZON_COLOUR = numpy.array([1.0, 1.0, 1.0], dtype=numpy.single)
SKY_COLOUR = numpy.array([0.5, 0.7, 1.0], dtype=numpy.single)
RNG = numpy.random.default_rng()



def generate_test_image():
    """
    Generate a test image.
    """

    image = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT))
    draw = ImageDraw.Draw(image)

    for row in range(IMG_HEIGHT):
        for column in range(IMG_WIDTH):
            r = int(255 * (column / (IMG_WIDTH - 1)))
            g = int(255 * (row / (IMG_HEIGHT - 1)))
            b = int(255 * 0.25)
            draw.point((column, row), fill=(r, g, b))

    image.show()


def generate_image_from_data(img_data):
    """
    Write out an image to disk.
    """

    image = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT))
    draw = ImageDraw.Draw(image)

    for row in range(IMG_HEIGHT):
        for column in range(IMG_WIDTH):
            # Flip row to account for image data having 0, 0 at bottom
            # left, rather than top left in a PIL image.
            flipped_row = IMG_HEIGHT - 1 - row
            if (column, flipped_row) in img_data:
                draw.point(
                    (column, row),
                    fill=(
                        int(255 * img_data[(column, flipped_row)][0]),
                        int(255 * img_data[(column, flipped_row)][1]),
                        int(255 * img_data[(column, flipped_row)][2]),
                    )
                )
            else:
                draw.point(
                    (column, row),
                    fill=(255, 255, 255)
                )

    image.show()
    image.save("tmp_image.png")


def render():

    # camera, object_groups, material_map = numpy_dielectric_scene()
    # camera, object_groups, material_map = numpy_glass_experiment_scene()
    # camera, object_groups, material_map = numpy_triangles_scene()
    # camera, object_groups, material_map = numpy_simple_sphere_scene()
    # camera, object_groups, material_map = numpy_one_weekend_demo_scene()
    camera, object_groups, material_map = ray_group_triangle_group_bunny_scene()
    # camera, object_groups, material_map = blender_cylinder_vert_normals_test_scene()
    # camera, object_groups, material_map = sphere_types_test_scene()
    # camera, object_groups, material_map = texture_test_scene()
    # camera, object_groups, material_map = numpy_bunnies_scene()
    # camera, object_groups, material_map = numpy_cow_scene()
    # camera, object_groups, material_map = disk_test_scene()
    # camera, object_groups, material_map = smooth_normal_test_scene()

    start_time = time.perf_counter()

    print("Generating arrays")
    ray_colours = numpy.ones(
        (IMG_WIDTH, IMG_HEIGHT, PIXEL_SAMPLES, MAX_BOUNCES + 1, 3), dtype=numpy.single
    )

    print("Filling ray arrays")
    ray_origins, ray_directions = camera.get_ray_components(IMG_WIDTH, IMG_HEIGHT, PIXEL_SAMPLES)
    print
    ray_origins = ray_origins.reshape(-1, 3)
    ray_directions = ray_directions.reshape(-1, 3)
    ray_colours = ray_colours.reshape(-1, MAX_BOUNCES + 1, 3)

    # For the 0th bounce, every ray is active
    active_ray_indecies = numpy.arange(ray_origins.shape[0])

    bounce = 0
    while bounce <= MAX_BOUNCES:
        print(f"Bounce {bounce} of {MAX_BOUNCES}, {active_ray_indecies.shape[0]} rays.")

        if bounce != MAX_BOUNCES:

            num_rays = active_ray_indecies.shape[0]
            ray_hits = numpy.full((num_rays,), False)
            hit_ts = numpy.full((num_rays,), 5001.0)
            hit_pts = numpy.full((num_rays, 3), 0.0)
            hit_normals = numpy.full((num_rays, 3), 0.0)
            hit_uvs = numpy.full((num_rays, 2), 0.0)
            hit_material_indecies = numpy.full((num_rays,), -1, dtype=numpy.ubyte)
            back_facing = numpy.full((num_rays,), False)

            for object_group in object_groups:

                # Eeeeew - this is getting a bit out of hand :(
                # Also need to catch the case where there are no hits.
                (
                    tmp_ray_hits,
                    tmp_hit_ts,
                    tmp_hit_pts,
                    tmp_hit_normals,
                    tmp_hit_uvs,
                    tmp_hit_material_indecies,
                    tmp_back_facing
                ) = object_group.get_hits(
                    ray_origins[active_ray_indecies],
                    ray_directions[active_ray_indecies],
                    0.001,
                    1000.0
                )

                ray_hits = ray_hits | tmp_ray_hits
                condition = tmp_ray_hits & (tmp_hit_ts < hit_ts)
                hit_ts = numpy.where(
                    condition,
                    tmp_hit_ts,
                    hit_ts
                )
                hit_pts = numpy.where(
                    condition[..., numpy.newaxis],
                    tmp_hit_pts,
                    hit_pts
                )
                hit_normals = numpy.where(
                    condition[..., numpy.newaxis],
                    tmp_hit_normals,
                    hit_normals
                )
                hit_uvs = numpy.where(
                    condition[..., numpy.newaxis],
                    tmp_hit_uvs,
                    hit_uvs)
                hit_material_indecies = numpy.where(
                    condition,
                    tmp_hit_material_indecies,
                    hit_material_indecies
                )
                back_facing = numpy.where(
                    condition,
                    tmp_back_facing,
                    back_facing
                )

            # print(ray_hits.shape[0])
            # print(hit_ts.shape[0])
            # print(hit_pts.shape[0])
            # print(hit_normals.shape[0])
            # print(hit_material_indecies.shape[0])
            # print(back_facing.shape[0])
            # print(numpy.any(ray_hits))
            ray_misses = numpy.logical_not(ray_hits)

            material_absorbsions = numpy.full((ray_hits.shape[0]), False)

            for material_index, material in material_map.items():
                material_matches = (hit_material_indecies == material_index) & ray_hits

                scatter_ray_origins, scatter_ray_dirs, scatter_cols, scatter_absorbtions = material.scatter(
                    ray_directions[active_ray_indecies[material_matches]],
                    hit_pts[material_matches],
                    hit_normals[material_matches],
                    hit_uvs[material_matches], 
                    back_facing[material_matches]
                )

                ray_origins[active_ray_indecies[material_matches]] = scatter_ray_origins
                ray_directions[active_ray_indecies[material_matches]] = scatter_ray_dirs
                ray_colours[active_ray_indecies[material_matches], bounce] = scatter_cols
                material_absorbsions[material_matches] = scatter_absorbtions


            # Y component is somewhere between -1 and 1. Map it into
            # a 0 to 1 range.
            # Lerp between horizon and sky colour based on mapped Y
            ts = (ray_directions[active_ray_indecies[ray_misses], 1] + 1.0) * 0.5
            ray_colours[active_ray_indecies[ray_misses], bounce] = (1.0 - ts)[..., numpy.newaxis] * HORIZON_COLOUR + ts[..., numpy.newaxis] * SKY_COLOUR

            # The next round of rays are the ones that hit something and were not absorbed
            active_ray_indecies = active_ray_indecies[numpy.logical_and(ray_hits, numpy.logical_not(material_absorbsions))]
        else:
            ray_colours[active_ray_indecies, bounce] = 0.0

        bounce += 1


    ray_bounce_cols_multiplied = numpy.prod(ray_colours, axis=1)
    ray_colours_stacked = ray_bounce_cols_multiplied.reshape(IMG_WIDTH, IMG_HEIGHT, PIXEL_SAMPLES, 3)
    ray_cols_sample_avg = numpy.mean(ray_colours_stacked, axis=2)
    sqrt_ray_cols = numpy.sqrt(ray_cols_sample_avg)
    
    print("Filling pixel data")
    pixel_data = {}
    for y_coord in range(IMG_HEIGHT):
        for x_coord in range(IMG_WIDTH):
            # The squareroot is for a 2.0 gamma correction
            pixel_data[(x_coord, y_coord)] = sqrt_ray_cols[x_coord, y_coord]

    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:0.4f} seconds.")

    return pixel_data


def numpy_bunnies_scene():
    cam_pos = numpy.array([3.0, 5.0, 10.0])
    cam_lookat = numpy.array([-1.0, 1.2, 0.0])
    # cam_pos = numpy.array([5.0, 5.0, 5.0])
    # cam_lookat = numpy.array([0.0, 0.5, 0.0])
    focus_dist = 10
    aperture = 0.0
    horizontal_fov = 60.0
    camera = Camera(cam_pos, cam_lookat, focus_dist, aperture, ASPECT_RATIO, horizontal_fov)

    ground_mat = materials.NumpyPointOnHemiSphereCheckerboardMaterial(
        numpy.array([1.0, 1.0, 1.0]),
        numpy.array([0.0, 0.0, 0.0]),
        numpy.array([0.5, 0.5, 0.5]),
        numpy.array([0.3, 0.3, 0.3]),
    )
    red_blue_mat = materials.NumpyPointOnHemiSphereCheckerboardMaterial(
        numpy.array([2.0, 2.0, 2.0]),
        numpy.array([0.2, 0.2, 0.2]),
        numpy.array([0.7, 0.3, 0.2]),
        numpy.array([0.1, 0.2, 0.5]),
    )
    metal_mat = materials.NumpyMetalMaterial(numpy.array([0.8, 0.8, 0.8]), 0.0)
    glass_mat = materials.NumpyDielectricMaterial(1.5)
    normal_mat = materials.NumpyNormalToRGBMaterial()

    material_map = {
        0: ground_mat,
        1: red_blue_mat,
        2: metal_mat,
        3: glass_mat,
        4: normal_mat
    }

    # Sphere setup
    sphere_group = SphereGroup()

    # Ground
    sphere_group.add_sphere(
        numpy.array([0.0, -1000.0, 0.0], dtype=numpy.single),
        1000.0,
        numpy.array([0,0,0], dtype=numpy.single),
        0
    )

    obj_mesh = OBJTriMesh()
    obj_mesh.read("bunny.obj")

    smallest_y = min([vertex[1] for vertex in obj_mesh.vertices])

    spacing = 2.0

    # Metal bunny
    metal_grp = TriangleGroup(2)
    offset_x = -spacing
    offset_z = -spacing
    for triangle in obj_mesh.faces:
        metal_grp.add_triangle(
            numpy.array(
                [
                    obj_mesh.vertices[triangle[0][0]][0] + offset_x,
                    obj_mesh.vertices[triangle[0][0]][1] - smallest_y,
                    obj_mesh.vertices[triangle[0][0]][2] + offset_z,
                ],
                dtype=numpy.single
            ),
            numpy.array(
                [
                    obj_mesh.vertices[triangle[1][0]][0] + offset_x,
                    obj_mesh.vertices[triangle[1][0]][1] - smallest_y,
                    obj_mesh.vertices[triangle[1][0]][2] + offset_z,
                ],
                    dtype=numpy.single
            ),
            numpy.array(
                [
                    obj_mesh.vertices[triangle[2][0]][0] + offset_x,
                    obj_mesh.vertices[triangle[2][0]][1] - smallest_y,
                    obj_mesh.vertices[triangle[2][0]][2] + offset_z,
                ],
                dtype=numpy.single
            )
        )

    # Glass bunny
    glass_grp = TriangleGroup(3)
    offset_x = spacing
    offset_z = spacing
    for triangle in obj_mesh.faces:
        glass_grp.add_triangle(
            numpy.array(
                [
                    obj_mesh.vertices[triangle[0][0]][0] + offset_x,
                    obj_mesh.vertices[triangle[0][0]][1] - smallest_y,
                    obj_mesh.vertices[triangle[0][0]][2] + offset_z,
                ],
                dtype=numpy.single
            ),
            numpy.array(
                [
                    obj_mesh.vertices[triangle[1][0]][0] + offset_x,
                    obj_mesh.vertices[triangle[1][0]][1] - smallest_y,
                    obj_mesh.vertices[triangle[1][0]][2] + offset_z,
                ],
                dtype=numpy.single
            ),
            numpy.array(
                [
                    obj_mesh.vertices[triangle[2][0]][0] + offset_x,
                    obj_mesh.vertices[triangle[2][0]][1] - smallest_y,
                    obj_mesh.vertices[triangle[2][0]][2] + offset_z,
                ],
                dtype=numpy.single
            )
        )

    # Red/blue bunny
    red_blue_grp = TriangleGroup(1)
    offset_x = -spacing
    offset_z = spacing
    for triangle in obj_mesh.faces:
        red_blue_grp.add_triangle(
            numpy.array(
                [
                    obj_mesh.vertices[triangle[0][0]][0] + offset_x,
                    obj_mesh.vertices[triangle[0][0]][1] - smallest_y,
                    obj_mesh.vertices[triangle[0][0]][2] + offset_z,
                ],
                dtype=numpy.single
            ),
            numpy.array(
                [
                    obj_mesh.vertices[triangle[1][0]][0] + offset_x,
                    obj_mesh.vertices[triangle[1][0]][1] - smallest_y,
                    obj_mesh.vertices[triangle[1][0]][2] + offset_z,
                ],
                dtype=numpy.single
            ),
            numpy.array(
                [
                    obj_mesh.vertices[triangle[2][0]][0] + offset_x,
                    obj_mesh.vertices[triangle[2][0]][1] - smallest_y,
                    obj_mesh.vertices[triangle[2][0]][2] + offset_z,
                ],
                dtype=numpy.single
            )
        )


    # Normal bunny
    normal_grp = TriangleGroup(4)
    offset_x = spacing
    offset_z = -spacing
    for triangle in obj_mesh.faces:
        normal_grp.add_triangle(
            numpy.array(
                [
                    obj_mesh.vertices[triangle[0][0]][0] + offset_x,
                    obj_mesh.vertices[triangle[0][0]][1] - smallest_y,
                    obj_mesh.vertices[triangle[0][0]][2] + offset_z,
                ],
                dtype=numpy.single
            ),
            numpy.array(
                [
                    obj_mesh.vertices[triangle[1][0]][0] + offset_x,
                    obj_mesh.vertices[triangle[1][0]][1] - smallest_y,
                    obj_mesh.vertices[triangle[1][0]][2] + offset_z,
                ],
                dtype=numpy.single
            ),
            numpy.array(
                [
                    obj_mesh.vertices[triangle[2][0]][0] + offset_x,
                    obj_mesh.vertices[triangle[2][0]][1] - smallest_y,
                    obj_mesh.vertices[triangle[2][0]][2] + offset_z,
                ],
                dtype=numpy.single
            )
        )


    return camera, [sphere_group, metal_grp, glass_grp, red_blue_grp, normal_grp], material_map


def numpy_cow_scene():
    cam_pos = numpy.array([11.0, 8.0, 9.0])
    cam_lookat = numpy.array([1.0, 3.0, -1.0])
    # cam_pos = numpy.array([5.0, 5.0, 5.0])
    # cam_lookat = numpy.array([0.0, 0.5, 0.0])
    focus_dist = 10
    aperture = 0.0
    horizontal_fov = 60.0
    camera = Camera(cam_pos, cam_lookat, focus_dist, aperture, ASPECT_RATIO, horizontal_fov)

    ground_mat = materials.NumpyPointOnHemiSphereCheckerboardMaterial(
        numpy.array([1.0, 1.0, 1.0]),
        numpy.array([0.0, 0.0, 0.0]),
        numpy.array([0.2, 0.7, 0.3]),
        numpy.array([0.1, 0.9, 0.2]),
    )
    black_white_mat = materials.NumpyPointOnHemiSphereCheckerboardMaterial(
        numpy.array([1.0, 1.0, 1.0]),
        numpy.array([0.2, 0.2, 0.2]),
        numpy.array([0.8, 0.8, 0.85]),
        numpy.array([0.25, 0.25, 0.2]),
    )

    material_map = {
        0: ground_mat,
        1: black_white_mat,
    }

    # Sphere setup
    sphere_group = SphereGroup()

    # Ground
    sphere_group.add_sphere(
        numpy.array([0.0, -1000.0, 0.0], dtype=numpy.single),
        1000.0,
        numpy.array([0,0,0], dtype=numpy.single),
        0
    )

    obj_mesh = OBJTriMesh()
    obj_mesh.read("cow.obj")

    smallest_y = min([vertex[1] for vertex in obj_mesh.vertices])

    spacing = 2.0

    # Metal bunny
    cow_grp = TriangleGroup(1)
    for triangle in obj_mesh.faces:
        cow_grp.add_triangle(
            numpy.array(
                [
                    obj_mesh.vertices[triangle[0][0]][0],
                    obj_mesh.vertices[triangle[0][0]][1] - smallest_y,
                    obj_mesh.vertices[triangle[0][0]][2],
                ],
                dtype=numpy.single
            ),
            numpy.array(
                [
                    obj_mesh.vertices[triangle[1][0]][0],
                    obj_mesh.vertices[triangle[1][0]][1] - smallest_y,
                    obj_mesh.vertices[triangle[1][0]][2],
                ],
                    dtype=numpy.single
            ),
            numpy.array(
                [
                    obj_mesh.vertices[triangle[2][0]][0],
                    obj_mesh.vertices[triangle[2][0]][1] - smallest_y,
                    obj_mesh.vertices[triangle[2][0]][2],
                ],
                dtype=numpy.single
            )
        )


    return camera, [sphere_group, cow_grp], material_map


def numpy_one_weekend_demo_scene():
    cam_pos = numpy.array([13.0, 2.0, 3.0])
    cam_lookat = numpy.array([0.0, 0.5, 0.0])
    focus_dist = 10.0
    #aperture = 0.1
    aperture = 0.0
    camera = Camera(cam_pos, cam_lookat, focus_dist, aperture, ASPECT_RATIO, 30.0)

    gray_diffuse_mat = materials.NumpyPointOnHemiSphereMaterial(
        numpy.array([0.5, 0.5, 0.5], dtype=numpy.single)
    )
    metal_mat = materials.NumpyMetalMaterial(
        numpy.array([0.9, 0.9, 0.9], dtype=numpy.single),
        0.0
    )
    glass_mat = materials.NumpyDielectricMaterial(1.5)
    discrete_rgb_mat = materials.NumpyNormalToDiscreteRGBMaterial()

    material_map = {
        0: gray_diffuse_mat,
        1: metal_mat,
        2: glass_mat,
        3: discrete_rgb_mat
    }

    # Sphere setup
    sphere_group = SphereGroup()

    # Discrete
    sphere_group.add_sphere(
        numpy.array([-4, 1, 0], dtype=numpy.single),
        1.0,
        numpy.array([1,0,0], dtype=numpy.single),
        3
    )

    # Glass
    sphere_group.add_sphere(
        numpy.array([0, 1, 0], dtype=numpy.single),
        1.0,
        numpy.array([0,1,0], dtype=numpy.single),
        2
    )

    # Metal
    sphere_group.add_sphere(
        numpy.array([4, 1, 0], dtype=numpy.single),
        1.0,
        numpy.array([0.9,0.9,0.9], dtype=numpy.single),
        1
    )

    # Ground
    sphere_group.add_sphere(
        numpy.array([0, -1000, 0], dtype=numpy.single),
        1000.0,
        numpy.array([0.5, 0.5, 0.5], dtype=numpy.single),
        0
    )

    # with open("sphere_data.json") as file_handle:
    #     sphere_data = json.load(file_handle)

    # for sphere in sphere_data:
    #     # sphere_x = sphere["pos"][0]
    #     # sphere_z = sphere["pos"][2]
    #     # if -3.5 < sphere_z < 3.5:
    #     #     continue
    #     sphere_group.add_sphere(
    #          numpy.array(sphere["pos"],  dtype=numpy.single),
    #          sphere["radius"],
    #          numpy.array([0,0,0], dtype=numpy.single),
    #          0
    #     )


    # print(len(sphere_data))
    # for sphere in sphere_data:
    #     colour = numpy.array([0.5, 0.5, 0.5], dtype=numpy.single)
    #     if sphere["material"] != "glass":
    #         colour = numpy.array(sphere["colour"])
    #     sphere_group.add_sphere(sphere["pos"], sphere["radius"], colour, 0)


    return camera, [sphere_group], material_map


def numpy_glass_experiment_scene():
    cam_pos = numpy.array([0,0,0], dtype=numpy.single)
    cam_lookat = numpy.array([0.0, 0.0, -5.0], dtype=numpy.single)
    focus_dist = 10.0
    aperture = 0.0
    camera = Camera(cam_pos, cam_lookat, focus_dist, aperture, ASPECT_RATIO, 90.0)


    ground_mat = materials.NumpyPointOnHemiSphereMaterial(
        numpy.array([(148/256), (116/256), (105/256)], dtype=numpy.single)
    )
    blue_mat = materials.NumpyPointOnHemiSphereMaterial(
        numpy.array([0.1, 0.2, 0.5], dtype=numpy.single)
    )
    discrete_normal_mat = materials.NumpyNormalToDiscreteRGBMaterial()
    metal_mat = materials.NumpyMetalMaterial(
        numpy.array([0.8, 0.8, 0.8], dtype=numpy.single),
        0.0
    )
    glass_mat = materials.NumpyDielectricMaterial(1.5)

    material_map = {
        0: ground_mat,
        1: blue_mat,
        2: discrete_normal_mat,
        3: metal_mat,
        4: glass_mat
    }

    sphere_group = SphereGroup()

    # Line of shperes left to right
    # Glass
    sphere_group.add_sphere(
        numpy.array([-6.0, 0.0, -10.0], dtype=numpy.single),
        3.0,
        numpy.array([0.5, 0.5, 0.5], dtype=numpy.single),
        4
    )
    # Blue
    sphere_group.add_sphere(
        numpy.array([0.0, 0.0, -10.0], dtype=numpy.single),
        3.0,
        numpy.array([0.5, 0.5, 0.5], dtype=numpy.single),
        1
    )
    # Discrete Normal
    sphere_group.add_sphere(
        numpy.array([6.0, 0.0, -10.0], dtype=numpy.single),
        3.0,
        numpy.array([0.5, 0.5, 0.5], dtype=numpy.single),
        2
    )

    # Floating sphere above and to the right of the left/right line.
    sphere_group.add_sphere(
        numpy.array([5.0, 6.0, -16.0], dtype=numpy.single),
        3.0,
        numpy.array([0.5, 0.5, 0.5], dtype=numpy.single),
        3
    )

    # Sphere embedded in the ground behind the glass sphere
    sphere_group.add_sphere(
        numpy.array([-9.0, -3.0, -16.0], dtype=numpy.single),
        3.0,
        numpy.array([0.5, 0.5, 0.5], dtype=numpy.single),
        2
    )


    for x in range(3):
        for y in range(3):
            sphere_group.add_sphere(
                numpy.array([(x*1.3)-12.0, (y*2.0)+1.5, -16.0], dtype=numpy.single),
                0.3,
                numpy.array([0.5, 0.5, 0.5], dtype=numpy.single),
                2
            )

    # Ground Sphere
    sphere_group.add_sphere(
        numpy.array([0.0, -503.0, -10.0], dtype=numpy.single),
        500,
        numpy.array([0.5, 0.5, 0.5], dtype=numpy.single),
        0
    )

    return camera, [sphere_group], material_map


def numpy_triangles_scene():
    cam_pos = numpy.array([0.0, 1.0, 6.0])
    cam_lookat = numpy.array([0.0, 0.5, 0.0])
    # cam_pos = numpy.array([5.0, 5.0, 5.0])
    # cam_lookat = numpy.array([0.0, 0.5, 0.0])
    focus_dist = 10
    aperture = 0.0
    horizontal_fov = 50.0
    camera = Camera(cam_pos, cam_lookat, focus_dist, aperture, ASPECT_RATIO, horizontal_fov)


    ground_mat = materials.NumpyPointOnHemiSphereMaterial(
        numpy.array([0.5, 0.5, 0.5], dtype=numpy.single)
    )

    material_map = {
        0: ground_mat,
    }

    tri_grp = TriangleGroup(0)

    # Blue triangle
    tri_grp.add_triangle(
        numpy.array([1.0, 0.0, 0.0], dtype=numpy.single),
        numpy.array([2.0, 0.0, 0.0], dtype=numpy.single),
        numpy.array([1.0, 2.0, 0.0], dtype=numpy.single),
    )

    # Green triangle
    tri_grp.add_triangle(
        numpy.array([-2.5, 0.0, 0.0], dtype=numpy.single),
        numpy.array([-1.5, 0.0, 0.0], dtype=numpy.single),
        numpy.array([-2.0, 0.75, 0.0], dtype=numpy.single),
    )

    # Mirror triangle
    offset = numpy.array([0.0, 0.0, -2.0], dtype=numpy.single)
    tri_grp.add_triangle(
        numpy.array([-2.0, 0.0, -1.0], dtype=numpy.single) + offset,
        numpy.array([2.0, 0.0, 1.0], dtype=numpy.single) + offset,
        numpy.array([0.0, 2.0, 0.0], dtype=numpy.single) + offset,
    )

    # Ground triangle
    tri_grp.add_triangle(
        numpy.array([-200, 0, 200], dtype=numpy.single),
        numpy.array([200, 0, 200], dtype=numpy.single),
        numpy.array([0, 0, -200], dtype=numpy.single),
    )

    return camera, [tri_grp], material_map


def numpy_simple_sphere_scene():
    cam_pos = numpy.array([10.0, 5.0, 10.0])
    cam_lookat = numpy.array([0.0, 0.5, 0.0])
    # cam_pos = numpy.array([5.0, 5.0, 5.0])
    # cam_lookat = numpy.array([0.0, 0.5, 0.0])
    focus_dist = 10
    aperture = 0.0
    horizontal_fov = 50.0
    camera = Camera(cam_pos, cam_lookat, focus_dist, aperture, ASPECT_RATIO, horizontal_fov)


    ground_mat = materials.NumpyPointOnHemiSphereMaterial(
        numpy.array([0.5, 0.5, 0.5], dtype=numpy.single)
    )
    checker_mat = materials.NumpyPointOnHemiSphereCheckerboardMaterial(
        numpy.array([2, 2, 2], dtype=numpy.single),
        numpy.array([0, 0, 0], dtype=numpy.single),
        numpy.array([0.5, 0.8, 0.5], dtype=numpy.single),
        numpy.array([0.9, 0.5, 0.5], dtype=numpy.single)
    )

    material_map = {
        0: ground_mat,
        1: checker_mat,
    }

    # Sphere setup
    sphere_group = SphereGroup()

    # Ground
    sphere_group.add_sphere(
        numpy.array([0.0, -1000.0, 0.0], dtype=numpy.single),
        1000.0,
        numpy.array([1,0,0], dtype=numpy.single),
        0
    )

    # Grey
    sphere_group.add_sphere(
        numpy.array([-1.0, 0.5, 0.0], dtype=numpy.single),
        0.5,
        numpy.array([1,0,0], dtype=numpy.single),
        0
    )

    # Checker
    sphere_group.add_sphere(
        numpy.array([2, 2, 0], dtype=numpy.single),
        2.0,
        numpy.array([1,0,0], dtype=numpy.single),
        1
    )

    return camera, [sphere_group], material_map


def ray_group_triangle_group_bunny_scene():
    cam_pos = numpy.array([-2.0, 3.5, 8.0])
    cam_lookat = numpy.array([-2.2, 1.7, 0.0])
    # cam_pos = numpy.array([5.0, 5.0, 5.0])
    # cam_lookat = numpy.array([0.0, 0.5, 0.0])
    focus_dist = 10
    aperture = 0.0
    horizontal_fov = 53.0
    camera = Camera(cam_pos, cam_lookat, focus_dist, aperture, ASPECT_RATIO, horizontal_fov)

    # ground_mat = materials.NumpyPointOnHemiSphereMaterial(
    #     numpy.array([0.5, 0.5, 0.5], dtype=numpy.single)
    # )

    ground_mat = materials.NumpyPointOnHemiSphereCheckerboardMaterial(
        numpy.array([2.0, 2.0, 2.0]),
        numpy.array([0.0, 0.0, 0.0]),
        numpy.array([0.5, 0.5, 0.5]),
        numpy.array([0.8, 0.8, 0.8]),
    )

    bunny_mat = materials.NumpyPointOnHemiSphereMaterial(
        numpy.array([0.2, 0.7, 0.1], dtype=numpy.single)
    )
    tex_mat = materials.NumpyPointOnHemiSphereTextureMaterial(
        "bunnyTexture.tif"
        # "horizontalLineStrengthMap.jpg"
        # "flagTexture.tif"
    )
    metal_mat = materials.NumpyMetalMaterial(
        numpy.array([0.8, 0.8, 0.8], dtype=numpy.single),
        0.0
    )

    material_map = {
        0: ground_mat,
        1: tex_mat,
        2: metal_mat,
    }

    tri_grp = TriangleGroup(1)

    # Ground triangle
    # tri_grp.add_triangle(
    #     numpy.array([-200, 0, 200], dtype=numpy.single),
    #     numpy.array([200, 0, 200], dtype=numpy.single),
    #     numpy.array([0, 0, -200], dtype=numpy.single),
    # )

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
            uv0=numpy.array(obj_mesh.uvs[triangle[0][1]]),
            uv1=numpy.array(obj_mesh.uvs[triangle[1][1]]),
            uv2=numpy.array(obj_mesh.uvs[triangle[2][1]]),
            normal0=obj_mesh.get_smooth_vertex_normal(triangle[0][0]),
            normal1=obj_mesh.get_smooth_vertex_normal(triangle[1][0]),
            normal2=obj_mesh.get_smooth_vertex_normal(triangle[2][0])
        )

    # Sphere setup
    sphere_group = SphereGroup()

    # Ground
    sphere_group.add_sphere(
        numpy.array([0.0, -1000.0, 0.0], dtype=numpy.single),
        1000.0,
        numpy.array([1,0,0], dtype=numpy.single),
        0
    )

    # Sphere
    # sphere_group.add_sphere(
    #     numpy.array([-4.4, 2.0, -1.5], dtype=numpy.single),
    #     1.7,
    #     numpy.array([1,0,0], dtype=numpy.single),
    #     2
    # )

    return camera, [tri_grp, sphere_group], material_map


def texture_test_scene():
    cam_pos = numpy.array([2.5, 2.5, 2.5])
    cam_lookat = numpy.array([0.5, 0.0, 0.5])
    # cam_pos = numpy.array([5.0, 5.0, 5.0])
    # cam_lookat = numpy.array([0.0, 0.5, 0.0])
    focus_dist = 10
    aperture = 0.0
    horizontal_fov = 50.0
    camera = Camera(cam_pos, cam_lookat, focus_dist, aperture, ASPECT_RATIO, horizontal_fov)

    ground_mat = materials.NumpyPointOnHemiSphereMaterial(
        numpy.array([0.5, 0.5, 0.5], dtype=numpy.single)
    )
    # Test texture is from https://ue4techarts.com/2017/04/22/how-to-iterate-textures-in-an-atlas-ue4/
    tex_mat = materials.NumpyPointOnHemiSphereTextureMaterial(
        # "bunnyTexture.tif"
        # "horizontalLineStrengthMap.jpg"
        # "flagTexture.tif"
        "uv_test.jpg"
    )

    material_map = {
        0: ground_mat,
        1: tex_mat,
    }

    tri_grp = TriangleGroup(1)

    obj_mesh = OBJTriMesh()
    obj_mesh.read("square.obj")

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
            uv0=numpy.array(obj_mesh.uvs[triangle[0][1]]),
            uv1=numpy.array(obj_mesh.uvs[triangle[1][1]]),
            uv2=numpy.array(obj_mesh.uvs[triangle[2][1]])
        )

    # tri_grp.add_triangle(
    #     numpy.array([-1, -1, -3], dtype=numpy.single),
    #     numpy.array([1, -1, -3], dtype=numpy.single),
    #     numpy.array([0, 1, -3], dtype=numpy.single),
    #     uv0=numpy.array([0, 0], dtype=numpy.single),
    #     uv1=numpy.array([1, 0], dtype=numpy.single),
    #     uv2=numpy.array([0.5, 1], dtype=numpy.single),
    #     # uv1=numpy.array([0, 0], dtype=numpy.single),
    #     # uv2=numpy.array([1, 0], dtype=numpy.single),
    #     # uv0=numpy.array([0, 1], dtype=numpy.single)
    #     # uv2=numpy.array([0, 0], dtype=numpy.single),
    #     # uv0=numpy.array([1, 0], dtype=numpy.single),
    #     # uv1=numpy.array([0, 1], dtype=numpy.single)
    # )

    # Sphere setup
    sphere_group = SphereGroup()

    # Ground
    sphere_group.add_sphere(
        numpy.array([0.0, -1000.0, 0.0], dtype=numpy.single),
        1000.0,
        numpy.array([1,0,0], dtype=numpy.single),
        0
    )

    return camera, [tri_grp, sphere_group], material_map


def smooth_normal_test_scene():
    cam_pos = numpy.array([2.5, 2.5, 2.5])
    cam_lookat = numpy.array([0.5, 0.5, 0.5])
    # cam_pos = numpy.array([5.0, 5.0, 5.0])
    # cam_lookat = numpy.array([0.0, 0.5, 0.0])
    focus_dist = 10
    aperture = 0.0
    horizontal_fov = 50.0
    camera = Camera(cam_pos, cam_lookat, focus_dist, aperture, ASPECT_RATIO, horizontal_fov)

    ground_mat = materials.NumpyPointOnHemiSphereCheckerboardMaterial(
        numpy.array([4.0, 4.0, 4.0]),
        numpy.array([0.0, 0.0, 0.0]),
        numpy.array([0.5, 0.5, 0.5]),
        numpy.array([0.8, 0.8, 0.8]),
    )

    # normal_mat = materials.NumpyNormalToRGBMaterial()

    metal_mat = materials.NumpyMetalMaterial(
        numpy.array([0.9, 0.9, 0.9], dtype=numpy.single),
        0.0
    )

    material_map = {
        0: ground_mat,
        1: metal_mat,
    }

    tri_grp = TriangleGroup(1)

    obj_mesh = OBJTriMesh()
    obj_mesh.read("angled_tris_standing.obj")

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
            uv0=numpy.array(obj_mesh.uvs[triangle[0][1]]),
            uv1=numpy.array(obj_mesh.uvs[triangle[1][1]]),
            uv2=numpy.array(obj_mesh.uvs[triangle[2][1]]),
            normal0=obj_mesh.get_smooth_vertex_normal(triangle[0][0]),
            normal1=obj_mesh.get_smooth_vertex_normal(triangle[1][0]),
            normal2=obj_mesh.get_smooth_vertex_normal(triangle[2][0])
        )


    # Sphere setup
    sphere_group = SphereGroup()

    # Ground
    sphere_group.add_sphere(
        numpy.array([0.0, -1000.0, 0.0], dtype=numpy.single),
        1000.0,
        numpy.array([1,0,0], dtype=numpy.single),
        0
    )

    return camera, [tri_grp, sphere_group], material_map


def blender_cylinder_vert_normals_test_scene():
    cam_pos = numpy.array([1.5, 1.5, 1.5])
    cam_lookat = numpy.array([-0.25, 0.0, -0.25])
    # cam_pos = numpy.array([5.0, 5.0, 5.0])
    # cam_lookat = numpy.array([0.0, 0.5, 0.0])
    focus_dist = 10
    aperture = 0.0
    horizontal_fov = 35.0
    camera = Camera(cam_pos, cam_lookat, focus_dist, aperture, ASPECT_RATIO, horizontal_fov)

    ground_mat = materials.NumpyPointOnHemiSphereMaterial(
        numpy.array([0.5, 0.5, 0.5], dtype=numpy.single)
    )

    checker_mat = materials.NumpyPointOnHemiSphereCheckerboardMaterial(
        numpy.array([4.0, 4.0, 4.0]),
        numpy.array([0.0, 0.0, 0.0]),
        numpy.array([1.0, 0.3, 0.3]),
        numpy.array([0.2, 1.0, 0.3]),
    )

    normal_mat = materials.NumpyNormalToRGBMaterial()

    metal_mat = materials.NumpyMetalMaterial(
        numpy.array([0.9, 0.9, 0.9], dtype=numpy.single),
        0.0
    )

    material_map = {
        0: ground_mat,
        1: checker_mat,
        2: metal_mat,
        3: normal_mat
    }

    tri_grp = TriangleGroup(2)

    obj_mesh = OBJTriMesh()
    obj_mesh.read("cylinder_smooth.obj")

    for triangle in obj_mesh.faces:
        tri_grp.add_triangle(
            numpy.array(obj_mesh.vertices[triangle[0][0]]),
            numpy.array(obj_mesh.vertices[triangle[1][0]]),
            numpy.array(obj_mesh.vertices[triangle[2][0]]),
            uv0=numpy.array(obj_mesh.uvs[triangle[0][1]]),
            uv1=numpy.array(obj_mesh.uvs[triangle[1][1]]),
            uv2=numpy.array(obj_mesh.uvs[triangle[2][1]]),
            normal0=numpy.array(obj_mesh.vertex_normals[triangle[0][2]]),
            normal1=numpy.array(obj_mesh.vertex_normals[triangle[1][2]]),
            normal2=numpy.array(obj_mesh.vertex_normals[triangle[2][2]])
        )

    tri_grp2 = TriangleGroup(2)

    obj_mesh2 = OBJTriMesh()
    obj_mesh2.read("cylinder_faceted.obj")

    for triangle in obj_mesh2.faces:
        tri_grp2.add_triangle(
            numpy.array(obj_mesh2.vertices[triangle[0][0]]),
            numpy.array(obj_mesh2.vertices[triangle[1][0]]),
            numpy.array(obj_mesh2.vertices[triangle[2][0]]),
            uv0=numpy.array(obj_mesh2.uvs[triangle[0][1]]),
            uv1=numpy.array(obj_mesh2.uvs[triangle[1][1]]),
            uv2=numpy.array(obj_mesh2.uvs[triangle[2][1]]),
            normal0=numpy.array(obj_mesh2.vertex_normals[triangle[0][2]]),
            normal1=numpy.array(obj_mesh2.vertex_normals[triangle[1][2]]),
            normal2=numpy.array(obj_mesh2.vertex_normals[triangle[2][2]])
        )



    # Sphere setup
    sphere_group = SphereGroup()

    # Ground
    sphere_group.add_sphere(
        numpy.array([0.0, -1000.0, 0.0], dtype=numpy.single),
        1000.0,
        numpy.array([1,0,0], dtype=numpy.single),
        0
    )

    # return camera, [tri_grp, tri_grp2, sphere_group], material_map
    return camera, [sphere_group], material_map


def sphere_types_test_scene():
    cam_pos = numpy.array([0, 3, 7])
    cam_lookat = numpy.array([0.0, 1, 0.0])
    # cam_pos = numpy.array([5.0, 5.0, 5.0])
    # cam_lookat = numpy.array([0.0, 0.5, 0.0])
    focus_dist = 10
    aperture = 0.0
    horizontal_fov = 50.0
    camera = Camera(cam_pos, cam_lookat, focus_dist, aperture, ASPECT_RATIO, horizontal_fov)

    ground_mat = materials.NumpyPointOnHemiSphereMaterial(
        numpy.array([0.5, 0.5, 0.5], dtype=numpy.single)
    )

    checker_mat = materials.NumpyPointOnHemiSphereCheckerboardMaterial(
        numpy.array([4.0, 4.0, 4.0]),
        numpy.array([0.0, 0.0, 0.0]),
        numpy.array([1.0, 0.3, 0.3]),
        numpy.array([0.2, 1.0, 0.3]),
    )

    normal_mat = materials.NumpyNormalToRGBMaterial()

    metal_mat = materials.NumpyMetalMaterial(
        numpy.array([0.9, 0.9, 0.9], dtype=numpy.single),
        0.0
    )

    material_map = {
        0: ground_mat,
        1: checker_mat,
        2: metal_mat,
        3: normal_mat
    }

    smooth_icosphere = TriangleGroup(2)

    obj_mesh = OBJTriMesh()
    obj_mesh.read("smooth-icosphere.obj")

    for triangle in obj_mesh.faces:
        smooth_icosphere.add_triangle(
            numpy.array(obj_mesh.vertices[triangle[0][0]]),
            numpy.array(obj_mesh.vertices[triangle[1][0]]),
            numpy.array(obj_mesh.vertices[triangle[2][0]]),
            uv0=numpy.array(obj_mesh.uvs[triangle[0][1]]),
            uv1=numpy.array(obj_mesh.uvs[triangle[1][1]]),
            uv2=numpy.array(obj_mesh.uvs[triangle[2][1]]),
            normal0=numpy.array(obj_mesh.vertex_normals[triangle[0][2]]),
            normal1=numpy.array(obj_mesh.vertex_normals[triangle[1][2]]),
            normal2=numpy.array(obj_mesh.vertex_normals[triangle[2][2]])
        )

    faceted_icosphere = TriangleGroup(2)

    obj_mesh2 = OBJTriMesh()
    obj_mesh2.read("faceted-icosphere.obj")

    for triangle in obj_mesh2.faces:
        faceted_icosphere.add_triangle(
            numpy.array(obj_mesh2.vertices[triangle[0][0]]),
            numpy.array(obj_mesh2.vertices[triangle[1][0]]),
            numpy.array(obj_mesh2.vertices[triangle[2][0]]),
            uv0=numpy.array(obj_mesh2.uvs[triangle[0][1]]),
            uv1=numpy.array(obj_mesh2.uvs[triangle[1][1]]),
            uv2=numpy.array(obj_mesh2.uvs[triangle[2][1]]),
            normal0=numpy.array(obj_mesh2.vertex_normals[triangle[0][2]]),
            normal1=numpy.array(obj_mesh2.vertex_normals[triangle[1][2]]),
            normal2=numpy.array(obj_mesh2.vertex_normals[triangle[2][2]])
        )



    # Sphere setup
    sphere_group = SphereGroup()

    # Ground
    sphere_group.add_sphere(
        numpy.array([0.0, -1000.0, 0.0], dtype=numpy.single),
        1000.0,
        numpy.array([1,0,0], dtype=numpy.single),
        1
    )

    # Comparison sphere
    sphere_group.add_sphere(
        numpy.array([0, 1.0, 0], dtype=numpy.single),
        1.0,
        numpy.array([1,0,0], dtype=numpy.single),
        2
    )

    # return camera, [tri_grp, tri_grp2, sphere_group], material_map
    return camera, [smooth_icosphere, faceted_icosphere, sphere_group], material_map


def disk_test_scene():

    cam_pos = numpy.array([3.0, 3.0, 3.0])
    cam_lookat = numpy.array([0.0, 0.0, 0.0])
    focus_dist = 10
    aperture = 0.0
    horizontal_fov = 40.0
    camera = Camera(cam_pos, cam_lookat, focus_dist, aperture, ASPECT_RATIO, horizontal_fov)

    ground_mat = materials.NumpyPointOnHemiSphereMaterial(
        numpy.array([0.5, 0.5, 0.5], dtype=numpy.single)
    )
    green_mat = materials.NumpyPointOnHemiSphereMaterial(
        numpy.array([0.3, 0.8, 0.1], dtype=numpy.single)
    )
    red_mat = materials.NumpyPointOnHemiSphereMaterial(
        numpy.array([0.8, 0.15, 0.2], dtype=numpy.single)
    )
    # Test texture is from https://ue4techarts.com/2017/04/22/how-to-iterate-textures-in-an-atlas-ue4/
    tex_mat = materials.NumpyPointOnHemiSphereTextureMaterial(
        "uv_test.jpg"
    )

    material_map = {
        0: ground_mat,
        1: green_mat,
        2: red_mat,
        3: tex_mat
    }

    # Sphere setup
    sphere_group = SphereGroup()

    # Ground
    sphere_group.add_sphere(
        numpy.array([0.0, -1000.0, 0.0], dtype=numpy.single),
        1000.0,
        numpy.array([1,0,0], dtype=numpy.single),
        0
    )

    centre = numpy.array([0,0.5,0], dtype=numpy.single)
    radius = 0.5
    normal = numpy.array([1,0,0], dtype=numpy.single)
    normal = normal / numpy.linalg.norm(normal)
    normal = normal.astype(numpy.single)
    material_index = 3
    disk1 = Disk(centre, radius, normal, material_index)

    centre = numpy.array([0,0.15,-1.5], dtype=numpy.single)
    radius = 1.5
    normal = numpy.array([0,1,0], dtype=numpy.single)
    normal = normal / numpy.linalg.norm(normal)
    normal = normal.astype(numpy.single)
    material_index = 3
    up = numpy.array([0,0,-1], dtype=numpy.single)
    disk2 = Disk(centre, radius, normal, material_index, up=up)

    return camera, [sphere_group, disk1, disk2], material_map


def main():
    print("Start render")
    img_data = render()
    print("End render")
    generate_image_from_data(img_data)


if __name__ == "__main__":
    main()
