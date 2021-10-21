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
from .sphere import Sphere
from .sphere_group import SphereGroup
from .sphere_group_ray_group import SphereGroupRayGroup
from .mttriangle import MTTriangle
from .mttriangle_group import MTTriangleGroup
from .mttriangle_group_ray_group import MTTriangleGroupRayGroup
from .obj_tri_mesh import OBJTriMesh
from .camera import Camera
from . import materials


IMG_WIDTH = 160 * 4
IMG_HEIGHT = 90 * 4
ASPECT_RATIO = IMG_WIDTH/IMG_HEIGHT
PIXEL_SAMPLES = 40
MAX_BOUNCES = 5
HORIZON_COLOUR = numpy.array([1.0, 1.0, 1.0], dtype=numpy.single)
SKY_COLOUR = numpy.array([0.5, 0.7, 1.0], dtype=numpy.single)
RNG = numpy.random.default_rng()
MAX_DEPTH = 4


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
    """
    Do the rendering of the image.
    """

    # world, camera = dielectric_debug_scene()
    # world, camera = non_numpy_triangle_noise_cmp_scene()
    world, camera = bunny_scene()

    img_data = {}
    pixel_coords = (
        (x, y) for y in range(IMG_HEIGHT) for x in range(IMG_WIDTH)
    )
    # pixel_coords = (
    #     (50, 45),
    #     (80, 75),
    #     (110, 45),
    #     (140, 45),
    #     (80, 15),
    #     (80, 45),
    # )

    start_time = time.perf_counter()

    next_percentage = 0.1
    total_rendertime_us = 0
    min_pixeltime_us = 99999999999999
    max_pixeltime_us = -1
    total_pixels = IMG_HEIGHT * IMG_WIDTH

    for pixel_num, coords in enumerate(pixel_coords, 1):
        x_coord, y_coord = coords
        start = time.perf_counter_ns()

        total_colour = numpy.array([0.0, 0.0, 0.0])
        for _ in range(PIXEL_SAMPLES):
            x_progress = (x_coord + random()) / IMG_WIDTH
            y_progress = (y_coord + random()) / IMG_HEIGHT
            ray = camera.get_ray(x_progress, y_progress)
            total_colour += get_ray_colour(ray, world, MAX_DEPTH)
            # The squareroot is for a 2.0 gamma correction
        img_data[(x_coord, y_coord)] = numpy.sqrt(total_colour/PIXEL_SAMPLES)

        end = time.perf_counter_ns()
        pixel_time_us = (end - start) // 1000

        total_rendertime_us += pixel_time_us
        if pixel_time_us > max_pixeltime_us:
            max_pixeltime_ns = pixel_time_us
        if pixel_time_us < min_pixeltime_us:
            min_pixeltime_us = pixel_time_us

        if pixel_num / total_pixels * 100 > next_percentage:
            avg_pixel_time_us = (total_rendertime_us / pixel_num)
            est_remaining_us = (avg_pixel_time_us * (total_pixels - pixel_num))
            human_pixel_time = humanize.precisedelta(
                datetime.timedelta(microseconds=avg_pixel_time_us),
                minimum_unit="milliseconds"
            )
            human_remain_time = humanize.precisedelta(
                datetime.timedelta(microseconds=est_remaining_us)
            )
            print(f"{next_percentage:.1f}% complete. Avg pixel:{human_pixel_time}. Est. remaining: {human_remain_time}")
            next_percentage += 0.1

    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:0.4f} seconds.")

    return img_data


def numpy_render():

    # Cam setup
    cam_pos = numpy.array([5.0, 2.0, 10.0])
    cam_lookat = numpy.array([0.0, 1.0, 0.0])
    pos_to_lookat = cam_lookat - cam_pos
    focus_dist = numpy.sqrt(pos_to_lookat.dot(pos_to_lookat))
    aperture = 0.5
    horizontal_fov = 60.0
    camera = Camera(cam_pos, cam_lookat, focus_dist, aperture, ASPECT_RATIO, horizontal_fov)

    # Sphere setup
    sphere_ray_group = SphereGroupRayGroup()

    # Red
    sphere_ray_group.add_sphere(
        numpy.array([-5, 2, 0], dtype=numpy.single),
        2.0,
        numpy.array([1,0,0], dtype=numpy.single),
    )

    # Green
    sphere_ray_group.add_sphere(
        numpy.array([0, 2, 0], dtype=numpy.single),
        2.0,
        numpy.array([0,1,0], dtype=numpy.single),
    )

    # Blue
    sphere_ray_group.add_sphere(
        numpy.array([5, 2, 0], dtype=numpy.single),
        2.0,
        numpy.array([0,0,1], dtype=numpy.single),
    )

    # Ground
    sphere_ray_group.add_sphere(
        numpy.array([0, -1000, 0],  dtype=numpy.single),
        1000.0,
        numpy.array([0.5, 0.5, 0.5], dtype=numpy.single)
    )


    # Bunch of small spheres
    for x in range(-10, 10):
        for z in range(-10, 10):
            sphere_ray_group.add_sphere(
                numpy.array([x, 0.3, z], dtype=numpy.single),
                0.3,
                RNG.uniform(low=0.0, high=1.0, size=3)
            )



    # cam_pos = numpy.array([3.0, 3.0, 2.0])
    # cam_lookat = numpy.array([0.0, 0.0, -1.0])
    # pos_to_lookat = cam_lookat - cam_pos
    # focus_dist = numpy.sqrt(pos_to_lookat.dot(pos_to_lookat))
    # aperture = 2.0
    # camera = Camera(cam_pos, cam_lookat, focus_dist, aperture, ASPECT_RATIO, 45.0)


    # # # Sphere setup
    # sphere_ray_group = SphereGroupRayGroup()

    # # Red
    # sphere_ray_group.add_sphere(
    #     numpy.array([-1, 0, -1], dtype=numpy.single),
    #     0.5,
    #     numpy.array([1,0,0], dtype=numpy.single),
    # )

    # # Blue
    # sphere_ray_group.add_sphere(
    #     numpy.array([0, 0, -1], dtype=numpy.single),
    #     0.5,
    #     numpy.array([0,0,1], dtype=numpy.single),
    # )

    # # Yellowish
    # sphere_ray_group.add_sphere(
    #     numpy.array([1, 0, -1], dtype=numpy.single),
    #     0.5,
    #     numpy.array([0.8,0.6,0.2], dtype=numpy.single),
    # )

    # # Ground
    # sphere_ray_group.add_sphere(
    #     numpy.array([0, -1000.5, 0],  dtype=numpy.single),
    #     1000.0,
    #     numpy.array([0.5, 0.5, 0.5], dtype=numpy.single)
    # )


    print("Generating arrays")
    # ray_origins = numpy.zeros(
    #     (IMG_WIDTH, IMG_HEIGHT, PIXEL_SAMPLES, 3), dtype=numpy.single
    # )
    # ray_directions = numpy.zeros(
    #     (IMG_WIDTH, IMG_HEIGHT, PIXEL_SAMPLES, 3), dtype=numpy.single
    # )
    ray_colours = numpy.zeros(
        (IMG_WIDTH, IMG_HEIGHT, PIXEL_SAMPLES, 3), dtype=numpy.single
    )

    print("Filling ray arrays")
    # for y_coord in range(IMG_HEIGHT):
    #     print(f"{y_coord} of {IMG_HEIGHT - 1}")
    #     for x_coord in range(IMG_WIDTH):
    #         for sample in range(PIXEL_SAMPLES):
    #             x_progress = (x_coord + random()) / IMG_WIDTH
    #             y_progress = (y_coord + random()) / IMG_HEIGHT
    #             ray = camera.get_ray(x_progress, y_progress)
    #             ray_origins[x_coord, y_coord, sample] = ray.origin
    #             ray_directions[x_coord, y_coord, sample] = ray.direction

    ray_origins, ray_directions = camera.get_ray_components(IMG_WIDTH, IMG_HEIGHT, PIXEL_SAMPLES)

    ray_origins = ray_origins.reshape(-1, 3)
    ray_directions = ray_directions.reshape(-1, 3)
    ray_colours = ray_colours.reshape(-1, 3)


    print("Getting ray hits")
    # sphere_hit_indecies, sphere_hit_ts = sphere_ray_group.get_hits(
    #     ray_origins, ray_directions, 0.00001, 5000.0
    # )

    # Need to chunk otherwise it runs out of memory :(
    num_chunks = 20
    ray_origins_chunks = numpy.split(ray_origins, num_chunks)
    ray_directions_chunks = numpy.split(ray_directions, num_chunks)
    print(f"Chunk 1 of {num_chunks}")
    sphere_hit_indecies, sphere_hit_ts = sphere_ray_group.get_hits(
        ray_origins_chunks[0], ray_directions_chunks[0], 0.00001, 5000.0
    )
    for chunk_index in range(1,num_chunks):
        print(f"Chunk {chunk_index+1} of {num_chunks}")
        sphere_hit_indecies_chunk, sphere_hit_ts_chunk = sphere_ray_group.get_hits(
            ray_origins_chunks[chunk_index], ray_directions_chunks[chunk_index], 0.00001, 5000.0
        )
        sphere_hit_indecies = numpy.concatenate((sphere_hit_indecies, sphere_hit_indecies_chunk), axis=0)
        sphere_hit_ts = numpy.concatenate((sphere_hit_ts, sphere_hit_ts_chunk), axis=0)


    ray_hits = sphere_hit_indecies > -1
    ray_misses = sphere_hit_indecies < 0
    ray_colours[ray_hits] = sphere_ray_group.colours[sphere_hit_indecies[ray_hits]]

    # Lerp between white and blue based on mapped Y
    ts = (ray_directions[ray_misses, 1] + 1.0) * 0.5
    ray_colours[ray_misses] = (1.0 - ts)[..., numpy.newaxis] * HORIZON_COLOUR + ts[..., numpy.newaxis] * SKY_COLOUR

    ray_colours_stacked = ray_colours.reshape(IMG_WIDTH, IMG_HEIGHT, PIXEL_SAMPLES, 3)
    ray_cols_sample_avg = numpy.mean(ray_colours_stacked, axis=2)
    
    print("Filling pixel data")
    pixel_data = {}
    for y_coord in range(IMG_HEIGHT):
        for x_coord in range(IMG_WIDTH):
            pixel_data[(x_coord, y_coord)] = ray_cols_sample_avg[x_coord, y_coord]

    return pixel_data


def numpy_bounce_render():

    # camera, object_groups, material_map = numpy_dielectric_scene()
    # camera, object_groups, material_map = numpy_glass_experiment_scene()
    # camera, object_groups, material_map = numpy_triangles_scene()
    # camera, object_groups, material_map = numpy_simple_sphere_scene()
    # camera, object_groups, material_map = numpy_one_weekend_demo_scene()
    camera, object_groups, material_map = ray_group_triangle_group_bunny_scene()
    # camera, object_groups, material_map = texture_test_scene()
    # camera, object_groups, material_map = numpy_bunnies_scene()
    # camera, object_groups, material_map = numpy_cow_scene()

    start_time = time.perf_counter()

    print("Generating arrays")
    ray_colours = numpy.ones(
        (IMG_WIDTH, IMG_HEIGHT, PIXEL_SAMPLES, MAX_BOUNCES + 1, 3), dtype=numpy.single
    )

    print("Filling ray arrays")
    ray_origins, ray_directions = camera.get_ray_components(IMG_WIDTH, IMG_HEIGHT, PIXEL_SAMPLES)
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
                    5000.0
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
            pixel_data[(x_coord, y_coord)] = sqrt_ray_cols[x_coord, y_coord]

    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:0.4f} seconds.")

    return pixel_data


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
    sphere_ray_group = SphereGroupRayGroup()

    # Ground
    sphere_ray_group.add_sphere(
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
    metal_grp = MTTriangleGroupRayGroup(2)
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
    glass_grp = MTTriangleGroupRayGroup(3)
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
    red_blue_grp = MTTriangleGroupRayGroup(1)
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
    normal_grp = MTTriangleGroupRayGroup(4)
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


    return camera, [sphere_ray_group, metal_grp, glass_grp, red_blue_grp, normal_grp], material_map


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
    sphere_ray_group = SphereGroupRayGroup()

    # Ground
    sphere_ray_group.add_sphere(
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
    cow_grp = MTTriangleGroupRayGroup(1)
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


    return camera, [sphere_ray_group, cow_grp], material_map


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


def numpy_comparison_scene():
    cam_pos = numpy.array([5.0, 2.0, 10.0])
    cam_lookat = numpy.array([0.0, 1.0, 0.0])
    pos_to_lookat = cam_lookat - cam_pos
    focus_dist = numpy.sqrt(pos_to_lookat.dot(pos_to_lookat))
    aperture = 0.0
    horizontal_fov = 60.0
    camera = Camera(cam_pos, cam_lookat, focus_dist, aperture, ASPECT_RATIO, horizontal_fov)

    ground_mat = materials.PointOnHemiSphereMaterial(numpy.array([0.5, 0.5, 0.5]))
    red_mat = materials.PointOnHemiSphereMaterial(numpy.array([1.0, 0.0, 0.0]))
    green_mat = materials.PointOnHemiSphereMaterial(numpy.array([0.0, 1.0, 0.0]))
    blue_mat = materials.PointOnHemiSphereMaterial(numpy.array([0.0, 0.0, 1.0]))

    world = World()

    all_spheres = SphereGroup()
    # Ground
    all_spheres.add_sphere(numpy.array([0.0, -1000.0, 0.0]), 1000.0, ground_mat)

    # Spheres
    all_spheres.add_sphere(numpy.array([-5.0, 2.0, 0.0]), 2.0, red_mat)
    all_spheres.add_sphere(numpy.array([0.0, 2.0, 0.0]), 2.0, green_mat)
    all_spheres.add_sphere(numpy.array([5.0, 2.0, 0.0]), 2.0, blue_mat)

    # Bunch of small spheres
    for x in range(-10, 10):
        for z in range(-10, 10):
            mat = materials.PointOnHemiSphereMaterial(RNG.uniform(low=0.0, high=1.0, size=3))
            all_spheres.add_sphere(
                numpy.array([x, 0.3, z], dtype=numpy.single),
                0.3,
                mat
            )

    world.renderables.append(all_spheres)

    return world, camera


def dielectric_debug_scene():
    cam_pos = numpy.array([13.0, 2.0, 3.0])
    cam_lookat = numpy.array([0.0, 0.5, 0.0])
    focus_dist = 10.0
    aperture = 0.0
    camera = Camera(cam_pos, cam_lookat, focus_dist, aperture, ASPECT_RATIO, 30.0)

    ground_mat = materials.PointOnHemiSphereMaterial(numpy.array([0.5, 0.5, 0.5]))
    glass_mat = materials.DielectricMaterial(1.5)

    world = World()

    # Ground
    # world.renderables.append(Sphere(numpy.array([0.0, -1000.0, 0.0]), 1000.0, ground_mat))

    # # Brown, Glass, Metal
    # world.renderables.append(Sphere(numpy.array([-4.0, 1.0, 0.0]), 1.0, ground_mat))
    # world.renderables.append(Sphere(numpy.array([0.0, 1.0, 0.0]), 1.0, ground_mat))
    # world.renderables.append(Sphere(numpy.array([4.0, 1.0, 0.0]), 1.0, glass_mat))



    # Ground
    all_spheres = SphereGroup()
    all_spheres.add_sphere(numpy.array([0.0, -1000.0, 0.0]), 1000.0, ground_mat)

    # Brown, Glass, Metal
    all_spheres.add_sphere(numpy.array([-4.0, 1.0, 0.0]), 1.0, ground_mat)
    all_spheres.add_sphere(numpy.array([0.0, 1.0, 0.0]), 1.0, ground_mat)
    all_spheres.add_sphere(numpy.array([4.0, 1.0, 0.0]), 1.0, glass_mat)

    world.renderables.append(all_spheres)



    return world, camera


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
    sphere_ray_group = SphereGroupRayGroup()

    # Discrete
    sphere_ray_group.add_sphere(
        numpy.array([-4, 1, 0], dtype=numpy.single),
        1.0,
        numpy.array([1,0,0], dtype=numpy.single),
        3
    )

    # Glass
    sphere_ray_group.add_sphere(
        numpy.array([0, 1, 0], dtype=numpy.single),
        1.0,
        numpy.array([0,1,0], dtype=numpy.single),
        2
    )

    # Metal
    sphere_ray_group.add_sphere(
        numpy.array([4, 1, 0], dtype=numpy.single),
        1.0,
        numpy.array([0.9,0.9,0.9], dtype=numpy.single),
        1
    )

    # Ground
    sphere_ray_group.add_sphere(
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
    #     sphere_ray_group.add_sphere(
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
    #     sphere_ray_group.add_sphere(sphere["pos"], sphere["radius"], colour, 0)


    return camera, [sphere_ray_group], material_map


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

    sphere_ray_group = SphereGroupRayGroup()

    # Line of shperes left to right
    # Glass
    sphere_ray_group.add_sphere(
        numpy.array([-6.0, 0.0, -10.0], dtype=numpy.single),
        3.0,
        numpy.array([0.5, 0.5, 0.5], dtype=numpy.single),
        4
    )
    # Blue
    sphere_ray_group.add_sphere(
        numpy.array([0.0, 0.0, -10.0], dtype=numpy.single),
        3.0,
        numpy.array([0.5, 0.5, 0.5], dtype=numpy.single),
        1
    )
    # Discrete Normal
    sphere_ray_group.add_sphere(
        numpy.array([6.0, 0.0, -10.0], dtype=numpy.single),
        3.0,
        numpy.array([0.5, 0.5, 0.5], dtype=numpy.single),
        2
    )

    # Floating sphere above and to the right of the left/right line.
    sphere_ray_group.add_sphere(
        numpy.array([5.0, 6.0, -16.0], dtype=numpy.single),
        3.0,
        numpy.array([0.5, 0.5, 0.5], dtype=numpy.single),
        3
    )

    # Sphere embedded in the ground behind the glass sphere
    sphere_ray_group.add_sphere(
        numpy.array([-9.0, -3.0, -16.0], dtype=numpy.single),
        3.0,
        numpy.array([0.5, 0.5, 0.5], dtype=numpy.single),
        2
    )


    for x in range(3):
        for y in range(3):
            sphere_ray_group.add_sphere(
                numpy.array([(x*1.3)-12.0, (y*2.0)+1.5, -16.0], dtype=numpy.single),
                0.3,
                numpy.array([0.5, 0.5, 0.5], dtype=numpy.single),
                2
            )

    # Ground Sphere
    sphere_ray_group.add_sphere(
        numpy.array([0.0, -503.0, -10.0], dtype=numpy.single),
        500,
        numpy.array([0.5, 0.5, 0.5], dtype=numpy.single),
        0
    )

    return camera, [sphere_ray_group], material_map


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

    tri_grp = MTTriangleGroupRayGroup(0)

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
    sphere_ray_group = SphereGroupRayGroup()

    # Ground
    sphere_ray_group.add_sphere(
        numpy.array([0.0, -1000.0, 0.0], dtype=numpy.single),
        1000.0,
        numpy.array([1,0,0], dtype=numpy.single),
        0
    )

    # Grey
    sphere_ray_group.add_sphere(
        numpy.array([-1.0, 0.5, 0.0], dtype=numpy.single),
        0.5,
        numpy.array([1,0,0], dtype=numpy.single),
        0
    )

    # Checker
    sphere_ray_group.add_sphere(
        numpy.array([2, 2, 0], dtype=numpy.single),
        2.0,
        numpy.array([1,0,0], dtype=numpy.single),
        1
    )

    return camera, [sphere_ray_group], material_map


def non_numpy_triangle_noise_cmp_scene():
    cam_pos = numpy.array([0.0, 1.0, 6.0])
    cam_lookat = numpy.array([0.0, 0.5, 0.0])
    # cam_pos = numpy.array([5.0, 5.0, 5.0])
    # cam_lookat = numpy.array([0.0, 0.5, 0.0])
    focus_dist = 10
    aperture = 0.0
    horizontal_fov = 50.0
    camera = Camera(cam_pos, cam_lookat, focus_dist, aperture, ASPECT_RATIO, horizontal_fov)

    ground_mat = materials.PointOnHemiSphereMaterial(numpy.array([0.5, 0.5, 0.5]))

    world = World()

    # Blue triangle
    world.renderables.append(MTTriangle(
        numpy.array([1.0, 0.0, 0.0]),
        numpy.array([2.0, 0.0, 0.0]),
        numpy.array([1.0, 2.0, 0.0]),
        ground_mat
    ))

    # Green triangle
    world.renderables.append(MTTriangle(
        numpy.array([-2.5, 0.0, 0.0]),
        numpy.array([-1.5, 0.0, 0.0]),
        numpy.array([-2.0, 0.75, 0.0]),
        ground_mat
    ))

    # Mirror triangle
    offset = numpy.array([0.0, 0.0, -2.0])
    world.renderables.append(MTTriangle(
        numpy.array([-2.0, 0.0, -1.0]) + offset,
        numpy.array([2.0, 0.0, 1.0]) + offset,
        numpy.array([0.0, 2.0, 0.0]) + offset,
        ground_mat
    ))

    # Ground triangle
    world.renderables.append(MTTriangle(
        numpy.array([-200, 0, 200], dtype=numpy.single),
        numpy.array([200, 0, 200], dtype=numpy.single),
        numpy.array([0, 0, -200], dtype=numpy.single),
        ground_mat
    ))

    return world, camera


def ray_group_triangle_group_bunny_scene():
    cam_pos = numpy.array([-2.0, 3.5, 8.0])
    cam_lookat = numpy.array([-2.2, 1.7, 0.0])
    # cam_pos = numpy.array([5.0, 5.0, 5.0])
    # cam_lookat = numpy.array([0.0, 0.5, 0.0])
    focus_dist = 10
    aperture = 0.0
    horizontal_fov = 53.0
    camera = Camera(cam_pos, cam_lookat, focus_dist, aperture, ASPECT_RATIO, horizontal_fov)

    ground_mat = materials.NumpyPointOnHemiSphereMaterial(
        numpy.array([0.5, 0.5, 0.5], dtype=numpy.single)
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

    tri_grp = MTTriangleGroupRayGroup(1)

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
            uv2=numpy.array(obj_mesh.uvs[triangle[2][1]])
        )

    # Sphere setup
    sphere_ray_group = SphereGroupRayGroup()

    # Ground
    sphere_ray_group.add_sphere(
        numpy.array([0.0, -1000.0, 0.0], dtype=numpy.single),
        1000.0,
        numpy.array([1,0,0], dtype=numpy.single),
        0
    )

    # Sphere in bunny
    sphere_ray_group.add_sphere(
        numpy.array([-4.4, 2.0, -1.5], dtype=numpy.single),
        1.7,
        numpy.array([1,0,0], dtype=numpy.single),
        2
    )

    return camera, [tri_grp, sphere_ray_group], material_map


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

    tri_grp = MTTriangleGroupRayGroup(1)

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
    sphere_ray_group = SphereGroupRayGroup()

    # Ground
    sphere_ray_group.add_sphere(
        numpy.array([0.0, -1000.0, 0.0], dtype=numpy.single),
        1000.0,
        numpy.array([1,0,0], dtype=numpy.single),
        0
    )

    return camera, [tri_grp, sphere_ray_group], material_map


def get_ray_colour(ray, world, depth):
    """
    Given a ray, get the colour from the scene
    """

    if depth <= 0:
        return numpy.array([0.0, 0.0, 0.0])

    # Make t_min slightly larger than 0 to prune out hits where the ray
    # hasn't travelled any distance at all.
    # This _really_ speeds up the calculation compared to when it's set
    # to 0. Not sure why! :(
    hit, hit_record = world.hit(ray, 0.00001, 5000.0)
    if hit:
        absorbed, surface_colour, scattered_ray = hit_record.material.scatter(
            ray,
            hit_record
        )
        if not absorbed:
            return surface_colour * get_ray_colour(scattered_ray, world, depth - 1)
        else:
            return surface_colour

    else:
        # Y component is somewhere between -1 and 1. Map it into
        # a 0 to 1 range.
        t = 0.5 * (ray.direction[1] + 1.0)

        # Lerp between white and blue based on mapped Y
        return (1.0 - t) * HORIZON_COLOUR + t * SKY_COLOUR


def main():
    print("Start render")
    # img_data = render()
    img_data = numpy_bounce_render()
    print("End render")
    generate_image_from_data(img_data)


if __name__ == "__main__":
    main()
