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

from . import scenes

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

    # camera, object_groups, material_map = scenes.numpy_dielectric_scene(ASPECT_RATIO)
    # camera, object_groups, material_map = scenes.numpy_glass_experiment_scene(ASPECT_RATIO)
    # camera, object_groups, material_map = scenes.numpy_triangles_scene(ASPECT_RATIO)
    # camera, object_groups, material_map = scenes.numpy_simple_sphere_scene(ASPECT_RATIO)
    # camera, object_groups, material_map = scenes.numpy_one_weekend_demo_scene(ASPECT_RATIO)
    camera, object_groups, material_map = scenes.ray_group_triangle_group_bunny_scene(ASPECT_RATIO)
    # camera, object_groups, material_map = scenes.blender_cylinder_vert_normals_test_scene(ASPECT_RATIO)
    # camera, object_groups, material_map = scenes.sphere_types_test_scene(ASPECT_RATIO)
    # camera, object_groups, material_map = scenes.texture_test_scene(ASPECT_RATIO)
    # camera, object_groups, material_map = scenes.numpy_bunnies_scene(ASPECT_RATIO)
    # camera, object_groups, material_map = scenes.numpy_cow_scene(ASPECT_RATIO)
    # camera, object_groups, material_map = scenes.disk_test_scene(ASPECT_RATIO)
    # camera, object_groups, material_map = scenes.smooth_normal_test_scene(ASPECT_RATIO)

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


def main():
    print("Start render")
    img_data = render()
    print("End render")
    generate_image_from_data(img_data)


if __name__ == "__main__":
    main()
