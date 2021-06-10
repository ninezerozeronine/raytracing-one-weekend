"""
Main entry point for renderer functionality
"""
import math
from random import random
import time
import datetime

from PIL import Image, ImageDraw
import numpy
import humanize

from .ray import Ray
from .renderable import World
from .sphere import Sphere
from .camera import Camera
from . import materials

IMG_WIDTH = 160
IMG_HEIGHT = 90
ASPECT_RATIO = IMG_WIDTH/IMG_HEIGHT
PIXEL_SAMPLES = 50
HORIZON_COLOUR = numpy.array([1.0, 1.0, 1.0])
SKY_COLOUR = numpy.array([0.5, 0.7, 1.0])
RNG = numpy.random.default_rng()
MAX_DEPTH = 50


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

    world, camera = dof_cam_scene()

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

    next_percentage = 1
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
            print(f"{next_percentage}% complete. Avg pixel:{human_pixel_time}. Est. remaining: {human_remain_time}")
            next_percentage += 1

    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:0.4f} seconds.")

    return img_data


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


def normal_to_rgb(normal):
    """
    Convert a normal to an rgb colour.

    Expects unit length normal.
    """

    return numpy.array([
            normal[0] + 1.0,
            normal[1] + 1.0,
            normal[2] + 1.0,
        ]) * 0.5


def main():
    print("Start render")
    img_data = render()
    print("End render")
    generate_image_from_data(img_data)


if __name__ == "__main__":
    main()
