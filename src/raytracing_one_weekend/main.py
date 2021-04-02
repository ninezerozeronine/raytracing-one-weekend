"""
Main entry point for renderer functionality
"""

from PIL import Image, ImageDraw

from .vec3 import Vec3
from .ray import Ray

IMG_HEIGHT = 90
IMG_WIDTH = 160
HORIZON_COLOUR = Vec3(1, 1, 1)
SKY_COLOUR = Vec3(0.5, 0.7, 1)


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
            draw.point(
                (column, row),
                fill=(
                    int(255 * img_data[(column, flipped_row)].r),
                    int(255 * img_data[(column, flipped_row)].g),
                    int(255 * img_data[(column, flipped_row)].b),
                )
            )

    image.show()
    image.save("tmp_image.png")


def render():
    """
    Do the rendering of the image.
    """
    viewport_width = 2.0
    viewport_height = viewport_width * (IMG_HEIGHT/IMG_WIDTH)
    viewport_vertical = Vec3(0, viewport_height, 0)
    viewport_horizontal = Vec3(viewport_width, 0, 0)
    focalplane_dist = 1.0

    camera_pos = Vec3(0, 0, 0)
    bottomleft_focalplane_pos = (
        # Start at camera position
        camera_pos

        # The camera is looking in -Z, this way X is to the right and Y
        # is up like a typical X/Y graph.
        # Move out to the focal plane in -Z, this puts us in the centre
        # of the focal plane
        + Vec3(0, 0, (focalplane_dist * -1))

        # Move to the bottom of the focalplane
        + viewport_vertical * -0.5

        # Move to the left of the focalplane
        + viewport_horizontal * -0.5
    )

    img_data = {}

    for y_index in range(IMG_HEIGHT):
        y_progress = y_index/IMG_HEIGHT
        for x_index in range(IMG_WIDTH):
            x_progress = x_index/IMG_WIDTH
            pt_on_viewport = (
                bottomleft_focalplane_pos
                + viewport_vertical * y_progress
                + viewport_horizontal * x_progress
            )
            ray_direction = pt_on_viewport - camera_pos
            pixel_ray = Ray(camera_pos, ray_direction)
            colour = get_ray_colour(pixel_ray)
            img_data[(x_index, y_index)] = colour

    return img_data


def get_ray_colour(ray):
    """
    Given a ray, get the colour from the scene
    """

    if ray_sphere_intersection(ray, Vec3(0, 0, -2), 0.5):
        return Vec3(1, 0, 0)

    normalised_ray = ray.direction.normalised()

    # Y component will now be somewhere between -1 and 1. Map it into
    # a 0 to 1 range.
    t = 0.5 * (normalised_ray.y + 1)

    # Lerp between white and blue based on mapped Y
    return (1.0 - t) * HORIZON_COLOUR + t * SKY_COLOUR


def ray_sphere_intersection(ray, centre, radius):
    """
    Check if a ray intersects a sphere.

    A point is on a sphere if the length of the vector from the centre
    of the sphere to a point on the sphere is equal to the radius of
    the sphere.

    If you take the dot product of a vector with itself, this is the
    length squared.

    If R is the vector from the center to a point on the sphere, and r
    is the radius then if R.R = r^2 the point is on the sphere.

    R is the vector from the centre (C) to a point (P) this means
    R = P - C (i.e. centre to origin, then origin to point).

    So if (P - C).(P - C) = r^2, the point is on the sphere.

    We want to find if a point on a ray intersects the sphere. We can
    plug in the point on ray representation (Pt on Ray = O + tD, or
    origin + multiplier of the direction) to the equation above:

    (O + tD - C).(O + tD - C) = r^2

    Exapnd this out and we get:

    O.O + t(D.O) - C.O

    + t(D.O) + t^2(D.D) - t(C.D)

    - C.O - t(C.D) + C.C = r^2

    Collecting terms we get:

    O.O + 2t(D.O) - 2(C.O) + t^2(D.D) - 2t(C.D) + C.C = r^2

    Note that (O - C).(O - C) = O.O - 2(C.O) + C.C so we can simplify
    to:

    2t(D.O) + t^2(D.D) -2t(C.D) + (O - C).(O - C) = r^2

    Collapsing the 2t factors and re-arranging a bit:

    t^2(D.D) + 2tD.(O - C) + (O - C).(O - C) = r^2

    Re arrange to equal zero and we have a quadratic in terms of t!

    t^2(D.D) + 2tD.(O - C) + (O - C).(O - C) - r^2 = 0

    Where:

    A = D.D
    B = 2D.(O - C)
    C = (O - C).(O - C) - r^2

    (O - C) is the vector from the centre of the sphere to the ray
    origin - C to O

    Using our old friend the quadratic equation::

        x = (-B +/- sqrt(B^2 - 4AC)) / (2A)

    We know that if B^2 - 4AC is less than 0 the equation has no
    roots - or - the ray doesn't intersect the sphere!
    """

    C_to_O = ray.origin - centre
    A = ray.direction.dot(ray.direction)
    B = 2 * ray.direction.dot(C_to_O)
    C = C_to_O.dot(C_to_O) - radius**2
    discriminant = B**2 - (4*A*C)
    return discriminant > 0


def main():
    print("Start render")
    img_data = render()
    print("End render")
    generate_image_from_data(img_data)


if __name__ == "__main__":
    main()
