import pytest

from math import sqrt

from raytracing_one_weekend import main
from raytracing_one_weekend.ray import Ray
from raytracing_one_weekend.vec3 import Vec3

root2over2 = sqrt(2)/2
root2 = sqrt(2)

@pytest.mark.parametrize("ray, centre, radius, expected", [
    # Note that length of ray dir doesn't really matter as direction is
    # normalised upon ray creation
    (
        # Ray: Origin at origin, direction in x with len 1
        # Sphere at 3, 0, 0, radius 1
        Ray(Vec3(0, 0, 0), Vec3(1, 0, 0)),
        Vec3(3, 0, 0),
        1,
        2,
    ),
    (
        # Ray: Origin at origin, direction at 45 deg to top right(along x=y) with len 1
        # Sphere at 3, 3, 0, radius 1
        Ray(Vec3(0, 0, 0), Vec3(1, 1, 0)),
        Vec3(3, 3, 0),
        1,
        3*root2 - 1,
    ),
    (
        # Ray: Origin at origin, direction at 45 deg to bottom left (along x=y) with len 1
        # Sphere at -3, -3, 0, radius 1
        Ray(Vec3(0, 0, 0), Vec3(-1, -1, 0)),
        Vec3(-3, -3, 0),
        1,
        3*root2 - 1,
    ),
    (
        # Ray: Origin at (0, 0, 5), direction at 45 deg to +ve area (along x=y)
        # Sphere at 3, 3, 5, radius 1
        Ray(Vec3(0, 0, 5), Vec3(1, 1, 0)),
        Vec3(3, 3, 5),
        1,
        3*root2 - 1,
    ),
    (
        # Ray: Origin at (0, 0, 0), direction at 45 deg to +ve area (along x=y=z)
        # Sphere at 3, 3, 3, radius 1
        Ray(Vec3(0, 0, 0), Vec3(1, 1, 1)),
        Vec3(3, 3, 3),
        1,
        3*sqrt(3) - 1,
    ),
    (
        # Ray: Origin at (-1, 2, 3), direction at 45 deg to +ve area (along x=y=z)
        # Sphere at 3, 3, 3, radius 1
        Ray(Vec3(-1, 2, 3), Vec3(1, 1, 1)),
        Vec3(2, 5, 6),
        1,
        3*sqrt(3) - 1,
    ),
    (
        # Ray: Origin at origin, direction = Vec3(1, 2, 0)
        # Sphere at 3, 5, 0, radius root2
        Ray(Vec3(0, 0, 0), Vec3(1, 2, 0)),
        Vec3(3, 5, 0),
        sqrt(2),
        2 * sqrt(5),
    ),
    (
        # Ray: Origin at origin, direction = Vec3(1, 2, 0)
        # Sphere at 4, 6, 0, radius 2 root2
        Ray(Vec3(0, 0, 0), Vec3(1, 2, 0)),
        Vec3(4, 6, 0),
        2 * sqrt(2),
        2 * sqrt(5),
    ),
])
def test_ray_sphere_intersection(ray, centre, radius, expected):
    t = main.ray_sphere_intersection(ray, centre, radius)
    assert(abs(t - expected)) < 0.001


@pytest.mark.parametrize("vecA, vecB", [
    (Vec3(1, 2, 3), Vec3(3, 2, 1)),
    (Vec3(1, -1, 3), Vec3(0, 2, 2)),
    (Vec3(1, 5, 3), Vec3(-2, -2, 1)),
    (Vec3(1, 0.5, 56), Vec3(3.4, 2, 20)),
])
def test_dot_prod_expand(vecA, vecB):
    a_minus_b = vecA - vecB
    method_1 = a_minus_b.dot(a_minus_b)

    a_dot_a = vecA.dot(vecA)
    two_a_dot_b = 2 * vecA.dot(vecB)
    b_dot_b = vecB.dot(vecB)
    method_2 = a_dot_a - two_a_dot_b + b_dot_b

    assert(abs(method_1 - method_2) < 0.0001)

@pytest.mark.parametrize("vecA, vecB, vecC", [
    (Vec3(1, 2, 3), Vec3(3, 2, 1), Vec3(4, 6, 1)),
    (Vec3(1, -1, 3), Vec3(0, 2, 2), Vec3(4, -1, 21)),
    (Vec3(1, 5, 3), Vec3(-2, -2, 1), Vec3(6, 1, 0.3)),
    (Vec3(1, 0.5, 56), Vec3(3.4, 2, 20), Vec3(3, 7, 9)),
])
def test_dot_prod_factor_out(vecA, vecB, vecC):
    a_dot_b = vecA.dot(vecB)
    a_dot_c = vecA.dot(vecC)
    method_1 = a_dot_b - a_dot_c

    b_minus_c = vecB - vecC
    method_2 = vecA.dot(b_minus_c)

    assert(abs(method_1 - method_2) < 0.0001)
