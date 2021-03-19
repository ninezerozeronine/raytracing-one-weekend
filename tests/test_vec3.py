import pytest

from raytracing_one_weekend.vec3 import Vec3


def vecs_are_equal(first, second):
    tolerance = 0.0001
    return (
        abs(first.x - second.x) < tolerance
        and abs(first.y - second.y) < tolerance
        and abs(first.z - second.z) < tolerance
    )


@pytest.mark.parametrize("a, b, expected", [
    (
        Vec3(1, 2, 3),
        Vec3(4, 5, 6),
        Vec3(5, 7, 9),
    ),
    (
        Vec3(0.1, 0.2, 0.3),
        Vec3(0.4, 0.5, 0.6),
        Vec3(0.5, 0.7, 0.9),
    ),
    (
        Vec3(1, 2, 3),
        Vec3(0.4, 0.5, 0.6),
        Vec3(1.4, 2.5, 3.6),
    ),
])
def test_add(a, b, expected):
    c = a + b
    assert vecs_are_equal(c, expected)


@pytest.mark.parametrize("a, b, expected", [
    (
        Vec3(1, 2, 3),
        Vec3(4, 5, 6),
        Vec3(5, 7, 9),
    ),
    (
        Vec3(0.1, 0.2, 0.3),
        Vec3(0.4, 0.5, 0.6),
        Vec3(0.5, 0.7, 0.9),
    ),
    (
        Vec3(1, 2, 3),
        Vec3(0.4, 0.5, 0.6),
        Vec3(1.4, 2.5, 3.6),
    ),
])
def test_iadd(a, b, expected):
    a += b
    assert vecs_are_equal(a, expected)
