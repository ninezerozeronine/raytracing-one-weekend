import pytest
from math import sqrt

from raytracing_one_weekend.vec3 import Vec3

TOLERANCE = 0.0001


def vecs_are_equal(first, second):
    return (
        abs(first.x - second.x) < TOLERANCE
        and abs(first.y - second.y) < TOLERANCE
        and abs(first.z - second.z) < TOLERANCE
    )


def test_bad_value():
    with pytest.raises(TypeError):
        a = Vec3("foo", 2, 3)


def test_xyz_property_gets():
    vec = Vec3(1, 2, 3)
    assert vec.x == 1 and vec.y == 2 and vec.z == 3


def test_xyz_property_sets():
    vec = Vec3(1, 2, 3)
    vec.x = 4
    vec.y = 5
    vec.z = 6
    assert vec.x == 4 and vec.y == 5 and vec.z == 6


def test_rgb_property_gets():
    vec = Vec3(1, 2, 3)
    assert vec.r == 1 and vec.g == 2 and vec.b == 3


def test_rgb_property_sets():
    vec = Vec3(1, 2, 3)
    vec.r = 4
    vec.g = 5
    vec.b = 6
    assert vec.r == 4 and vec.g == 5 and vec.b == 6


def test_index_get():
    vec = Vec3(1, 2, 3)
    assert vec[0] == 1 and vec[1] == 2 and vec[2] == 3


def test_index_set():
    vec = Vec3(1, 2, 3)
    vec[0] = 4
    vec[1] = 5
    vec[2] = 6
    assert vec[0] == 4 and vec[1] == 5 and vec[2] == 6


@pytest.mark.parametrize("test_input", [
    "hello",
    [1, 2, 3],
    {"a": 1},
])
def test_bad_key_type(test_input):
    with pytest.raises(TypeError):
        vec = Vec3(1, 2, 3)
        data = vec[test_input]


@pytest.mark.parametrize("test_input", [
    -1,
    5
])
def test_bad_key_value(test_input):
    with pytest.raises(IndexError):
        vec = Vec3(1, 2, 3)
        data = vec[test_input]


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


@pytest.mark.parametrize("a, b", [
    (
        Vec3(1, 2, 3),
        3,
    ),
    (
        Vec3(1, 2, 3),
        "hello",
    ),
])
def test_add_raises(a, b):
    with pytest.raises(TypeError):
        res = a + b


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


@pytest.mark.parametrize("a, b", [
    (
        Vec3(1, 2, 3),
        3,
    ),
    (
        Vec3(1, 2, 3),
        "hello",
    ),
])
def test_iadd_raises(a, b):
    with pytest.raises(TypeError):
        a += b


def test_iadd_identity():
    a = Vec3(1, 2, 3)
    b = Vec3(1, 2, 3)
    orig = id(a)
    a += b
    assert orig == id(a)


@pytest.mark.parametrize("a, b, expected", [
    (
        Vec3(5, 7, 9),
        Vec3(1, 2, 3),
        Vec3(4, 5, 6),
    ),
    (
        Vec3(0.5, 0.7, 0.9),
        Vec3(0.4, 0.5, 0.6),
        Vec3(0.1, 0.2, 0.3),
    ),
    (
        Vec3(1.4, 2.5, 3.6),
        Vec3(1, 2, 3),
        Vec3(0.4, 0.5, 0.6),
    ),
])
def test_sub(a, b, expected):
    c = a - b
    assert vecs_are_equal(c, expected)


@pytest.mark.parametrize("a, b", [
    (
        Vec3(1, 2, 3),
        3,
    ),
    (
        Vec3(1, 2, 3),
        "hello",
    ),
])
def test_sub_raises(a, b):
    with pytest.raises(TypeError):
        res = a - b


@pytest.mark.parametrize("a, b, expected", [
    (
        Vec3(5, 7, 9),
        Vec3(1, 2, 3),
        Vec3(4, 5, 6),
    ),
    (
        Vec3(0.5, 0.7, 0.9),
        Vec3(0.4, 0.5, 0.6),
        Vec3(0.1, 0.2, 0.3),
    ),
    (
        Vec3(1.4, 2.5, 3.6),
        Vec3(1, 2, 3),
        Vec3(0.4, 0.5, 0.6),
    ),
])
def test_isub(a, b, expected):
    a -= b
    assert vecs_are_equal(a, expected)


@pytest.mark.parametrize("a, b", [
    (
        Vec3(1, 2, 3),
        3,
    ),
    (
        Vec3(1, 2, 3),
        "hello",
    ),
])
def test_isub_raises(a, b):
    with pytest.raises(TypeError):
        a -= b


def test_isub_identity():
    a = Vec3(1, 2, 3)
    b = Vec3(1, 2, 3)
    orig = id(a)
    a -= b
    assert orig == id(a)


@pytest.mark.parametrize("a, b, expected", [
    (
        Vec3(5, 7, 9),
        2,
        Vec3(10, 14, 18),
    ),
    (
        Vec3(0.5, 0.7, 0.9),
        0.5,
        Vec3(0.25, 0.35, 0.45),
    ),
    (
        Vec3(1.4, 2.5, 3.6),
        2,
        Vec3(2.8, 5, 7.2),
    ),
])
def test_mul(a, b, expected):
    c = a * b
    assert vecs_are_equal(c, expected)


@pytest.mark.parametrize("a, b, expected", [
    (
        2,
        Vec3(5, 7, 9),
        Vec3(10, 14, 18),
    ),
    (
        0.5,
        Vec3(0.5, 0.7, 0.9),
        Vec3(0.25, 0.35, 0.45),
    ),
    (
        2,
        Vec3(1.4, 2.5, 3.6),
        Vec3(2.8, 5, 7.2),
    ),
])
def test_rmul(a, b, expected):
    c = a * b
    assert vecs_are_equal(c, expected)


@pytest.mark.parametrize("a, b", [
    (
        Vec3(1, 2, 3),
        Vec3(4, 5, 6),
    ),
    (
        Vec3(1, 2, 3),
        "hello",
    ),
])
def test_mul_raises(a, b):
    with pytest.raises(TypeError):
        res = a * b


@pytest.mark.parametrize("a, b, expected", [
    (
        Vec3(5, 7, 9),
        2,
        Vec3(10, 14, 18),
    ),
    (
        Vec3(0.5, 0.7, 0.9),
        0.5,
        Vec3(0.25, 0.35, 0.45),
    ),
    (
        Vec3(1.4, 2.5, 3.6),
        2,
        Vec3(2.8, 5, 7.2),
    ),
])
def test_imul(a, b, expected):
    a *= b
    assert vecs_are_equal(a, expected)


@pytest.mark.parametrize("a, b", [
    (
        Vec3(1, 2, 3),
        Vec3(3, 4, 5),
    ),
    (
        Vec3(1, 2, 3),
        "hello",
    ),
])
def test_imul_raises(a, b):
    with pytest.raises(TypeError):
        a *= b


def test_imul_identity():
    a = Vec3(1, 2, 3)
    b = 6
    orig = id(a)
    a *= b
    assert orig == id(a)


@pytest.mark.parametrize("a, b, expected", [
    (
        Vec3(5, 7, 9),
        2,
        Vec3(2.5, 3.5, 4.5),
    ),
    (
        Vec3(0.5, 0.7, 0.9),
        0.5,
        Vec3(1, 1.4, 1.8),
    ),
    (
        Vec3(1.4, 2.5, 3.6),
        2,
        Vec3(0.7, 1.25, 1.8),
    ),
])
def test_truediv(a, b, expected):
    c = a / b
    assert vecs_are_equal(c, expected)


@pytest.mark.parametrize("a, b", [
    (
        Vec3(1, 2, 3),
        Vec3(4, 5, 6),
    ),
    (
        Vec3(1, 2, 3),
        "hello",
    ),
])
def test_truediv_raises(a, b):
    with pytest.raises(TypeError):
        res = a / b


@pytest.mark.parametrize("a, b, expected", [
    (
        Vec3(5, 7, 9),
        2,
        Vec3(2.5, 3.5, 4.5),
    ),
    (
        Vec3(0.5, 0.7, 0.9),
        0.5,
        Vec3(1, 1.4, 1.8),
    ),
    (
        Vec3(1.4, 2.5, 3.6),
        2,
        Vec3(0.7, 1.25, 1.8),
    ),
])
def test_itruediv(a, b, expected):
    a /= b
    assert vecs_are_equal(a, expected)


@pytest.mark.parametrize("a, b", [
    (
        Vec3(1, 2, 3),
        Vec3(3, 4, 5),
    ),
    (
        Vec3(1, 2, 3),
        "hello",
    ),
])
def test_itruediv_raises(a, b):
    with pytest.raises(TypeError):
        a /= b


def test_itruediv_identity():
    a = Vec3(1, 2, 3)
    b = 6
    orig = id(a)
    a /= b
    assert orig == id(a)


@pytest.mark.parametrize("a, b, expected", [
    (
        Vec3(1, 2, 3),
        Vec3(1, 2, 3),
        True
    ),
    (
        Vec3(0.001, 0.002, 0.003),
        Vec3(0.001, 0.002, 0.003),
        True
    ),
    (
        Vec3(1, 2, 3),
        Vec3(4, 2, 3),
        False
    ),
])
def test_eq(a, b, expected):
    equal = a == b
    assert equal == expected


def test_eq_raises():
    with pytest.raises(TypeError):
        a = Vec3(1, 2, 3)
        same = a == "foo"


@pytest.mark.parametrize("x, y, z, length", [
    (0, 3, 4, 5),
    (0.1, 0.2, 0.3, 0.3741657387),
])
def test_length(x, y, z, length):
    vec = Vec3(x, y, z)
    assert abs(vec.length() - length) < TOLERANCE


@pytest.mark.parametrize("x, y, z, length_squared", [
    (0, 3, 4, 25),
    (0.1, 0.2, 0.3, 0.14),
])
def test_length_squared(x, y, z, length_squared):
    vec = Vec3(x, y, z)
    assert abs(vec.length_squared() - length_squared) < TOLERANCE


@pytest.mark.parametrize("a, b, result", [
    (
        Vec3(4, 8, 10),
        Vec3(9, 2, 7),
        122
    ),
    (
        Vec3(1, 2, 3),
        Vec3(1, 5, 7),
        32
    ),
])
def test_dot(a, b, result):
    assert a.dot(b) == result


def test_dot_raises():
    with pytest.raises(TypeError):
        a = Vec3(1, 3, 4)
        a.dot("foo")


@pytest.mark.parametrize("a, b, result", [
    (
        Vec3(1, 0, 0),
        Vec3(0, 1, 0),
        Vec3(0, 0, 1)
    ),
    (
        Vec3(1, 2, 3),
        Vec3(1, 5, 7),
        Vec3(-1, -4, 3)
    ),
])
def test_cross(a, b, result):
    assert vecs_are_equal(a.cross(b), result)


def test_cross_raises():
    with pytest.raises(TypeError):
        a = Vec3(1, 3, 4)
        a.cross("foo")


@pytest.mark.parametrize("data, result", [
    (
        Vec3(5, 0, 0),
        Vec3(1, 0, 0),
    ),
    (
        Vec3(2, -4, 1),
        Vec3(2/sqrt(21), -4/sqrt(21), 1/sqrt(21)),
    ),
])
def test_normalise(data, result):
    data.normalise()
    assert vecs_are_equal(data, result)


@pytest.mark.parametrize("data, result", [
    (
        Vec3(5, 0, 0),
        Vec3(1, 0, 0),
    ),
    (
        Vec3(2, -4, 1),
        Vec3(2/sqrt(21), -4/sqrt(21), 1/sqrt(21)),
    ),
])
def test_normalised(data, result):
    assert vecs_are_equal(data.normalised(), result)