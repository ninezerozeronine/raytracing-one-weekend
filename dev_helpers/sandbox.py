import sys
import os
import time
import datetime
import random
import pprint
import json
import timeit

import numpy
import humanize

RNG = numpy.random.default_rng()

sys.path.append(os.path.abspath("../src"))

from raytracing_one_weekend import main


def write_sphere_json():
    sphere_data = gen_many_spheres_data()
    with open("sphere_data.json", "w") as file_handle:
        json.dump(
            sphere_data, file_handle, sort_keys=True, indent=4,
            ensure_ascii=False
        )


def gen_many_spheres_data():
    big_sphere_positions = [
        (-4, 0),
        (0, 0),
        (4, 0)
    ]

    sphere_data = []
    for x in range(-10, 10):
        for z in range(-5, 5):
            sphere = {}
            x_pos = x + (random.random() * 0.7)
            z_pos = z + (random.random() * 0.7)

            # Reject spheres too close to the big ones.
            reject = False
            for big_sphere_pos in big_sphere_positions:
                big_x_min = big_sphere_pos[0] - 1.1
                big_x_max = big_sphere_pos[0] + 1.1
                big_z_min = big_sphere_pos[1] - 1.1
                big_z_max = big_sphere_pos[1] + 1.1
                if (big_x_min < x_pos < big_x_max) and (big_z_min < z_pos < big_z_max):
                    reject = True
                    break

            if reject:
                continue

            radius = random.uniform(0.17, 0.23)
            sphere["pos"] = [x_pos, radius, z_pos]
            sphere["radius"] = radius

            mat_choice = random.random()
            if mat_choice < 0.8:
                # Diffuse
                sphere["material"] = "diffuse"
                sphere["colour"] = [
                    random.random()*random.random(),
                    random.random()*random.random(),
                    random.random()*random.random()
                ]
            elif mat_choice < 0.95:
                # Metal
                sphere["material"] = "metal"
                sphere["colour"] = [
                    random.uniform(0.5, 1.0),
                    random.uniform(0.5, 1.0),
                    random.uniform(0.5, 1.0)
                ]
                sphere["fuzziness"] = random.uniform(0, 0.5)
            else:
                # Glass
                sphere["material"] = "glass"
                sphere["ior"] = 1.5

            sphere_data.append(sphere)

    return sphere_data


def numpy_speedup_test():
    a_vecs = numpy.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
        dtype=numpy.single
    )
    b_vecs = numpy.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
        dtype=numpy.single
    )

    print(a_vecs)
    print(a_vecs[1][2])
    # print(numpy.vdot(a_vecs, b_vecs))
    # print(numpy.dot(a_vecs, b_vecs))
    # print(numpy.dot(a_vecs[0], b_vecs[0]))

    # This calculates the dot products of all the vectors
    print(numpy.sum(a_vecs * b_vecs, axis=1))

    print(numpy.einsum("ij,ij->i", a_vecs, b_vecs))

    print(a_vecs ** 2)

    setup = """
import numpy
RNG = numpy.random.default_rng()
lots_a = RNG.random((50, 3))
lots_b = RNG.random((50, 3))
"""
    slow = """
dots = []
for i in range(50):
    dots.append(numpy.dot(lots_a[i], lots_b[i]))
"""

    fast = """
dots = numpy.sum(lots_a * lots_b, axis=1)
    """

    faster = """
dots = numpy.einsum("ij,ij->i", lots_a, lots_b)
"""

    print(timeit.timeit(slow, setup=setup, number=10000))
    print(timeit.timeit(fast, setup=setup, number=10000))
    print(timeit.timeit(faster, setup=setup, number=10000))


def numpy_dot_cross_speed_test():

    setup = """
import numpy
RNG = numpy.random.default_rng()
lots_a = RNG.random((50, 3))
lots_b = RNG.random((50, 3))
"""
    dot_func = """
dots = []
for i in range(50):
    dots.append(numpy.dot(lots_a[i], lots_b[i]))
"""

    dot_meth = """
dots = []
for i in range(50):
    dots.append(lots_a[i].dot(lots_b[i]))
    """

    cross_func = """
crosses = []
for i in range(50):
    crosses.append(numpy.cross(lots_a[i], lots_b[i]))
"""

    cross_meth = """
crosses = []
for i in range(50):
    crosses.append(lots_a[i].cross(lots_b[i]))
    """

    print(timeit.timeit(dot_func, setup=setup, number=10000))
    print(timeit.timeit(dot_meth, setup=setup, number=10000))
    # print(timeit.timeit(cross_func, setup=setup, number=1000))
    # print(timeit.timeit(cross_meth, setup=setup, number=1000))


def numpy_vectorise_tests():
    t_min = 0.0001
    t_max = 5000
    origin = numpy.array([0, 0, 0], dtype=numpy.single)
    direction = numpy.array([0, 0, 1], dtype=numpy.single)
    centres = numpy.array(
        [
            [0, 0, 4],
            [2, 0.1, 4],
            [0, 0, -4],
            [-5, 0, 4]
        ],
        dtype=numpy.single
    )
    radii = numpy.array([1, 3, 1, 1], dtype=numpy.single)

    C_to_Os = origin - centres
    print("C to Os")
    print(C_to_Os)

    # ij,i-i> wasn't giving correct results
    Hs = numpy.einsum("ij,j->i", C_to_Os, direction)
    print("Hs")
    print(Hs)

    Cs = numpy.einsum("ij,ij->i", C_to_Os, C_to_Os) - numpy.square(radii)
    print("Cs")
    print(Cs)

    discriminants = numpy.square(Hs) - Cs
    print("Discriminants")
    print(discriminants)

    # https://stackoverflow.com/questions/52622172/numpy-where-function-can-not-avoid-evaluate-sqrtnegative
    # smaller_ts = numpy.where(discriminants > 0.00001, -Hs - numpy.sqrt(discriminants), t_max + 1)
    smaller_ts = numpy.full_like(discriminants, t_max + 1)
    mask = discriminants > 0.00001
    smaller_ts[mask] = -Hs[mask] - numpy.sqrt(discriminants[mask])
    # smaller_ts[mask] -= numpy.sqrt(discriminants[mask])
    print("Smaller ts")
    print(smaller_ts)

    # larger_ts = numpy.where(discriminants > 0.00001, -Hs + numpy.sqrt(discriminants), t_max + 1)
    larger_ts = numpy.full_like(discriminants, t_max + 1)
    mask = discriminants > 0.00001
    larger_ts[mask] = -Hs[mask] + numpy.sqrt(discriminants[mask])
    print("Larger ts")
    print(larger_ts)


    # https://stackoverflow.com/questions/37973135/numpy-argmin-for-elements-greater-than-a-threshold
    # smallest_smaller_t_index = numpy.argmin(smaller_ts)
    valid_idxs = numpy.where(smaller_ts > t_min)[0]
    smallest_smaller_t_index = valid_idxs[smaller_ts[valid_idxs].argmin()]
    print("Smallest smaller t index")
    print(smallest_smaller_t_index)

    #smallest_larger_t_index = numpy.argmin(larger_ts)
    valid_idxs = numpy.where(larger_ts > t_min)[0]
    smallest_larger_t_index = valid_idxs[larger_ts[valid_idxs].argmin()]
    print("Smallest larger t index")
    print(smallest_larger_t_index)

    smallest_t = smaller_ts[smallest_smaller_t_index]
    index = smallest_smaller_t_index
    if smallest_t > t_max:
        smallest_t = larger_ts[smallest_larger_t_index]
        index = smallest_larger_t_index
        if smallest_t > t_max:
            print("No hits!")
    
    print("Smallest t")
    print(smallest_t)

    print("Smallest T index")
    print(index)


    # test_hs = numpy.array([32, 5, 10])
    # test_discs = numpy.array([-0.1, 4, 16])
    # test_smaller_ts = numpy.where(test_discs > 0.00001, -test_hs - numpy.sqrt(test_discs), -42)
    # print(test_smaller_ts)


main.main()
# write_sphere_json()
# numpy_speedup_test()
# numpy_dot_cross_speed_test()
# numpy_vectorise_tests()


# start = time.perf_counter_ns()
# for i in range(10):
#     print(i)
# end = time.perf_counter_ns()

# total_time_ns = end - start
# print(total_time_ns)

# total_time_us = total_time_ns // 1000
# print(humanize.precisedelta(datetime.timedelta(microseconds=total_time_us), minimum_unit="microseconds"))


# a = numpy.array([1, 2, 3])
# b = numpy.array([4, 5, 6])
# a *= 3
# print(a)

# print(a.dot(a))

# print(a - b)
# print(b[0])
