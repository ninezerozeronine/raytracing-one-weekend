import sys
import os
import time
import datetime
import random
import pprint
import json
import timeit
from textwrap import dedent


import numpy
import humanize


RNG = numpy.random.default_rng()

sys.path.append(os.path.abspath("../src"))

from raytracing_one_weekend import main, mttriangle_group, obj_tri_mesh


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

    setup = dedent("""
    import numpy
    RNG = numpy.random.default_rng()
    lots_a = RNG.random((50, 3))
    lots_b = RNG.random((50, 3))
    """)

    slow = dedent("""
    dots = []
    for i in range(50):
        dots.append(numpy.dot(lots_a[i], lots_b[i]))
    """)

    fast = dedent("""
    dots = numpy.sum(lots_a * lots_b, axis=1)
    """)

    faster = dedent("""
    dots = numpy.einsum("ij,ij->i", lots_a, lots_b)
    """)

    print(timeit.timeit(slow, setup=setup, number=10000))
    print(timeit.timeit(fast, setup=setup, number=10000))
    print(timeit.timeit(faster, setup=setup, number=10000))


def numpy_dot_speed_test():
    """
    ndarraysa dont have a .cross method
    """

    setup = dedent("""
    import numpy
    RNG = numpy.random.default_rng()
    lots_a = RNG.random((50, 3))
    lots_b = RNG.random((50, 3))
    """)

    dot_func = dedent("""
    dots = []
    for i in range(50):
        dots.append(numpy.dot(lots_a[i], lots_b[i]))
    """)

    dot_meth = dedent("""
    dots = []
    for i in range(50):
        dots.append(lots_a[i].dot(lots_b[i]))
    """)

    print(timeit.timeit(dot_func, setup=setup, number=10000))
    print(timeit.timeit(dot_meth, setup=setup, number=10000))


def numpy_preallocate_speed_test():
    setup = dedent("""
    import numpy
    RNG = numpy.random.default_rng()
    lots_a = RNG.random((5000, 3), dtype=numpy.float64)
    lots_b = RNG.random((5000, 3), dtype=numpy.float64)
    res = numpy.full(5000, 0.1234, dtype=numpy.float64)
    """)

    no_pre_sum = dedent("""
    dots = numpy.sum(lots_a * lots_b, axis=1)
    dots[0] = 1.2
    """)

    pre_sum = dedent("""
    numpy.sum(lots_a * lots_b, axis=1, out=res)
    """)

    no_pre_ein = dedent("""
    dots = numpy.einsum("ij,ij->i", lots_a, lots_b)
    """)

    pre_ein = dedent("""
    numpy.einsum("ij,ij->i", lots_a, lots_b, out=res)
    """)

    print(timeit.timeit(no_pre_sum, setup=setup, number=10000))
    print(timeit.timeit(pre_sum, setup=setup, number=10000))
    print(timeit.timeit(no_pre_ein, setup=setup, number=10000))
    print(timeit.timeit(pre_ein, setup=setup, number=10000))


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


def numpy_triangle_vectorise():
    directions = numpy.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])

    dottest0 = numpy.array([
        [1.0, 2.0, 3.0],
        [4.0, -5.0, 6.0]
    ])

    dottest1 = numpy.array([
        [1.0, 1.0, 1.0],
        [2.0, 1.0, 0.0]
    ])

    #print(dottest0.dot(dottest1))
    # print(numpy.einsum("ij,ij->i", dottest0, dottest1))

    # mask = dottest0[:, 2] > 4.0
    # print(mask)
    # print(numpy.logical_and(mask, numpy.array([True, True])))
    # print(numpy.einsum("ij,ij->i", dottest0[mask], dottest1[mask]))


    tBs = numpy.array([
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 2.0, 0.0]
    ])

    tAs = numpy.array([
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 2.0, 0.0]
    ])

    # print(numpy.array([5.0, 5.0, 5.0]) - tBs)
    # print(numpy.array([5.0, 5.0, 5.0]) * numpy.array([1.0, 2.0, 3.0]))
    # mask = numpy.array([5.0, 6.0, 7.0]) > 5.5
    # print(1/numpy.array([1, 2, 3])[mask])


    tp_vecs = numpy.cross(directions, tBs)
    # print(p_vecs)

    tdets = numpy.einsum("ij,ij->i", tAs, tp_vecs)
    # print(dets)

    triangles = [
        # 0 at the back
        [
            [-1.0, 0.0, -3.0],
            [1.0, 0.0, -3.0],
            [0.0, 2.0, -3.1],
        ],

        # 1 in the air
        [
            [-1.0, 2.0, -2.0],
            [1.0, 2.0, -2.0],
            [0.0, 4.0, -2.1],
        ],

        # 2 hit this one
        [
            [-1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [0.0, 2.0, -1.1],
        ],

        # 3 behind ray
        [
            [-1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 2.0, 1.1],
        ],

        # 4 parallel to ray
        [
            [0.0, 0.0, -4.0],
            [0.0, 0.0, -6.0],
            [0.0, 2.0, -5.0],
        ],


    ]

    pt0s = numpy.array([
        triangles[0][0],
        triangles[1][0],
        triangles[2][0],
        triangles[3][0],
        triangles[4][0],
    ])

    pt1s = numpy.array([
        triangles[0][1],
        triangles[1][1],
        triangles[2][1],
        triangles[3][1],
        triangles[4][1],
    ])

    pt2s = numpy.array([
        triangles[0][2],
        triangles[1][2],
        triangles[2][2],
        triangles[3][2],
        triangles[4][2],
    ])

    direction = numpy.array([0.0, 0.0, -1.0])
    origin = numpy.array([0.0, 0.1, 0.0])
    t_min = 0.00001
    t_max = 5000.0
    As = pt1s - pt0s
    Bs = pt2s - pt0s
    normals = numpy.cross(As, Bs)
    # print(normals)
    # https://stackoverflow.com/questions/2850743/numpy-how-to-quickly-normalize-many-vectors
    normals /= numpy.sqrt((normals*normals).sum(axis=1))[..., numpy.newaxis]
    # print(normals)

    p_vecs = numpy.cross(direction, Bs)
    dets = numpy.einsum("ij,ij->i", As, p_vecs)
    print(dets)
    valid_tri_idxs = numpy.absolute(dets) > 0.00001
    print(valid_tri_idxs)

    inv_dets = numpy.zeros_like(dets)
    inv_dets[valid_tri_idxs] = 1.0/dets[valid_tri_idxs]

    t_vecs = origin - pt0s

    Us = numpy.zeros_like(dets)
    Us[valid_tri_idxs] = numpy.einsum(
        "ij,ij->i",
        t_vecs[valid_tri_idxs],
        p_vecs[valid_tri_idxs]
    ) * inv_dets[valid_tri_idxs]
    # print(Us)
    valid_tri_idxs = numpy.logical_and(
        valid_tri_idxs,
        numpy.logical_not(numpy.logical_or(Us > 1.0, Us < 0))
    )

    print(valid_tri_idxs)

    q_vecs = numpy.cross(t_vecs, As)
    Vs = numpy.zeros_like(dets)
    Vs[valid_tri_idxs] = numpy.einsum(
        "ij,j->i",
        q_vecs[valid_tri_idxs],
        direction
    ) * inv_dets[valid_tri_idxs]
    valid_tri_idxs = numpy.logical_and(
        valid_tri_idxs,
        numpy.logical_not(numpy.logical_or(Vs < 0.0, (Us + Vs) > 1.0))
    )

    print(valid_tri_idxs)

    Ts = numpy.zeros_like(dets)
    Ts[valid_tri_idxs] = numpy.einsum(
        "ij,ij->i",
        Bs[valid_tri_idxs],
        q_vecs[valid_tri_idxs]
    ) * inv_dets[valid_tri_idxs]
    print(Ts)

    valid_t_idxs = numpy.asarray(
        numpy.logical_not(numpy.logical_or(Ts < t_min, Ts > t_max))
    ).nonzero()[0]
    if valid_t_idxs.size > 0:
        smallest_t_index = valid_t_idxs[Ts[valid_t_idxs].argmin()]
        smallest_t = Ts[smallest_t_index]
    else:
        print("No valid hits")
        return None

    print(smallest_t)
    print(smallest_t_index)

    valid_tri_idxs = numpy.logical_and(
        valid_tri_idxs,
        numpy.logical_not(numpy.logical_or(Ts < t_min, Ts > t_max))
    )

    print(valid_tri_idxs)

    print(numpy.mean(
        numpy.concatenate((pt0s, pt1s, pt2s), axis=0),
        axis=0
    ))


def mttri_grp_test():
    triangles = [
        # 0 at the back
        [
            [-1.0, 0.0, -3.0],
            [1.0, 0.0, -3.0],
            [0.0, 2.0, -3.1],
        ],

        # 1 in the air
        [
            [-1.0, 2.0, -2.0],
            [1.0, 2.0, -2.0],
            [0.0, 4.0, -2.1],
        ],

        # 2 hit this one
        [
            [-1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [0.0, 2.0, -1.1],
        ],

        # 3 behind ray
        [
            [-1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 2.0, 1.1],
        ],

        # 4 parallel to ray
        [
            [0.0, 0.0, -4.0],
            [0.0, 0.0, -6.0],
            [0.0, 2.0, -5.0],
        ],
    ]

    grp = mttriangle_group.MTTriangleGroup()
    for triangle in triangles:
        grp.add_triangle(
            numpy.array(triangle[0]),
            numpy.array(triangle[1]),
            numpy.array(triangle[2]),
            None
        )

    print(grp.bounds_centre)
    print(grp.bounds_radius)


def test_obj_read():
    obj_mesh = obj_tri_mesh.OBJTriMesh()
    obj_mesh.read("bunny.obj")

    for triangle in obj_mesh.faces[0:1]:
        print(triangle)
        print(obj_mesh.vertices[triangle[0][0]][0])
        print(obj_mesh.vertices[triangle[0][0]][1])
        print(obj_mesh.vertices[triangle[0][0]][2])
        print("-")
        print(obj_mesh.vertices[triangle[1][0]][0])
        print(obj_mesh.vertices[triangle[1][0]][1])
        print(obj_mesh.vertices[triangle[1][0]][2])
        print("-")
        print(obj_mesh.vertices[triangle[2][0]][0])
        print(obj_mesh.vertices[triangle[2][0]][1])
        print(obj_mesh.vertices[triangle[2][0]][2])
        print(" ")


main.main()
# test_obj_read()
# write_sphere_json()
# numpy_triangle_vectorise()
# mttri_grp_test()
# numpy_speedup_test()
# numpy_preallocate_speed_test()
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
