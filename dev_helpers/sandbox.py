import sys
import os
import time
import datetime
import random
import pprint
import json
import timeit
from textwrap import dedent
from math import sqrt
from functools import reduce

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


def mask_speed_test():
    setup = dedent("""
    import numpy
    RNG = numpy.random.default_rng()
    nums = RNG.random((50000, 3))
    """)

    no_mask = dedent("""
    sqrts = numpy.sqrt(numpy.maximum(0.00001, nums))
    """)

    mask = dedent("""
    res = numpy.full_like(nums, 10.0)
    mask = nums > 0.00001
    res[mask] = numpy.sqrt(nums[mask])
    """)

    print(timeit.timeit(no_mask, setup=setup, number=1000))
    print(timeit.timeit(mask, setup=setup, number=1000))


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


def numpy_axis_combo_tests():

    As = numpy.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            # [7.0, 8.0, 9.0],
        ]
    )

    Bs = numpy.array(
        [
            [6.0, 6.0, 6.0],
            [7.0, 7.0, 7.0],
            [8.0, 8.0, 8.0],
            [9.0, 9.0, 9.0],
        ]
    )

    Cs = numpy.array([0.1, 0.2, 0.3, 0.4])

    # print(As[:])
    # print("-")
    # print(As[:, numpy.newaxis])
    # print("-")
    # print(As[numpy.newaxis, :])

    # # Dot products
    # # Need to test einsum here!
    # print("dot")
    # print(As[0].dot(Bs[0]))
    # print(As.dot(Bs.T)[0, 0])
    # print(As[1].dot(Bs[3]))
    # print(As.dot(Bs.T)[1, 3])
    # print("")

    # # Cross Products
    # print("cross")
    # print(numpy.cross(As[0], Bs[0]))
    # print(numpy.cross(As[:, numpy.newaxis], Bs)[0, 0])

    # print(numpy.cross(As[0], Bs[3]))
    # print(numpy.cross(As[:, numpy.newaxis], Bs)[0, 3])

    # print(numpy.cross(As[1], Bs[1]))
    # print(numpy.cross(As[:, numpy.newaxis], Bs)[1, 1])
    # print("")

    # Adding
    print("add")
    print(As[0] + (Bs[0]))
    res = As[:, numpy.newaxis] + Bs
    print(res[0, 0])
    print(res.shape)
    print("-")

    print(As[1] + (Bs[3]))
    res = As[:, numpy.newaxis] + Bs
    print(res[1, 3])
    print("-")
    print(res)
    print("")

    single = numpy.array([1.0, 2.0, 3.0], dtype=numpy.single)
    grid = numpy.array(
        [
            [1.0, 1.0, 0.0],
            [2.0, 0.0, 2.0],
            [0.0, 3.0, 3.0],
        ],
        dtype=numpy.single
    )

    res = numpy.einsum("i,ki", single, grid)
    print(res)

    vecs = numpy.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        dtype=numpy.single
    )
    vec_grid = numpy.array(
        [
            [
                [1, 1, 0],
                [2, 0, 2],
                [0, 3, 3],
                [4, 4, 4],
            ],
            [
                [5, 0, 0],
                [0, 6, 0],
                [0, 0, 7],
                [8, 8, 8],
            ],
        ]
    )
    res = numpy.einsum("...i,...ki", vecs, vec_grid)
    print(res)


    vec_grid1 = numpy.array(
        [
            [
                [1, 2, 3],
                [4, 5, 6],
                [0, 3, 3],
                [4, 4, 4],
            ],
            [
                [5, 0, 0],
                [0, 6, 0],
                [0, 0, 7],
                [3, 0, 8],
            ],
        ]
    )
    vec_grid2 = numpy.array(
        [
            [
                [1, 1, 1],
                [0, 0, 2],
                [0, 3, 3],
                [4, 4, 4],
            ],
            [
                [5, 0, 0],
                [0, 6, 0],
                [0, 0, 7],
                [0, 9, 8],
            ],
        ]
    )

    res = numpy.einsum("...ij,...ij->...i", vec_grid1, vec_grid2)
    print(res)

    vals = numpy.array([1, 2, 3, 4], dtype=numpy.single)
    val_grid = numpy.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ],
        dtype=numpy.single
    )
    # res = val_grid[:, numpy.newaxis] - vals**2
    res = val_grid - vals**2
    print(res)


def sphere_colour_test():

    hit_indecies = numpy.array([3,2,-1,0])
    colours = numpy.array([0,0,0,0])
    sphere_colours = numpy.array([5,6,7,8])
    print(hit_indecies)
    print(colours)
    print(sphere_colours)
    ray_hits = hit_indecies > -1
    ray_misses = hit_indecies < 0
    colours[ray_hits] = sphere_colours[hit_indecies[ray_hits]]
    print(colours)
    print(colours[ray_hits])
    print(colours[ray_misses])


def ray_array_test():
    # width = 5
    # height = 3
    # samples = 2

    # ray_array = numpy.zeros((width, height, samples, 3), dtype=numpy.single)
    # count = 0
    # for y in range(height):
    #     for x in range(width):
    #         for s in range(samples):
    #             ray_array[x, y, s] = [count, count, count]
    #             count += 1

    # print(count - 1)
    # print(ray_array[4,2,1])
    # print(ray_array[1,1,0])
    # flattened = ray_array.reshape(-1, 3)
    # print(flattened.shape)
    # stacked = flattened.reshape(width, height, samples,3)
    # print(stacked.shape)
    # print(stacked[4,2,1])
    # print(stacked[1,1,0])

    # ray_sums = numpy.mean(ray_array, axis=2)
    # print(ray_sums.shape)
    # test_x = 1
    # test_y = 2
    # print(ray_array[test_x, test_y, 0], ray_array[test_x, test_y, 1])
    # print(ray_sums[test_x, test_y, 0])

    # ray_array[0,...,0] = 0
    # ray_array[1,...,0] = 1
    # ray_array[2,...,0] = 2
    # print(ray_array[1,0,0])
    # print(ray_array[1,1,0])
    # print(ray_array[1,1,1])
    # print(ray_array[2,1,1])

    # ray_array[:,0,...,1] = 0
    # ray_array[:,1,...,1] = 1
    # ray_array[:,2,...,1] = 2
    # print(ray_array[1,0,0])
    # print(ray_array[2,1,0])
    # print(ray_array[3,2,1])






    lens_radius = 0.1
    U = numpy.array([0.5, 0.0, 0.0])
    V = numpy.array([0.0, 0.5, 0.0])
    viewport_horizontal = numpy.array([1.0,0.0,0.0])
    viewport_vertical = numpy.array([0.0,1.0,0.0])
    bottomleft_focalplane_pos = numpy.array([0.0,0.0,0.0])
    camera_pos = numpy.array([5.0, 5.0, 5.0])

    width = 90
    height = 60
    samples = 50


    pixel_positions = numpy.zeros(
        (width, height, samples, 3), dtype=numpy.single
    )
    for x_coord in range(width):
        pixel_positions[x_coord,...,0] = x_coord

    for y_coord in range(height):
        pixel_positions[:,y_coord,...,1] = y_coord

    print(pixel_positions[70, 55, 45])

    sample_offsets = RNG.uniform(low=0.0, high=1.0, size=(width, height, samples, 3))
    pixel_positions += sample_offsets
    print(pixel_positions[70, 55, 45])

    pixel_positions[...,0] /= width
    pixel_positions[...,1] /= height
    viewport_percentages = pixel_positions

    print(viewport_percentages[0, 0, 0])
    print(viewport_percentages[45, 30, 0])
    print(viewport_percentages[89, 59, 0])

    # Create an array of offests in an xy disk for every sample.
    xy_disk_coords = RNG.uniform(low=0.0, high=1.0, size=(width, height, samples, 3))
    flattened = xy_disk_coords.reshape(-1, 3)
    flattened[..., 2] = 0
    while True:
        dots = numpy.einsum("ij,ij->i", flattened, flattened)
        if not numpy.any(dots > 1.0):
            break
        new_coords = RNG.uniform(low=0.0, high=1.0, size=(width, height, samples, 3))
        new_flattened = new_coords.reshape(-1, 3)
        new_flattened[..., 2] = 0
        flattened[dots > 1.0] = new_flattened[dots > 1.0]
    xy_disk_coords = flattened.reshape(width, height, samples, 3)

    print("---")

    print(xy_disk_coords[0,0,0])
    print(xy_disk_coords[2,2,1])

    offset_amounts = lens_radius * xy_disk_coords

    print(offset_amounts[0,0,0])
    print(offset_amounts[2,2,1])

    # print(offset_amounts.shape)
    # print(U.shape)
    offset_vecs = (
        offset_amounts[..., 0, numpy.newaxis] * U
        + offset_amounts[..., 1, numpy.newaxis] * V
    )

    print(offset_vecs.shape)
    print(offset_vecs[0,0,0])
    print(offset_vecs[2,2,1])



    # Turn the viewport percentages into positions in space on the viewport
    pts_on_viewport = (
        bottomleft_focalplane_pos
        + viewport_horizontal * viewport_percentages[..., 0, numpy.newaxis]
        + viewport_vertical * viewport_percentages[..., 1, numpy.newaxis]
    )

    print(pts_on_viewport.shape)

    # viewport_percentages[..., :] *= viewport_horizontal
    # viewport_percentages[..., :] *= viewport_vertical
    # pts_on_viewport = viewport_percentages + bottomleft_focalplane_pos

    ray_origins = offset_vecs + camera_pos
    ray_dirs = pts_on_viewport - ray_origins

    return ray_origins, ray_dirs




    print(pixel_positions[0, 0, 0])
    print(pixel_positions[45, 30, 0])
    print(pixel_positions[89, 59, 0])




    # x_coords = numpy.arange(width)
    # y_coords = numpy.arange(height)
    # sample_indecies = numpy.arange(samples)



    # print(res)
    # print("")

    # Find the smallest dot product using the second A
    # dots = As.dot(Bs.T)
    # print(dots)
    # index = numpy.argmin(dots, axis=1)
    # print(index)
    # print(dots[index])


    # print(As[:, None] * Bs)
    # print(numpy.multiply.outer(As, Bs))


def ray_parallelisation_test():
    dtype = numpy.single
    ray_origins = numpy.array(
        [
            [4.5, 0.0, -2.0],
            [3.5, 0.0, -2.0],
            [2.0, 0.0, -2.0],
            [5.0, 0.0, 5.0],
            [0.0, 0.0, 0.0],
            [11, 0, 5],
            [5, 0, 10],
            [5.5, 0, -2]
        ],
        dtype=dtype
    )
    ray_dirs = numpy.array(
        [
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
            [-sqrt(2)/2.0, 0.0, -sqrt(2)/2.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=dtype
    )
    sphere_centres = numpy.array(
        [
            [5, 0, -5],
            [5, 0, -8],
            [3, 0, 3],
            [10, 0, 0],
            [5, 0, 10],
        ],
        dtype=dtype
    )
    sphere_radii = numpy.array(
        [
            1,
            2,
            3,
            1.000001,
            2
        ],
        dtype=dtype
    )

    # This is a grid of vectors num_rays by num_spheres in size
    # It's every origin minus every sphere centre
    #
    #    C0    C1    C2
    #    -----------------
    # R0|R0-C0 R0-C1 R0-C2
    # R1|R1-C0 R1-C1 R1-C2
    C_to_Os = ray_origins[:, numpy.newaxis] - sphere_centres

    # This is a grid of scalars num_rays by num_spheres in size
    # It's as if we take the C_to_Os gris, and then for each row
    # (which corresponds to a ray), find the dot product of the ray for
    # that row and each C_to_O
    #
    #    C0         C1         C2
    #    --------------------------------
    # R0|R0.(R0-C0) R0.(R0-C1) R0.(R0-C2)
    # R1|R1.(R1-C0) R1.(R1-C1) R1.(R1-C2)
    Hs = numpy.einsum("...i,...ki", ray_dirs, C_to_Os)

    # This is a grid of scalars num_rays by num_spheres in size.
    # It's the dot product of each C_to_O with itself, minus the radius
    # of the sphere for that column squared
    #
    #    S0                 S1                 S2
    #    ------------------------------------------------------
    # R0|C20.C20 - S0.r^2   C20.C20 - S1.r^2   C20.C20 - S2.r^2
    # R1|C20.C20 - S0.r^2   C20.C20 - S1.r^2   C20.C20 - S2.r^2
    Cs = numpy.einsum("...ij,...ij->...i", C_to_Os, C_to_Os) - sphere_radii**2


    # This is a grid of scalars num_rays by num_spheres in size.
    discriminants = numpy.square(Hs) - Cs

    sqrt_discriminants = numpy.sqrt(numpy.maximum(0.00001, discriminants))
    smaller_ts = -Hs - sqrt_discriminants
    larger_ts = -Hs + sqrt_discriminants
    ts = numpy.where(
        (smaller_ts > 0.0) & (smaller_ts < larger_ts),
        smaller_ts,
        larger_ts
    )
    t_filter = (discriminants > 0.00001) & (ts > 0.00001)
    final_ts = numpy.where(t_filter, ts, 101)
    print(final_ts)
    print("-")

    # nearset_hits = reduce(numpy.minimum, final_ts.T)
    # print(nearset_hits)
    # print("-")

    # A 1D array num_rays long that contains the index of the
    # sphere with the smallest t
    smallest_t_indecies = numpy.argmin(final_ts, axis=1)
    print(smallest_t_indecies)

    # A 1D array num_rays long containing the t values for each ray
    smallest_ts = final_ts[numpy.arange(ray_origins.shape[0]), smallest_t_indecies]
    print(smallest_ts)

    # A 1D array num_rays long that contains the index of the
    # sphere with the smallest t, or -1 if the ray hit no spheres
    sphere_hit_indecies = numpy.where(
        smallest_ts < 100,
        smallest_t_indecies,
        -1
    )
    print(sphere_hit_indecies)




    # smallest_t_indecies = numpy.argmin(final_ts, axis=1)
    # print(smallest_t_indecies)
    # print(final_ts[(0, 1, 2, 3, 4, 5, 6), smallest_t_indecies])
    # print(final_ts[numpy.arange(ray_origins.shape[0]), smallest_t_indecies])
    # print(numpy.where(final_ts[(0, 1, 2, 3, 4, 5, 6), smallest_t_indecies] < 100, smallest_t_indecies, -1))


    # Now know for each ray if it hits a sphere, and if it does, which sphere and how far along the ray the coliision is.
    # For each hit need to:
    # - Determine the scatter ray and queue it up for the next bounce
    # - Determine the hit colour

    







    # valid_smallest = numpy.where()


    # valid_idxs = numpy.transpose(numpy.nonzero(final_ts < 100))
    # valid_idxs = numpy.nonzero(final_ts < 100)

    # print(valid_idxs)
    # print(final_ts[valid_idxs])

    # mask = numpy.where(final_ts < 100)
    # valid_idxs = numpy.nonzero(final_ts < 100)
    # smallest_index = valid_idxs[final_ts[valid_idxs]]


    # print(mask)
    # print(mask[0])
    # print(final_ts[final_ts < 100])
    # print(numpy.argmin(final_ts[final_ts < 100], axis=1))


    # print(discriminants)

    # ray_index = 3
    # sphere_index = 3
    # C_to_O = ray_origins[ray_index] - sphere_centres[sphere_index]
    # H = ray_dirs[ray_index].dot(C_to_O)
    # C = C_to_O.dot(C_to_O) - sphere_radii[sphere_index]
    # discriminant = H**2 - C
    # print(discriminant)
    # print(discriminants[ray_index, sphere_index])


def vec_subset_test():
    valid = numpy.array(
        [
            True,
            False,
            True,
            True,
        ],
        dtype=numpy.bool_
    )

    vecs = numpy.array(
        [
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
        ],
        dtype=numpy.single
    )

    print(vecs)
    vecs[valid] = vecs[valid] + 2
    print(vecs)
    new_data = vecs[valid] + 2
    print(new_data)


def array_split_assign_test():
    pass
    # res = numpy.arange(50)
    # nums = numpy.arange(50)
    # res[0:10] = nums[0:10]
    # res[10:20] = nums[10:20] * 10
    # parts = numpy.array_split(nums, 3)
    # for part in parts:
    #     print(part)
    # numpy.
    # rejoined = numpy.concatenate(parts)
    # print(rejoined)


def disk_coords_test():
    width = 900
    height = 600
    samples = 50

    xy_disk_coords = RNG.uniform(low=0.0, high=1.0, size=(width, height, samples, 3))
    flattened = xy_disk_coords.reshape(-1, 3)
    flattened[..., 2] = 0
    while True:
        print("Loop")
        dots = numpy.einsum("ij,ij->i", flattened, flattened)
        if not numpy.any(dots > 1.0):
            break
        new_coords = RNG.uniform(low=0.0, high=1.0, size=(width, height, samples, 3))
        new_flattened = new_coords.reshape(-1, 3)
        new_flattened[..., 2] = 0
        flattened[dots > 1.0] = new_flattened[dots > 1.0]

    print(flattened[0])
    print(numpy.einsum("ij,ij->i", flattened, flattened)[0])


def masked_assign_test():
    """
    Remap a continuous array to specific indecies of an array 
    """
    nums = numpy.arange(10)
    nums[nums % 2 == 0] = numpy.array([-1,-2,-3,-4,-5])
    print(nums)


def test_material():
    hit_points = numpy.zeros((50,3))
    hit_normals = numpy.zeros((50,3))
    hit_normals[:,1] = 1.0
    print(hit_normals)

    pts_in_hemisph = RNG.uniform(low=-1.0, high=1.0, size=(hit_points.shape[0], 3))

    # Would be good to optimise this so that we only check the newly
    # regenerated points
    while True:
        lengths_squared = numpy.einsum("ij,ij->i", pts_in_hemisph, pts_in_hemisph)

        invalid_pts = lengths_squared > 1.0
        num_bad_pts = numpy.count_nonzero(invalid_pts)
        if num_bad_pts == 0:
            break
        new_pts = RNG.uniform(low=-1.0, high=1.0, size=(num_bad_pts, 3))
        pts_in_hemisph[invalid_pts] = new_pts

    # Reverse any points in the wrong hemisphere
    cosine_angles = numpy.einsum("ij,ij->i", pts_in_hemisph, hit_normals)
    facing_wrong_way = cosine_angles < 0.0
    pts_in_hemisph[facing_wrong_way] *= -1.0


    # Make sure none of the points are very close to 0,0,0. If they
    # are, replace with normal
    lengths_squared = numpy.einsum("ij,ij->i", pts_in_hemisph, pts_in_hemisph)
    too_short = lengths_squared < 0.00001
    pts_in_hemisph[too_short] = hit_normals[too_short]

    print(pts_in_hemisph)

    ray_dirs = numpy.zeros((3,3))
    ray_dirs[0,1] = 1
    ray_dirs[1,1] = -5
    ray_dirs[2,1] = 10
    ray_dirs /= numpy.sqrt(numpy.einsum("ij,ij->i", ray_dirs, ray_dirs))[..., numpy.newaxis]
    print(ray_dirs)


def get_array_slice_start_end(length, num_slices, slice_index):
    """

    """

    if not (length > 0):
        raise Exception("Invalid length, must be > 0")

    if not (1 <= num_slices <= length):
        raise Exception("Invalid number of slices, must be >= 1 and <= length")

    if not (0 <= slice_index < length):
        raise Exception("Invalid slice index, must be >= 0 and < length")

    slice_size = length // num_slices
    # if (length % slice_size) != 0:
    #     if (length - (slice_size * (num_slices - 1))) >= num_slices:
    #         slice_size += 1
    slice_start = slice_size * slice_index
    if slice_index == (num_slices - 1):
        slice_end = length
    else:
        slice_end = (slice_index + 1) * slice_size

    return slice_start, slice_end


def test_array_slice(length, num_slices):
    nums = list(range(1,length+1))
    for slice_index in range(num_slices):
        slice_start, slice_end = get_array_slice_start_end(length, num_slices, slice_index)
        print(nums[slice_start:slice_end], end=" ")


# for i in range(4,20):
#     test_array_slice(i,4)
#     print("")



# main.main()
# masked_assign_test()
test_material()
# test_array_slice(8,3)
# array_split_assign_test()
# numpy_axis_combo_tests()
# ray_parallelisation_test()
# sphere_colour_test()
# ray_array_test()
# disk_coords_test()
# vec_subset_test()
# mask_speed_test()
# test_obj_read()
# write_sphere_json()
# numpy_triangle_vectorise()
# mttri_grp_test()
# numpy_speedup_test()
# numpy_preallocate_speed_test()
# numpy_dot_speed_test()
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
