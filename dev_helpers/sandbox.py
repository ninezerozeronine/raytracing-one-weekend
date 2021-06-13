import sys
import os
import time
import datetime
import random
import pprint
import json

import numpy
import humanize

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


main.main()
# write_sphere_json()


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
