"""
Hold results from testing rays against objects
"""

import numpy

class RayResults():
    """
    Hold the results of testing a set of rays agains objects.
    """

    def __init__(self):
        """
        Initialise the class.
        """
        self.hits = None
        self.ts = None
        self.pts = None
        self.normals = None
        self.uvs = None
        self.material_ids = None
        self.back_facing = None

    def set(self, hits, ts, pts, normals, uvs, material_ids, back_facing):
        """
        Set the data from existing results
        """
        self.hits = hits
        self.ts = ts
        self.pts = pts
        self.normals = normals
        self.uvs = uvs
        self.material_ids = material_ids
        self.back_facing = back_facing


    def set_blank(self, num_results, t_value, material_id):
        """
        Set the results to a blank/null dataset.

        Args:
            num_results (int): How many reults/rays to create in the
                dataset
            t_value (float): The value to set for the t values in the
                dataset. Typically set to just over the maximum t value
                that is permissable while running intersection checks.
            material_id (int): ID of the material to be set in the
                dataset. Needs to be less than or equal to 255.
        """

        self.hits = numpy.full(num_results, False)
        self.ts = numpy.full(num_results, t_value)
        self.pts = numpy.zeros((num_results, 3), dtype=numpy.single)
        self.normals = numpy.zeros((num_results, 3), dtype=numpy.single)
        self.uvs = numpy.zeros((num_results, 2), dtype=numpy.single)
        self.material_ids = numpy.full((num_results), material_id, dtype=numpy.ubyte)
        self.back_facing = numpy.full(num_results, False)

    def concatenate(self, other):
        """
        Extend this RayResults object with the contents of another.

        Args:
            other (RayResults): The other set of data to append onto
                the end of this dataset.
        """

        self.hits = numpy.concatenate((self.hits, other.hits), axis=0)
        self.ts = numpy.concatenate((self.ts, other.ts), axis=0)
        self.pts = numpy.concatenate((self.pts, other.pts), axis=0)
        self.normals = numpy.concatenate((self.normals, other.normals), axis=0)
        self.uvs = numpy.concatenate((self.uvs, other.uvs), axis=0)
        self.material_ids = numpy.concatenate((self.material_ids, other.material_ids), axis=0)
        self.back_facing = numpy.concatenate((self.back_facing, other.back_facing), axis=0)

    def merge(self, mask, other):
        """
        Merge some other results into these results.

        Mergin in the sense that certain values in the current results
        will be overwritten by the values from the other results.

        Args:
            mask (numpy.array): Array of True/False values that
                determine which of the current results should be
                overwritten.
            other (RayResults): The other set of data to merge into
                this dataset.
        """

        self.hits[mask] = other.hits
        self.ts[mask] = other.ts
        self.pts[mask] = other.pts
        self.normals[mask] = other.normals
        self.uvs[mask] = other.uvs
        self.material_ids[mask] = other.material_ids
        self.back_facing[mask] = other.back_facing