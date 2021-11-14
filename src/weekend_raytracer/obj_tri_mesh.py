"""
Read OBJ files and pull data from them.

https://all3dp.com/1/obj-file-format-3d-printing-cad/

Note that indecies of things in the OBj file start at 1 - yay!
"""
import numpy

class OBJTriMesh():
    """
    Read OBJ files and pull data from them.
    """

    def __init__(self):
        self.vertices = []
        self.vertex_normals = []
        self.uvs = []
        self.faces = []

    def _clear(self):
        self.vertices = []
        self.vertex_normals = []
        self.uvs = []
        self.faces = []

    def get_smooth_vertex_normal(self, vertex_index):
        """
        The points are defined in a counter clockwise order looking from
        the direction the normal points in:

          2
          |\
        B | \
          |  \
          0---1
            A

        A = 0 -> 1
        B = 0 -> 2
        C = AxB

        A is X-like, B is Y-like and C is Z-like from a right handed
        coordinate system.
        """
        adjacent_normals = None
        for face in self.faces:
            for face_pt in face:
                if vertex_index == face_pt[0]:
                    # Calculate normal of face
                    A = (
                        numpy.array(self.vertices[face[1][0]], dtype=numpy.single)
                        - numpy.array(self.vertices[face[0][0]], dtype=numpy.single)
                    )
                    B = (
                        numpy.array(self.vertices[face[2][0]], dtype=numpy.single)
                        - numpy.array(self.vertices[face[0][0]], dtype=numpy.single)
                    )
                    normal = numpy.cross(A, B)
                    normal /= numpy.linalg.norm(normal)
                    if adjacent_normals is None:
                        adjacent_normals = numpy.array([normal], dtype=numpy.single)
                    else:
                        adjacent_normals = numpy.append(adjacent_normals, [normal], axis=0)

        average_normal = numpy.mean(adjacent_normals, axis=0)
        average_normal /= numpy.linalg.norm(average_normal)
        return average_normal

    def read(self, filepath):

        self._clear()

        with open(filepath) as file_handle:
            obj_lines = file_handle.read().splitlines()

        for line_no, line in enumerate(obj_lines, start=1):
            tokens = line.split()
            if not tokens:
                continue

            # Process vertex
            if tokens[0] == "v":
                self.vertices.append(
                    (
                        float(tokens[1]),
                        float(tokens[2]),
                        float(tokens[3]),
                    )
                )
                continue

            # Process vertex normal
            if tokens[0] == "vn":
                self.vertex_normals.append(
                    (
                        float(tokens[1]),
                        float(tokens[2]),
                        float(tokens[3]),
                    )
                )
                continue

            # Process UV
            if tokens[0] == "vt":
                self.uvs.append(
                    (
                        float(tokens[1]),
                        float(tokens[2]),
                    )
                )
                continue

            # Process Face
            if tokens[0] == "f":
                face = self._process_face(line_no, line, tokens)
                if face is not None:
                    self.faces.append(face)
                continue

        print(f"Read {len(self.vertices)} vertices, {len(self.vertex_normals)} vertex normals, {len(self.uvs)} UVs and {len(self.faces)} faces.")

    def _process_face(self, line_no, line, tokens):
        # Skip if not a triangle
        if len(tokens) != 4:
            print(f"Non triangular face on line {line_no}: {line}")
            return None
        point_defs = tokens[1:]
        # print(point_defs)
        face_points = []
        for point_def in point_defs:
            item_indecies = point_def.split("/")
            # print(item_indecies)

            vert_index = -1
            uv_index = -1
            vertex_normal_index = -1

            for index, item_index in enumerate(item_indecies):
                # First element is a vert
                if index == 0:
                    vert_index = int(item_index)
                    if vert_index > len(self.vertices):
                        print(
                            f"Trying to add out of range vertex ({vert_index}) on line {line_no} - {line}"
                        )
                        vert_index = -1
                    else:
                        vert_index -= 1
                    continue

                # Second element (if present) is a UV
                if index == 1:
                    if item_index:
                        uv_index = int(item_index)
                        if uv_index > len(self.uvs):
                            print(
                                f"Trying to add out of range uv ({uv_index}) on line {line_no} - {line}"
                            )
                            uv_index = -1
                        else:
                            uv_index -= 1
                    continue

                # Third element (if present) is a vertex normal
                if index == 2:
                    if item_index:
                        vertex_normal_index = int(item_index)
                        if vertex_normal_index > len(self.vertex_normals):
                            print(
                                f"Trying to add out of range vertex_normal ({vertex_normal_index}) on line {line_no} - {line}"
                            )
                            vertex_normal_index = -1
                        else:
                            vertex_normal_index -= 1
                    continue

            # Bundle up the vert, uv and normal info into a fac point and add it to the list
            # print((vert_index, uv_index, vertex_normal_index))
            face_points.append((vert_index, uv_index, vertex_normal_index))

        return face_points
