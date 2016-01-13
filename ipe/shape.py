class MatrixTransform:
    def __init__(self, coefficients):
        self.coefficients = tuple(coefficients)

    def transform(self, x1, x2):
        a = self.coefficients
        return (a[0] * x1 + a[2] * x2 + a[4],
                a[1] * x1 + a[3] * x2 + a[5])


def load_matrix(data):
    return MatrixTransform(map(float, data.split()))


class Curve:
    def __init__(self):
        self.edges = []
        self.closed = False
        self.matrix = None

    def get_edges(self):
        edges = list(self.edges)
        if self.closed:
            edges.append((edges[-1][1], edges[0][0]))
        return [self.transform_edge(e) for e in edges]

    def transform_edge(self, e):
        u, v = e
        return self.transform_point(u), self.transform_point(v)

    def transform_point(self, v):
        if self.matrix is None:
            return v
        x, y = v.real, v.imag
        xp, yp = self.matrix.transform(x, y)
        return complex(xp, yp)

    def is_line_segment(self):
        return not self.closed and len(self.edges) == 1

    def is_polygon(self):
        return self.closed

    def endpoints(self):
        return self.transform_edge(self.edges[0])


class Shape:
    def __init__(self, curves):
        self.curves = curves

    def is_line_segment(self):
        return len(self.curves) == 1 and self.curves[0].is_line_segment()

    def is_polygon(self):
        return len(self.curves) == 1 and self.curves[0].is_polygon()

    def get_edges(self):
        return [e for c in self.curves for e in c.get_edges()]

    def endpoints(self):
        return self.curves[0].endpoints()


def load_shape(data, matrix_data=None):
    """Load Ipe shape data from a string.

    Mirrors ipe/src/ipelib/ipeshape.cpp Shape::load.
    """

    matrix = None if matrix_data is None else load_matrix(matrix_data)

    curves = []
    args = []  # list of coordinates
    subpath = None
    current_position = 0j

    def pop_point():
        y, x = args.pop(), args.pop()
        return complex(x, y)

    for tok in data.split():
        if tok == "h":
            # Closing path
            if not subpath:
                raise ValueError()
            subpath.closed = True
            subpath = None
        elif tok == "m":
            if len(args) != 2:
                raise ValueError()
            subpath = Curve()
            subpath.matrix = matrix
            curves.append(subpath)
            current_position = pop_point()
        elif tok == "l":
            if len(args) != 2:
                raise ValueError()
            v = pop_point()
            subpath.edges.append((current_position, v))
            current_position = v
        elif tok in 'q c a s e u'.split():
            raise NotImplementedError
        else:
            # Must be a number
            args.append(float(tok))

    if len(curves) == 0:
        raise ValueError()
    if any(len(c.edges) == 0 for c in curves):
        raise ValueError()
    return Shape(curves)
