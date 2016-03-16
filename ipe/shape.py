class MatrixTransform:
    def __init__(self, coefficients):
        self.coefficients = tuple(coefficients)

    def transform(self, x1, x2):
        a = self.coefficients
        return (a[0] * x1 + a[2] * x2 + a[4],
                a[1] * x1 + a[3] * x2 + a[5])

    @classmethod
    def identity(cls):
        """
        >>> MatrixTransform.identity().transform(123, 456)
        (123, 456)
        """
        return cls((1, 0, 0, 1, 0, 0))


def load_matrix(data):
    return MatrixTransform(map(float, data.split()))


class Curve:
    def __init__(self, edges=None, closed=False, matrix=None):
        self.edges = [] if edges is None else list(edges)
        self.closed = closed
        self.matrix = matrix

    def __repr__(self):
        return "Curve(edges=%r, closed=%r, matrix=%r)" % (
            self.edges, self.closed, self.matrix)

    @classmethod
    def make_polyline(cls, points):
        if not all(isinstance(p, complex) for p in points):
            raise TypeError(str(points))
        edges = [
            (u, v)
            for u, v in zip(points[:-1], points[1:])
        ]
        return cls(edges)

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

    def to_ipe_path(self):
        directives = []
        pos = None
        for u, v in self.edges:
            if pos is None:
                pos = u
                directives.append('%g %g m\n' % (u.real, u.imag))
            if pos != u:
                raise Exception("to_ipe_path: Edges are not connected")
            pos = v
            directives.append('%g %g l\n' % (v.real, v.imag))
        if self.closed:
            directives.append('h\n')
        return ''.join(directives)


class Shape:
    def __init__(self, curves):
        self.curves = curves

    def __repr__(self):
        return "Shape(%r)" % (self.curves,)

    @classmethod
    def make_polyline(cls, points):
        return cls([Curve.make_polyline(points)])

    def is_line_segment(self):
        return len(self.curves) == 1 and self.curves[0].is_line_segment()

    def is_polygon(self):
        return len(self.curves) == 1 and self.curves[0].is_polygon()

    def get_edges(self):
        return [e for c in self.curves for e in c.get_edges()]

    def endpoints(self):
        return self.curves[0].endpoints()

    def to_ipe_path(self):
        return ''.join(c.to_ipe_path() for c in self.curves)


def load_shape(data, attrib=None):
    """Load Ipe shape data from a string.

    Mirrors ipe/src/ipelib/ipeshape.cpp Shape::load.
    """

    if attrib is None:
        attrib = dict()
    matrix_data = attrib.get('matrix')
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
