class Curve:
    def __init__(self):
        self.vertices = []
        self.closed = False

    def edges(self):
        e = [(u, v) for u, v in zip(self.vertices[:-1], self.vertices[1:])]
        if self.closed:
            e.append((self.vertices[-1], self.vertices[0]))
        return e


def load(data):
    """Load Ipe shape data from a string.

    Mirrors ipe/src/ipelib/ipeshape.cpp Shape::load.
    """

    shape = []  # list of Curve
    args = []  # list of coordinates
    subpath = None
    current_position = 0j

    def pop_point():
        return complex(

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
            shape.append(subpath)
            current_position =
