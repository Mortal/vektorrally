import re


class IpeObject:
    def __init__(self, matrix):
        self.layer = None
        if isinstance(matrix, str):
            matrix = load_matrix(matrix)
        self.matrix = matrix

    def transform(self, xy):
        x, y = xy
        if self.matrix is None:
            return (x, y)
        else:
            return self.matrix.transform(x, y)


class MatrixTransform:
    def __init__(self, coefficients):
        self.coefficients = tuple(coefficients)

    def transform(self, x1, x2):
        a = self.coefficients
        return (a[0] * x1 + a[2] * x2 + a[4],
                a[1] * x1 + a[3] * x2 + a[5])

    def determinant(self):
        a = self.coefficients
        return a[0] * a[3] - a[1] * a[2]

    def and_then(self, other):
        """
        Compose matrix transformations.
        >>> a = MatrixTransform((3, 0, 0, 2, 16, 16))
        >>> b = MatrixTransform((2, 0, 0, 2, -16, 0))
        >>> ab = a.and_then(b)
        >>> ab.transform(0, 0)
        (16, 32)
        >>> ab.transform(0, 16)
        (16, 96)
        >>> ab.transform(16, 16)
        (112, 96)
        >>> ab.coefficients
        (6, 0, 0, 4, 16, 32)
        >>> MatrixTransform((1, 1, 0, 1, 16, 16)).and_then(
        ... MatrixTransform((0, -1, 1, 1, -16, 0))).coefficients
        (1, 0, 1, 1, 0, 0)
        """
        (z1, z2), (a1, a2), (b1, b2) = [
            other.transform(*self.transform(*p))
            for p in [(0, 0), (1, 0), (0, 1)]]
        return type(self)((a1-z1, a2-z2, b1-z1, b2-z2, z1, z2))

    @classmethod
    def identity(cls):
        """
        >>> MatrixTransform.identity().transform(123, 456)
        (123, 456)
        """
        return cls((1, 0, 0, 1, 0, 0))


def load_matrix(data):
    return MatrixTransform(map(float, data.split()))


class Text(IpeObject):
    def __init__(self, text, attrib, matrix=None):
        self.text = text
        self.attrib = attrib
        super().__init__(matrix=matrix or attrib.get('matrix'))

        x, y = map(float, self.attrib['pos'].split())
        self.position = self.transform((x, y))

    def __repr__(self):
        return '<Text %r>' % (self.text[:100],)


class Image(IpeObject):
    def __init__(self, bitmap_data, attrib, matrix=None):
        self.data = bitmap_data
        self.attrib = attrib
        super().__init__(matrix=matrix or attrib.get('matrix'))
        x1, y1, x2, y2 = self.attrib['rect'].split()
        x = (float(x1) + float(x2)) / 2
        y = (float(y1) + float(y2)) / 2
        self.position = self.transform((x, y))

    @property
    def bitmap_data(self):
        raise NotImplementedError


class Reference(IpeObject):
    def __init__(self, attrib, matrix=None):
        self.attrib = attrib
        super().__init__(matrix=matrix or attrib.get('matrix'))
        x, y = map(float, self.attrib.get('pos', '0 0').split())
        self.position = self.transform((x, y))

    def __repr__(self):
        name = self.attrib['name']
        name = re.sub(r'^mark/(.*)\(sx\)', r'\1', name)
        return '<Reference %s>' % name


class Group(IpeObject):
    def __init__(self, children, attrib):
        self.children = children
        self.attrib = attrib
        super().__init__(matrix=None)


parse_text = Text
parse_image = Image
parse_use = Reference
make_group = Group
