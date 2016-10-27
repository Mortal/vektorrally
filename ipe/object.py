class IpeObject:
    def __init__(self):
        self.layer = None


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
    def __init__(self, text, attrib):
        super().__init__()
        self.text = text
        self.attrib = attrib
        matrix_data = attrib.get('matrix')
        self.matrix = None if matrix_data is None else load_matrix(matrix_data)

        x, y = map(float, self.attrib['pos'].split())
        if self.matrix is None:
            self.position = (x, y)
        else:
            self.position = self.matrix.transform(x, y)

    @classmethod
    def parse(cls, text, attrib):
        return cls(text=text, attrib=attrib)


class Image(IpeObject):
    def __init__(self, bitmap_data, attrib):
        self.data = bitmap_data
        self.attrib = attrib

    @classmethod
    def parse(cls, bitmap_data, attrib):
        return cls(bitmap_data, attrib)

    @property
    def bitmap_data(self):
        raise NotImplementedError


class Reference(IpeObject):
    def __init__(self, attrib):
        self.attrib = attrib

    @classmethod
    def parse(cls, attrib):
        return cls(attrib)


class Group(IpeObject):
    def __init__(self, children, attrib):
        self.children = children
        self.attrib = attrib


parse_text = Text.parse
parse_image = Image.parse
parse_use = Reference.parse
make_group = Group
