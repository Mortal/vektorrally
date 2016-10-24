class IpeObject:
    def __init__(self):
        self.layer = None


class Text(IpeObject):
    def __init__(self, text, attrib):
        super().__init__()
        self.text = text
        self.attrib = attrib

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
