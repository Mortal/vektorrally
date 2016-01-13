from xml.etree import ElementTree

import ipe.shape


class IpePage:
    def __init__(self, tree):
        self.root = tree.getroot()
        page = self.page
        self.shapes = []
        for child in page:
            if child.tag in ('layer', 'view'):
                continue
            elif child.tag == 'use':
                # marker (skip)
                continue
            elif child.tag == 'image':
                # image (skip)
                continue
            elif child.tag == 'path':
                self.shapes.append(
                    ipe.shape.load_shape(child.text, child.get('matrix')))
            else:
                raise Exception("Unknown tag %s" % (child,))

        self.lines = []
        self.polygons = []
        for s in self.shapes:
            if s.is_line_segment():
                self.lines.append(s)
            elif s.is_polygon():
                self.polygons.append(s)

    @property
    def page(self):
        pages = self.root.findall('page')
        assert pages, 'root must contain exactly one <page>'
        if len(pages) > 1:
            raise Exception("Multiple pages")
        return pages[0]


def parse(filename):
    return IpePage(ElementTree.parse(filename))
