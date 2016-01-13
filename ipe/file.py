from xml.etree import ElementTree

import ipe.shape


class IpePage:
    def __init__(self, page):
        # ipeiml.cpp ImlParser::parsePage
        self.page = page
        self.objects = []

        for child in page:
            if child.tag == 'notes':
                pass
            elif child.tag == 'layer':
                pass
            elif child.tag == 'view':
                pass
            else:
                # ipefactory.cpp createObject
                self.parse_object(child)

    @property
    def lines(self):
        return (o for o in self.objects if o.is_line_segment())

    @property
    def polygons(self):
        return (o for o in self.objects if o.is_polygon())

    def parse_object(self, child):
        if child.tag == 'path':
            self.objects.append(
                ipe.shape.load_shape(child.text, child.attrib))
        elif child.tag == 'text':
            return  # skip
        elif child.tag == 'image':
            return  # skip
        elif child.tag == 'use':
            return  # skip
        elif child.tag == 'group':
            for c in child:
                self.parse_object(c)
        else:
            raise Exception("Unknown tag %s" % (child,))

    def add_shape(self, shape, **kwargs):
        path = ElementTree.SubElement(self.page, 'path', **kwargs)
        path.text = '\n' + shape.to_ipe_path()
        path.tail = '\n'


class IpeDoc:
    def __init__(self, tree):
        self.root = tree.getroot()
        self.pages = [
            IpePage(el)
            for el in self.root.findall('page')
        ]

    def save(self, filename):
        self.root.set('creator', 'vektorrally.py')
        with open(filename, 'wb') as fp:
            fp.write(
                b'<?xml version="1.0"?>\n' +
                b'<!DOCTYPE ipe SYSTEM "ipe.dtd">\n')
            fp.write(ElementTree.tostring(self.root))


def parse(filename):
    return IpeDoc(ElementTree.parse(filename))