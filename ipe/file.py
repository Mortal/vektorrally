from xml.etree import ElementTree

import ipe.shape
from ipe.object import parse_text, parse_image, parse_use, make_group


class IpeBitmap:
    def __init__(self, element):
        self.element = element


class IpePage:
    def __init__(self, page, document):
        # ipeiml.cpp ImlParser::parsePage
        self.document = document
        self.page_element = page
        self.objects = []
        self.title = page.get('title', '')
        self.section = page.get('section', '')
        self.subsection = page.get('subsection', '')
        self.marked = page.get('marked') == 'no'
        self.layers = []
        self.views = []
        self.current_layer = None

        for child in page:
            if child.tag == 'notes':
                pass
            elif child.tag == 'layer':
                self.layers.append(child.attrib['name'])
            elif child.tag == 'view':
                self.views.append({
                    'layers': child.attrib['layers'].split(),
                    'active': child.attrib['active'],
                    'marked': child.get('marked') == 'yes'})
            else:
                self.current_layer = child.get('layer', self.current_layer)
                if self.current_layer is None:
                    if not self.layers:
                        self.layers.append('alpha')
                    self.current_layer = self.layers[0]
                # ipefactory.cpp createObject
                object = self.parse_object(child)
                object.layer = self.current_layer
                self.objects.append(object)

    @property
    def lines(self):
        return (o for o in self.objects if o.is_line_segment())

    @property
    def polygons(self):
        return (o for o in self.objects if o.is_polygon())

    def parse_object(self, child):
        if child.tag == 'path':
            return ipe.shape.load_shape(child.text, child.attrib)
        elif child.tag == 'text':
            return parse_text(child.text, child.attrib)
        elif child.tag == 'image':
            if child.get('bitmap'):
                bitmap_id = int(child.attrib['bitmap'])
                bitmap = self.document.bitmaps[bitmap_id]
                return parse_image(bitmap, child.attrib)
            else:
                return parse_image(child.text, child.attrib)
        elif child.tag == 'use':
            return parse_use(child.attrib)
        elif child.tag == 'group':
            return make_group([self.parse_object(c) for c in child],
                              child.attrib)
        else:
            raise Exception("Unknown tag %s" % (child,))

    def add_shape(self, shape, **kwargs):
        path = ElementTree.SubElement(self.page_element, 'path', **kwargs)
        path.text = '\n' + shape.to_ipe_path()
        path.tail = '\n'

    def add_text(self, text, pos):
        pos_str = '%g %g' % (pos.real, pos.imag)
        t = ElementTree.SubElement(self.page_element, 'text',
            pos=pos_str,
            type='label',
            halign='center',
            valign='center')
        t.text = text
        t.tail = '\n'


class IpeDoc:
    def __init__(self, tree):
        self.root = tree.getroot()
        self.bitmaps = {
            int(el.get('id')): IpeBitmap(el)
            for el in self.root.findall('bitmap')
        }
        self.pages = [
            IpePage(el, self)
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
