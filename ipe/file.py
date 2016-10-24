from xml.etree import ElementTree as ET

import ipe.shape
from ipe.object import (
    parse_text, parse_image, parse_use, make_group, Text, Group,
)


class IpeBitmap:
    def __init__(self, element):
        self.element = element


class View:
    def __init__(self, layers, active, marked):
        self.layers = layers
        self.active = active
        self.marked = marked

    @property
    def standard(self):
        return (self.layers == ['alpha'] and
                self.active == 'alpha' and
                not self.marked)

    def __str__(self):
        marked = ' marked' if self.marked else ''
        if len(self.layers) == 1 and self.active == self.layers[0]:
            if self.layers == ['alpha']:
                return '<View%s>' % marked
            else:
                return '<View [%s]%s>' % (self.layers[0], marked)
        else:
            return '<View [%s] active=%s%s>' % (
                ' '.join(self.layers),
                self.active,
                marked)


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
                self.views.append(View(
                    layers=child.attrib['layers'].split(),
                    active=child.attrib['active'],
                    marked=child.get('marked') == 'yes'))
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
        if not self.views:
            self.views.append(View(
                layers=list(self.layers),
                active=self.layers[0],
                marked=False))

    def __repr__(self):
        return '<IpePage: %s>' % self.summary

    def copy(self):
        from copy import deepcopy

        return deepcopy(self)

    @property
    def summary(self):
        t = next(
            (t for view_texts in self.text_strings() for t in view_texts),
            None)
        return '%d objects%s%s%s' % (
            len(self.objects),
            ', %d views' % len(self.views) if len(self.views) > 1 else '',
            ', %d layers' % len(self.layers) if len(self.layers) > 1 else '',
            ', "%s"' % t if t else '')

    def describe(self, indent=''):
        yield indent + self.summary
        view_data = zip(self.views, self.text_strings())
        for i, (view, texts) in enumerate(view_data):
            if len(self.views) != 1 or not view.standard:
                yield indent + 'View %s: %s' % (i, view)
            for t in texts[:1]:
                t_abbr = t if len(t) < 70 else (t[:67] + '...')
                yield indent + '- %s' % t_abbr

    def itertexts(self):
        def walk(l, g):
            for o in l:
                if isinstance(o, Text):
                    yield (g or o.layer, o)
                elif isinstance(o, Group):
                    yield from walk(o.children, g or o.layer)

        return walk(self.objects, None)

    def text_strings(self):
        layer_occurrence = {}
        for i, v in enumerate(self.views):
            for l in v.layers:
                layer_occurrence.setdefault(l, i)
        texts_occurrence = {}
        for g, t in self.itertexts():
            try:
                i = layer_occurrence[g]
            except KeyError:
                continue
            else:
                texts_occurrence.setdefault(i, []).append(t)
        for i, ts in texts_occurrence.items():
            ts.sort(key=lambda t: -t.position[1])
            ts[:] = [' '.join(w for w in t.text.split()
                              if not w.startswith('\\'))
                     for t in ts]
            ts[:] = [t for t in ts if t]
        return [texts_occurrence.get(i, [])
                for i, l in enumerate(self.views)]

    def prune_objects(self):
        "Remove objects in layers that are invisible."
        visible_layers = set.union(set(v['layers']) for v in self.views)
        self.objects = [
            o for o in self.objects
            if o.layer in visible_layers]

    def prune(self):
        "Remove layers and views that contain no objects."
        nonempty_layers = set(o.layer for o in self.objects)
        if nonempty_layers:
            print("prune: No objects in page, aborting")
            return
        self.layers = [l for l in self.layers if l in nonempty_layers]
        for v in self.views:
            v.layers = [l for l in self.layers if l in nonempty_layers]
            if v.active not in nonempty_layers:
                if v.layers:
                    v.active = v.layers[0]
                else:
                    v.active = next(iter(nonempty_layers))

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
        path = ET.SubElement(self.page_element, 'path', **kwargs)
        path.text = '\n' + shape.to_ipe_path()
        path.tail = '\n'

    def add_text(self, text, pos):
        pos_str = '%g %g' % (pos.real, pos.imag)
        t = ET.SubElement(self.page_element, 'text',
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
        self.style_names = self.extract_style_names(
            self.root.find('ipestyle'))
        self.pages = [
            IpePage(el, self)
            for el in self.root.findall('page')
        ]

    def save(self, filename):
        self.root.set('creator', 'vektorrally.py')
        children = list(self.root)
        page_children = [c for c in children if c.tag == 'page']
        for c in page_children:
            self.root.remove(c)
        for p in self.pages:
            self.root.append(p.page_element)
        with open(filename, 'w') as fp:
            fp.write(
                '<?xml version="1.0"?>\n' +
                '<!DOCTYPE ipe SYSTEM "ipe.dtd">\n')
            fp.write(ET.tostring(self.root, encoding='unicode'))

    @staticmethod
    def extract_style_names(ipestyle_element):
        res = {}
        for child in ipestyle_element:
            try:
                name = child.attrib['name']
            except KeyError:
                continue
            res.setdefault(child.tag, set()).add(name)
        return res

    def __repr__(self):
        return "<IpeDoc: %d pages %s>" % (len(self.pages), self.pages)

    def describe(self):
        for i, page in enumerate(self.pages):
            yield 'Page %d:' % i
            yield from page.describe('  ')


def parse(filename):
    return IpeDoc(ET.parse(filename))
