from xml.etree import ElementTree

import ipe.shape


def parse(filename):
    tree = ElementTree.parse(args.filename)
    root = tree.getroot()
    page = root.findall('page')
    assert page, 'root must contain exactly one <page>'
    if len(page) > 1:
        raise Exception("Multiple pages")
    for child in page:
        if child.tag in ('layer', 'view'):
            continue
        if child.tag == 'use':
            # marker -- skip
            continue
