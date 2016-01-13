import os
import re
import argparse

from xml.etree import ElementTree

import numpy as np

import ipe.file


def strs_array(iterable):
    return np.array([float(x) for x in iterable])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid', '-g', type=int, default=16)
    parser.add_argument('filename')
    args = parser.parse_args()

    output_filename = '%s-solved%s' % tuple(os.path.splitext(args.filename))

    ipe_page = ipe.file.parse(args.filename)
    polygons = ipe_page.polygons
    if len(ipe_page.lines) != 1:
        raise Exception("Ipe file should have exactly one line segment")
    line = ipe_page.lines[0].endpoints()
    line = np.array([[line[0].real, line[0].imag],
                     [line[1].real, line[1].imag]])

    edges = [e for p in polygons for e in p.get_edges()]
    edge_p, edge_q = map(np.asarray, zip(*edges))
    edge_x1 = edge_p.real
    edge_x2 = edge_q.real
    edge_y1 = edge_p.imag
    edge_y2 = edge_q.imag
    edge_p = np.array([edge_x1, edge_y1]).T
    edge_q = np.array([edge_x2, edge_y2]).T

    def on_segment(p, q, r):
        """Given collinear p, q, r, does q lie on pr?"""
        p, q, r = np.asarray(p), np.asarray(q), np.asarray(r)
        s, d = p.shape[:-1], p.shape[-1]
        assert s == q.shape[:-1] == r.shape[:-1]
        assert d == q.shape[-1] == r.shape[-1]
        # p, q, r = p.reshape(-1, d), q.reshape(-1, d), r.reshape(-1, d)

        c1, c2 = np.minimum(p, r) <= q, q <= np.maximum(p, r)
        return (c1 & c2).all(axis=-1)  # .reshape(s)

    def orient(p, q, r):
        """1 => pqr is left turn"""
        p, q, r = np.asarray(p), np.asarray(q), np.asarray(r)
        s, d = p.shape[:-1], p.shape[-1]
        assert s == q.shape[:-1] == r.shape[:-1]
        assert 2 == d == q.shape[-1] == r.shape[-1]

        pq = q - p
        qr = r - q
        val = pq[..., 1] * qr[..., 0] - pq[..., 0] * qr[..., 1]
        return np.sign(val)

    def intersects(p1, q1, p2, q2):
        p1, q1 = np.asarray(p1), np.asarray(q1)
        p2, q2 = np.asarray(p2), np.asarray(q2)
        s, d = p1.shape[:-1], p1.shape[-1]
        assert s == q1.shape[:-1] == p2.shape[:-1] == q2.shape[:-1]
        assert 2 == d == q1.shape[-1] == p2.shape[-1] == q2.shape[-1]
        o1 = orient(p1, q1, p2)
        o2 = orient(p1, q1, q2)
        o3 = orient(p2, q2, p1)
        o4 = orient(p2, q2, q1)

        res = (o1 != o2) & (o3 != o4)
        s_ = (1,) if s == () else s
        res = res.reshape(s_)
        o1, o2, o3, o4 = np.asarray((o1, o2, o3, o4)).reshape((4,) + s_)
        res[o1 == 0] = on_segment(p1, p2, q1).reshape(s_)[o1 == 0]
        res[o2 == 0] = on_segment(p1, q2, q1).reshape(s_)[o2 == 0]
        res[o3 == 0] = on_segment(p2, p1, q2).reshape(s_)[o3 == 0]
        res[o4 == 0] = on_segment(p2, q1, q2).reshape(s_)[o4 == 0]

        return res.reshape(s)

    def intersects_edges(p, q):
        N = len(edge_p)
        r = intersects(edge_p, edge_q, np.tile(p, (N, 1)), np.tile(q, (N, 1)))
        print(r.shape)
        print(r.nonzero()[0])
        return r

    def valid(p, q):
        p = np.asarray(p)[:2]
        q = np.asarray(q)[:2]
        if (p == q).all():
            return True
        N = len(edge_p)
        pp = np.tile(p, (N, 1))
        qq = np.tile(q, (N, 1))
        r = intersects(edge_p, edge_q, pp, qq)
        if r.any():
            return False
        if intersects(p, q, line[0], line[1]).any():
            op = orient(line[0], line[1], p)
            oq = orient(line[0], line[1], q)
            if op >= oq:
                return False
        return True

    def win(p, q):
        p, q = np.asarray(p)[:2], np.asarray(q)[:2]
        if (p == q).all():
            return False
        if intersects(p, q, line[0], line[1]).any():
            op = orient(line[0], line[1], p)
            oq = orient(line[0], line[1], q)
            if -1 == op < oq:
                return True
        return False

    p, q = line
    initial_point = ((p + q) / 2).tolist()[0]
    initial = (initial_point.real, initial_point.imag, 0, 0)
    bfs = [initial]
    parent = {initial: initial}
    i = 0
    winner = None
    while i < len(bfs):
        u = bfs[i]
        i += 1
        if i % 100 == 0:
            print("head=%d tail=%d" % (i, len(bfs)))
        p = parent[u]
        if win(p, u):
            winner = u
            break
        px, py, vx, vy = u
        for dx in (-args.grid, 0, args.grid):
            for dy in (-args.grid, 0, args.grid):
                v = (px + vx + dx, py + vy + dy, vx + dx, vy + dy)
                if v not in parent and valid(u, v):
                    parent[v] = u
                    bfs.append(v)
    if winner:
        winpath = ElementTree.SubElement(page, 'path', stroke='red')
        directives = []
        u = winner
        while True:
            directives.append([u[0], u[1], 'l'])
            if u == parent[u]:
                break
            u = parent[u]
        directives[-1][-1] = 'm'
        directives.reverse()
        winpath.text = (
            '\n%s\n' % '\n'.join(' '.join(tuple(map(str, p))) for p in directives))
        winpath.tail = '\n'
    root.set('creator', 'vektorrally.py')
    with open(output_filename, 'wb') as fp:
        fp.write(b'<?xml version="1.0"?>\n<!DOCTYPE ipe SYSTEM "ipe.dtd">\n')
        fp.write(ElementTree.tostring(root))


if __name__ == "__main__":
    main()
