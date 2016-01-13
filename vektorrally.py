import os
import argparse

from collections import namedtuple

import numpy as np

import ipe.file
from ipe.shape import Shape


State = namedtuple('State', 'pos vel'.split())


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

    edges = [e for p in polygons for e in p.get_edges()]
    edge_p, edge_q = map(np.asarray, zip(*edges))

    def on_segment(p, q, r):
        """Given collinear p, q, r, does q lie on pr?"""
        p, q, r = np.asarray(p), np.asarray(q), np.asarray(r)
        s1, s2 = p.shape, q.shape
        n1, n2 = np.product(s1), np.product(s2)
        assert p.shape == r.shape
        p, r = p.reshape(n1, 1), r.reshape(n1, 1)
        q = q.reshape(1, n2)
        c1 = np.minimum(p.real, r.real) <= q.real
        c2 = q.real <= np.maximum(p.real, r.real)
        c3 = np.minimum(p.imag, r.imag) <= q.imag
        c4 = q.imag <= np.maximum(p.imag, r.imag)
        return (c1 & c2 & c3 & c4).reshape(s1 + s2)

    def orient(p, q, r):
        """1 => pqr is left turn"""
        p, q, r = np.asarray(p), np.asarray(q), np.asarray(r)
        s1, s2 = p.shape, r.shape
        n1, n2 = np.product(s1), np.product(s2)
        assert p.shape == q.shape
        pq = (q - p).reshape((n1, 1))
        qr = r.reshape((1, n2)) - q.reshape((n1, 1))
        # return np.sign((qr / pq).imag)
        return np.sign(pq.imag * qr.real - pq.real * qr.imag).reshape(s1 + s2)

    def intersects(p1, q1, p2, q2):
        p1, q1 = np.asarray(p1), np.asarray(q1)
        p2, q2 = np.asarray(p2), np.asarray(q2)
        s1, s2 = p1.shape, p2.shape
        n1, n2 = np.product(s1), np.product(s2)
        assert p1.shape == q1.shape
        assert p2.shape == q2.shape
        p1, q1 = p1.ravel(), q1.ravel()
        p2, q2 = p2.ravel(), q2.ravel()
        o1 = orient(p1, q1, p2)
        o2 = orient(p1, q1, q2)
        o3 = orient(p2, q2, p1).T
        o4 = orient(p2, q2, q1).T

        res = (o1 != o2) & (o3 != o4)
        res = res.reshape(n1, n2)
        o1, o2, o3, o4 = np.asarray((o1, o2, o3, o4)).reshape(4, n1, n2)
        res[o1 == 0] = on_segment(p1, p2, q1).reshape(n1, n2)[o1 == 0]
        res[o2 == 0] = on_segment(p1, q2, q1).reshape(n1, n2)[o2 == 0]
        res[o3 == 0] = on_segment(p2, p1, q2).reshape(n1, n2)[o3 == 0]
        res[o4 == 0] = on_segment(p2, q1, q2).reshape(n1, n2)[o4 == 0]
        return res.reshape(s1 + s2)

    def valid(p, q):
        if p.pos == q.pos:
            return True
        r = intersects(edge_p, edge_q, p.pos, q.pos)
        if r.any():
            return False
        if intersects(p.pos, q.pos, line[0], line[1]):
            op = orient(line[0], line[1], p.pos)
            oq = orient(line[0], line[1], q.pos)
            if op >= oq:
                return False
        return True

    def win(p, q):
        if p == q:
            return False
        if intersects(p, q, line[0], line[1]).any():
            op = orient(line[0], line[1], p)
            oq = orient(line[0], line[1], q)
            if -1 == op < oq:
                return True
        return False

    p, q = line
    initial = State((p + q) / 2, 0j)
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
        if win(p.pos, u.pos):
            winner = u
            break
        for d in (-1-1j, -1j, 1-1j, -1, 0, 1, -1+1j, 1j, 1+1j):
            v = State(u.pos + u.vel + args.grid * d, u.vel + args.grid * d)
            if v not in parent and valid(u, v):
                parent[v] = u
                bfs.append(v)
    if winner:
        points = []
        u = winner
        while True:
            points.append(u[0])
            if u == parent[u]:
                break
            u = parent[u]
        points.reverse()
        ipe_page.add_shape(Shape.make_polyline(points), stroke='red')
        ipe_page.save(output_filename)


if __name__ == "__main__":
    main()
