import os
import argparse

from collections import namedtuple

import numpy as np

import ipe.file
from ipe.shape import Shape

from vektorrally_util import unique_pairs, orient, intersects, linrange


State = namedtuple('State', 'pos vel'.split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid', '-g', type=int, default=16)
    parser.add_argument('--overwrite', '-f', action='store_true')
    parser.add_argument('filename')
    args = parser.parse_args()

    if args.overwrite:
        output_filename = args.filename
    else:
        output_filename = (
            '%s-solved%s' % tuple(os.path.splitext(args.filename)))

    ipe_doc = ipe.file.parse(args.filename)
    if len(ipe_doc.pages) != 1:
        raise Exception("Ipe file should have exactly one page")
    ipe_page = ipe_doc.pages[0]

    polygons = list(ipe_page.polygons)
    lines = list(ipe_page.lines)
    if len(lines) != 1:
        raise Exception("Ipe file should have exactly one line segment")
    line = lines[0].endpoints()

    edges = [e for p in polygons for e in p.get_edges()]
    edge_p, edge_q = map(np.asarray, zip(*edges))

    def valid(p, q):
        p, q = np.asarray(p), np.asarray(q)
        assert p.shape == q.shape
        if p.shape == ():
            return valid([p], [q])[0]
        i = intersects(p, q, line[0], line[1])
        op = orient(line[0], line[1], p)
        oq = orient(line[0], line[1], q)

        r = np.ones(p.shape, dtype=np.bool)
        r[i & (op >= oq)] = False

        i2 = intersects(edge_p, edge_q, p, q)
        i2 = i2.any(axis=tuple(range(edge_p.ndim)))
        r[i2] = False
        return r

    def win(p, q):
        p, q = np.asarray(p), np.asarray(q)
        assert p.shape == q.shape
        if p.shape == ():
            return win([p], [q])[0]

        r = np.zeros(p.shape, dtype=np.bool)

        i = intersects(p, q, line[0], line[1])
        op = orient(line[0], line[1], p)
        oq = orient(line[0], line[1], q)
        r[i & (-1 == op) & (op < oq)] = True

        r[p == q] = False
        return r

    p, q = line
    g = args.grid
    if p.real % g or p.imag % g or q.real % g or q.imag % g:
        raise Exception(
            "The start/finish line should lie on a %s pt boundary" % g)
    if p.real != q.real and p.imag != q.imag:
        raise Exception(
            "The start/finish line should be vertical or horizontal")
    if p.real == q.real:
        initials = p.real + 1j * linrange(p.imag, q.imag + g, g)
    else:
        initials = 1j * p.imag + linrange(p.real, q.real + g, g)

    bfs_pos = [i for i in initials]
    bfs_vel = [0j for i in initials]
    parent = {State(i, 0j): State(i, 0j) for i in initials}

    diff = np.array([-1-1j, -1j, 1-1j, -1, 0, 1, -1+1j, 1j, 1+1j])
    diff = diff.reshape((1, -1)) * g

    winner = None
    dist = 0
    while len(bfs_pos) > 0:
        dist += 1
        print(dist, len(bfs_pos))
        p_pos = np.array(
            [parent[State(p, v)].pos for p, v in zip(bfs_pos, bfs_vel)])
        u_pos = np.asarray(bfs_pos).reshape((-1, 1))
        u_vel = np.asarray(bfs_vel).reshape((-1, 1))
        v_pos = u_pos + u_vel + diff
        v_vel = u_vel + diff

        w = win(p_pos, u_pos.ravel()).nonzero()[0]
        if len(w) > 0:
            winner = State(u_pos[w[0], 0], u_vel[w[0], 0])
            break

        m1 = np.array(
            [[State(p, v) not in parent for p, v in zip(ps, vs)]
             for ps, vs in zip(v_pos, v_vel)])
        m2 = valid(np.repeat(u_pos, diff.shape[1], 1), v_pos)
        print("%s neighbors, %s not visited, %s valid, %s" %
              (np.product(m1.shape), m1.sum(), m2.sum(), (m1 & m2).sum()))
        for i, j in zip(*(m1 & m2).nonzero()):
            s = State(v_pos[i, j], v_vel[i, j])
            parent[s] = State(u_pos[i, 0], u_vel[i, 0])
        print("%s visited" % len(parent))
        bfs_pos = v_pos[m1 & m2].ravel()
        bfs_vel = v_vel[m1 & m2].ravel()
        bfs_pos, bfs_vel = unique_pairs(bfs_pos, bfs_vel)

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
        ipe_doc.save(output_filename)
    else:
        print("Could not solve")
        for u, v in sorted(parent.items())[:100]:
            print("parent(%s) = %s" % (u, v))


if __name__ == "__main__":
    main()
