import os
import argparse

from collections import namedtuple

import numpy as np

import ipe.file
from ipe.shape import Shape


State = namedtuple('State', 'pos vel'.split())


def unique(xs, ys):
    indices = np.lexsort((ys, xs))
    xs = xs[indices]
    ys = ys[indices]
    diff = np.concatenate(([True], (xs[1:] != xs[:-1]) & (ys[1:] != ys[:-1])))
    return xs[diff], ys[diff]


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
    initial = State((p + q) / 2, 0j)

    bfs_pos = [initial.pos]
    bfs_vel = [initial.vel]
    parent = {initial: initial}

    diff = np.array([-1-1j, -1j, 1-1j, -1, 0, 1, -1+1j, 1j, 1+1j])
    diff = diff.reshape((1, -1)) * args.grid

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
        bfs_pos, bfs_vel = unique(bfs_pos, bfs_vel)

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
