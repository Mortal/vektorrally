import os
import argparse

from collections import namedtuple

import numpy as np

import ipe.file
from ipe.shape import Shape

from vektorrally_util import unique_pairs, orient, intersects, linrange


class Map:
    def __init__(self, ipe_page, grid_size):
        self.ipe_page = ipe_page

        obstacle_edges = [e for p in ipe_page.polygons for e in p.get_edges()]
        self.obstacle_p, self.obstacle_q = map(
            np.asarray, zip(*obstacle_edges))

        lines = list(ipe_page.lines)
        if len(lines) != 1:
            raise Exception("Ipe file should have exactly one line segment")

        p, q = self.goal = lines[0].endpoints()
        self.grid_size = g = grid_size
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

        diff = np.array([-1-1j, -1j, 1-1j, -1, 0, 1, -1+1j, 1j, 1+1j])
        diff = diff * grid_size
        self.diff = diff.tolist()

        self.initials = [
            i for i in initials.tolist()
            if self.valid([i] * len(diff), i + diff).any()
        ]

    def valid(self, p, q):
        p, q = np.asarray(p), np.asarray(q)
        assert p.shape == q.shape
        if p.shape == ():
            return self.valid([p], [q])[0]
        i = intersects(p, q, self.goal[0], self.goal[1])
        op = orient(self.goal[0], self.goal[1], p)
        oq = orient(self.goal[0], self.goal[1], q)

        r = np.ones(p.shape, dtype=np.bool)
        r[i & (op >= oq)] = False

        i2 = intersects(self.obstacle_p, self.obstacle_q, p, q)
        i2 = i2.any(axis=tuple(range(self.obstacle_p.ndim)))
        r[i2] = False
        return r

    def win(self, p, q):
        p, q = np.asarray(p), np.asarray(q)
        assert p.shape == q.shape
        if p.shape == ():
            return self.win([p], [q])[0]

        r = np.zeros(p.shape, dtype=np.bool)

        i = intersects(p, q, self.goal[0], self.goal[1])
        op = orient(self.goal[0], self.goal[1], p)
        oq = orient(self.goal[0], self.goal[1], q)
        r[i & (-1 == op) & (op < oq)] = True

        r[p == q] = False
        return r


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
    m = Map(ipe_page, args.grid)

    winner = solve_2d(m, ipe_page)
    if winner:
        ipe_page.add_shape(Shape.make_polyline(winner), stroke='red')
        ipe_doc.save(output_filename)


def solve_2d(m, ipe_page):
    bfs = list(m.initials)
    dists = {v: 0 for v in bfs}

    def dist(v):
        return dists.get(v, float('inf'))

    i = 0
    winner = None
    while i < len(bfs):
        u = bfs[i]
        d_u = dists[u]
        i += 1
        for d in m.diff:
            if m.valid(u - d, u):
                if d_u + 1 < dist(u - d):
                    dists[u - d] = d_u + 1
                    bfs.append(u - d)

    first_edges = [
        (initial, initial + d)
        for initial in m.initials
        for d in m.diff
    ]
    valid_edges = [(u, v) for u, v in first_edges if m.valid(u, v)]
    useful_edges = [(u, v) for u, v in valid_edges if v in dists]
    if not useful_edges:
        print("Could not find path")
        return
    best_edge = min(
        range(len(useful_edges)),
        key=lambda i: dists[useful_edges[i][1]])
    points = [useful_edges[best_edge][0], useful_edges[best_edge][1]]
    while dists[points[-1]] > 0:
        points.append(
            min((points[-1] + d for d in m.diff
                 if m.valid(points[-1], points[-1] + d)), key=dist))
    for v, d in dists.items():
        ipe_page.add_text(r'\dist{%g}' % d, v)
    return points


def solve_numpy_bfs(m):
    State = namedtuple('State', 'pos vel'.split())

    valid = m.valid
    win = m.win
    diff = np.array(m.diff).reshape(1, -1)

    bfs_pos = [i for i in m.initials]
    bfs_vel = [0j for i in m.initials]
    parent = {State(i, 0j): State(i, 0j) for i in m.initials}

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
        return points
    else:
        print("Could not solve")
        for u, v in sorted(parent.items())[:100]:
            print("parent(%s) = %s" % (u, v))
        return None


def solve_list_bfs(m):
    State = namedtuple('State', 'pos vel'.split())
    bfs = [State(i, 0j) for i in m.initials]
    parent = {State(i, 0j): State(i, 0j) for i in m.initials}
    i = 0
    winner = None
    while i < len(bfs):
        u = bfs[i]
        i += 1
        if i % 100 == 0:
            print("head=%d tail=%d" % (i, len(bfs)))
        p = parent[u]
        if m.win(p.pos, u.pos):
            winner = u
            break
        for d in m.diff:
            v = State(u.pos + u.vel + d, u.vel + d)
            if v not in parent and m.valid(u.pos, v.pos):
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
        return points


if __name__ == "__main__":
    main()
