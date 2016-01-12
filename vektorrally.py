import re

from xml.etree import ElementTree

import numpy as np


def strs_array(iterable):
    return np.array([float(x) for x in iterable])


def main():
    tree = ElementTree.parse('bane.ipe')
    root = tree.getroot()
    page = root.find('page')
    boundaries = []
    line = None
    for p in page.findall('path'):
        path = ' '.join(p.text.split())
        if path.endswith(' l'):
            # Line
            mo = re.match(r'^(\S+) (\S+) m (\S+) (\S+) l$', path)
            if mo is None:
                raise Exception("Not a line")
            p1 = strs_array([mo.group(1), mo.group(2)])
            q1 = strs_array([mo.group(3), mo.group(4)])
            line = [p1, q1]
        elif path.endswith(' h'):
            # Polygon
            mo = re.match('(\S+ \S+ m (\S+ \S+ l )*)h', path)
            if mo is None:
                raise Exception("Not a polygon")
            data = mo.group(1).split()
            xs = strs_array(data[::3])
            ys = strs_array(data[1::3])
            boundaries.append((xs, ys))

    edge_x1 = np.concatenate([b[0] for b in boundaries])
    edge_x2 = np.concatenate([np.roll(b[0], -1) for b in boundaries])
    edge_y1 = np.concatenate([b[1] for b in boundaries])
    edge_y2 = np.concatenate([np.roll(b[1], -1) for b in boundaries])
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
        """-1 => pqr is left turn"""
        p, q, r = np.asarray(p), np.asarray(q), np.asarray(r)
        s, d = p.shape[:-1], p.shape[-1]
        assert s == q.shape[:-1] == r.shape[:-1]
        assert 2 == d == q.shape[-1] == r.shape[-1]

        pq = q - p
        qr = r - q
        val = pq[..., 1] * qr[..., 0] - pq[..., 0] * qr[..., 1]
        return np.sign(val)

    def intersects(p1, q1, p2, q2):
        o1 = orient(p1, q1, p2)
        o2 = orient(p1, q1, q2)
        o3 = orient(p2, q2, p1)
        o4 = orient(p2, q2, q1)

        res = (o1 != o2) & (o3 != o4)
        res[o1 == 0] = on_segment(p1, p2, q1)[o1 == 0]
        res[o2 == 0] = on_segment(p1, q2, q1)[o2 == 0]
        res[o3 == 0] = on_segment(p2, p1, q2)[o3 == 0]
        res[o4 == 0] = on_segment(p2, q1, q2)[o4 == 0]

        return res

    def intersects_edges(p, q):
        N = len(edge_p)
        r = intersects(edge_p, edge_q, np.tile(p, (N, 1)), np.tile(q, (N, 1)))
        print(r.shape)
        print(r.nonzero()[0])
        return r

    def valid(p, q):
        N = len(edge_p)
        pp = np.tile(p, (N, 1))
        qq = np.tile(q, (N, 1))
        r = intersects(edge_p, edge_q, pp, qq)
        if r.any():
            return False
        if intersects(p, q, line[0], line[1]).any():
            op = orient(line[0], line[1], p)
            oq = orient(line[0], line[1], q)
            if op < oq:
                return False
        return True

    def win(p, q):
        if (p == q).all():
            return False
        if intersects(p, q, line[0], line[1]).any():
            op = orient(line[0], line[1], p)
            oq = orient(line[0], line[1], q)
            if op < oq == 1:
                return True
        return False

    p, q = line
    i_x, i_y = (p + q) / 2
    initial = np.array([i_x, i_y, 0, 0])
    bfs = [initial]
    parent = {initial: initial}
    i = 0
    while i < len(bfs):
        u = bfs[i][:2]
        px, py, vx, vy = bfs[i]
        if win(parent[bfs[i]][:2], bfs[i][:2]):
            print(u)
            break
        i += 1
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                v = np.array([px + vx + dx, py + vy + dy, vx + dx, vy + dy])
                if v not in parent and valid(u, v[:2]):
                    parent[v] = bfs[i]
                    bfs.append(v)


if __name__ == "__main__":
    main()
