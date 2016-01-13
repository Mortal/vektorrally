The Ipe input file should contain two polygons, one inside the other,
and a line segment (the start/finish line) from the inner to the outer.

The vektorrally.py script then tries to find a valid route,
starting on the right side of the start line
and finishing on the left side of the start line,
whilst avoiding the obstacle polygons.
If found, the shortest route is written to `foo-solved.ipe`
(for an input file named `foo.ipe`).

For more information on the Vector rally game itself, see e.g.
https://en.wikipedia.org/wiki/Vector_rally
