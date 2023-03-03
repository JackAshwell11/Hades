from __future__ import annotations

from typing import NamedTuple
from heapq import heappop, heappush


class Point(NamedTuple):
    x: int
    y: int


class Rect(NamedTuple):
    top_left: Point
    bottom_right: Point
    name: str

    def center_x(self) -> int:
        return round((self.top_left.x + self.bottom_right.x) / 2)

    def center_y(self) -> int:
        return round((self.top_left.y + self.bottom_right.y) / 2)

    def get_distance_to(self, other: Rect) -> int:
        return max(
            abs(self.center_x() - other.center_x()), abs(self.center_y() - other.center_y())
        )


class Edge(NamedTuple):
    cost: float
    source: Rect
    destination: Rect

    def __repr__(self) -> str:
        return f"{self.cost} - {self.source.name} - {self.destination.name}"



def create_connections(complete_graph: dict[Rect, list[Rect]]) -> set[Edge]:
    start = next(iter(complete_graph))
    unexplored: list[Edge] = [Edge(0, start, start)]
    visited: set[Rect] = set()
    mst: set[Edge] = set()
    while len(mst) < len(complete_graph) and unexplored:
        lowest: Edge = heappop(unexplored)

        if lowest.destination in visited:
            continue

        visited.add(lowest.destination)
        for neighbour in complete_graph[lowest.destination]:
            if neighbour not in visited:
                heappush(unexplored, Edge(lowest.destination.get_distance_to(neighbour), lowest.destination, neighbour))

        print(f"{lowest.cost}  {lowest.source.top_left.x} {lowest.source.top_left.y} {lowest.source.bottom_right.x} {lowest.source.bottom_right.y}  {lowest.destination.top_left.x} {lowest.destination.top_left.y} {lowest.destination.bottom_right.x} {lowest.destination.bottom_right.y}")
        if lowest.source != lowest.destination:
            print("add")
            mst.add(lowest)

    return mst


def test_map_create_connections_valid():
    valid_rect_one = Rect(Point(3, 3), Point(5, 7), "valid 1")
    valid_rect_two = Rect(Point(3, 5), Point(4, 0), "valid 2")
    temp_rect_one = Rect(Point(0, 0), Point(3, 3), "temp 1")
    temp_rect_two = Rect(Point(10, 10), Point(12, 12), "temp 2")
    print()
    f = create_connections({
        temp_rect_one: [valid_rect_one, valid_rect_two, temp_rect_two],
        valid_rect_one: [valid_rect_two, temp_rect_one, temp_rect_two],
        valid_rect_two: [valid_rect_one, temp_rect_one, temp_rect_two],
        temp_rect_two: [valid_rect_one, valid_rect_two, temp_rect_one],
    })
    assert f == {
        Edge(2, temp_rect_one, valid_rect_two), Edge(4, valid_rect_one, temp_rect_one), Edge(7, valid_rect_one, temp_rect_two)
    }




# v1 = Rect(Point(0, 0), Point(0, 0), "v1")
# v2 = Rect(Point(1, 0), Point(1, 0), "v2")
# v3 = Rect(Point(0, 3), Point(0, 3), "v3")
# v4 = Rect(Point(3, 0), Point(3, 0), "v4")
# v5 = Rect(Point(1, 7), Point(1, 7), "v5")
# v6 = Rect(Point(3, 4), Point(3, 4), "v6")
# print(create_connections(
#     {
#         v1: [v2, v4],
#         v2: [v3, v5, v6],
#         v3: [v2, v6],
#         v4: [v1],
#         v5: [v2],
#         v6: [v2, v3]
#     }
# ))
