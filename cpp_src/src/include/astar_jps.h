// Std includes
#include <iostream>
#include <optional>
#include <queue>

// Custom includes
#include "primitives.h"

// ----- CONSTANTS ------------------------------
/* Represents the north, south, east, west, north-east, north-west, south-east
 * and south-west directions on a compass */
const std::vector<Point> INTERCARDINAL_OFFSETS = {
    {-1, -1}, {0, -1}, {1, -1}, {-1, 0}, {1, 0}, {-1, 1}, {0, 1}, {1, 1},
};

// ----- STRUCTURES ------------------------------
/// Represents a grid position and its costs from the start position
///
/// Parameters
/// ----------
/// cost - The cost to traverse to this neighbour.
/// pair - The position in the grid.
struct Neighbour {
  int cost;
  Point pair;
  Point parent;

  inline bool operator<(const Neighbour nghbr) const {
    // The priority_queue data structure gets the maximum priority, so we need
    // to override that functionality to get the minimum priority
    return cost > nghbr.cost;
  }
};

// ----- FUNCTIONS ------------------------------
inline bool walkable(std::vector<std::vector<TileType>> &grid, int x, int y) {
  return ((0 <= x) && (x < grid[0].size()) && (0 <= y) && (y < grid.size()) &&
          (grid[y][x] != TileType::Obstacle));
}

std::optional<Point> jump(std::vector<std::vector<TileType>> &grid,
                          Point current, Point &parent, Point &end) {
  if (!walkable(grid, current.x, current.y)) {
    return std::nullopt;
  }

  if (current == end) {
    return current;
  }

  int dx = current.x - parent.x, dy = current.y - parent.y;
  if (dx != 0 && dy != 0) {
    if ((!walkable(grid, current.x - dx, current.y) &&
         walkable(grid, current.x - dx, current.y + dy)) or
        (!walkable(grid, current.x, current.y - dy) &&
         walkable(grid, current.x + dx, current.y - dy))) {
      return current;
    }

    if (jump(grid, Point{current.x + dx, current.y}, current, end) or
        jump(grid, Point{current.x, current.y + dy}, current, end)) {
      return current;
    }
  } else if (dx != 0) {
    if ((!walkable(grid, current.x, current.y - 1) &&
         walkable(grid, current.x + dx, current.y - 1)) or
        (!walkable(grid, current.x, current.y + 1) &&
         walkable(grid, current.x + dx, current.y + 1))) {
      return current;
    }
  } else {
    if ((!walkable(grid, current.x - 1, current.y) &&
         walkable(grid, current.x - 1, current.y - dy)) or
        (!walkable(grid, current.x + 1, current.y) &&
         walkable(grid, current.x + 1, current.y - dy))) {
      return current;
    }
  }

  if (walkable(grid, current.x + dx, current.y) or
      walkable(grid, current.x, current.y + dy)) {
    return jump(grid, Point{current.x + dx, current.y + dy}, current, end);
  }
}

std::vector<Point> prune_neighbours(Point &current, Point &parent) {
  std::vector<Point> neighbours;
  int dx = (current.x - parent.x) / std::max(std::abs(current.x - parent.x), 1),
      dy = (current.y - parent.y) / std::max(std::abs(current.y - parent.y), 1);

  if (dx != 0 && dy != 0) {
    neighbours.emplace_back(current.x + dx, current.y);
    neighbours.emplace_back(current.x, current.y + dy);
    neighbours.emplace_back(current.x + dx, current.y + dy);

    // not sure about these
    neighbours.emplace_back(current.x + dx, current.y - dy);
    neighbours.emplace_back(current.x - dx, current.y + dy);
  } else if (dx != 0) {
    neighbours.emplace_back(current.x + dx, current.y - 1);
    neighbours.emplace_back(current.x + dx, current.y);
    neighbours.emplace_back(current.x + dx, current.y + 1);
  } else if (dy != 0) {
    neighbours.emplace_back(current.x - 1, current.y + dy);
    neighbours.emplace_back(current.x, current.y + dy);
    neighbours.emplace_back(current.x + 1, current.y + dy);
  } else {
    std::for_each(INTERCARDINAL_OFFSETS.begin(), INTERCARDINAL_OFFSETS.end(),
                  [&current, &neighbours](Point offset) {
                    neighbours.emplace_back(current.x + offset.x,
                                            current.y + offset.y);
                  });
  }

  return neighbours;
}

/// Calculate the shortest path in a grid from one pair to another using the A*
/// algorithm.
///
/// Further reading which may be useful:
/// `The A* algorithm <https://en.wikipedia.org/wiki/A*_search_algorithm>`_
///
/// Parameters
/// ----------
/// grid - The 2D grid which represents the dungeon.
/// start - The start pair for the algorithm.
/// end - The end pair for the algorithm.
///
/// Returns
/// -------
/// A vector of points mapping out the shortest path from start to end.
std::vector<Point>
calculate_astar_path(std::vector<std::vector<TileType>> &grid, Point &start,
                     Point &end) {
  // Set up a few variables needed for the pathfinding
  std::vector<Point> result;
  std::priority_queue<Neighbour> queue;
  std::unordered_map<Point, Point> came_from = {{start, start}};
  std::unordered_map<Point, int> distances = {{start, 0}};
  queue.push({0, start});

  // Loop until the priority queue is empty
  while (!queue.empty()) {
    // Get the lowest cost pair from the priority queue
    Point current = queue.top().pair;
    Point parent = queue.top().parent;
    queue.pop();

    // Check if we've reached our target
    if (current == end) {
      // Backtrack through came_from to get the path
      while (!(came_from[current] == current)) {
        // Add the current pair to the result list
        result.emplace_back(current.x, current.y);

        // Get the next pair in the path
        current = came_from[current];
      }

      // Add the start pair and exit out of the loop
      result.emplace_back(start.x, start.y);
      break;
    }

    // Add all the neighbours to the heap with their cost being f = g + h:
    //   f - The total cost of traversing the neighbour.
    //   g - The distance between the start pair and the neighbour pair.
    //   h - The estimated distance from the neighbour pair to the end pair.
    //   We're using the Chebyshev distance for this.
    for (Point neighbour : prune_neighbours(current, parent)) {
      std::optional<Point> jump_point = jump(grid, neighbour, current, end);
      if (jump_point.has_value() && !came_from.count(jump_point.value())) {
        // Store the jump_point's parent and calculate its distance from the
        // start pair
        came_from.emplace(jump_point.value(), current);
        distances.emplace(jump_point.value(), distances[current] + 1);

        // Check if the jump_point is an obstacle. If so, set the total cost to
        // infinity, otherwise, set it to f = g + h
        int f_cost = distances[jump_point.value()] +
                     std::max(abs(jump_point->x - current.x),
                              abs(jump_point->y - current.y));

        // Add the jump_point to the priority queue
        queue.push({f_cost, jump_point.value(), current});
      }
    }
  }

  // Return result
  return result;
}
