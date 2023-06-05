// Std includes
#include <array>
#include <queue>
#include <stdexcept>
#include <unordered_map>

// Custom includes
#include "astar.hpp"
#include "primitives.hpp"

// ----- STRUCTURES ------------------------------
/// Represents a grid position and its costs from the start position
///
/// @param cost - The cost to traverse to this neighbour.
/// @param destination - The destination point in the grid.
struct Neighbour {
  int cost;
  Point destination;

  inline bool operator<(const Neighbour nghbr) const {
    // The priority_queue data structure gets the maximum priority, so we need
    // to override that functionality to get the minimum priority
    return cost > nghbr.cost;
  }
};

// ----- CONSTANTS ------------------------------
// Represents the north, south, east, west, north-east, north-west, south-east and south-west directions on a compass
const std::array<Point, 8> INTERCARDINAL_OFFSETS = {
    Point{-1, -1}, Point{0, -1}, Point{1, -1}, Point{-1, 0}, Point{1, 0}, Point{-1, 1}, Point{0, 1}, Point{1, 1},
};

// ----- FUNCTIONS ------------------------------
std::vector<Point> calculate_astar_path(Grid &grid, const Point start, const Point end) {
  // Check if the grid size is not zero, if not, set up a few variables needed
  // for the pathfinding
  if (!grid.width) {
    throw std::length_error("Grid size must be bigger than 0.");
  }
  std::vector<Point> result;
  std::priority_queue<Neighbour> queue;
  std::unordered_map<Point, Neighbour> neighbours{{start, {0, start}}};
  queue.push({0, start});

  // Loop until the priority queue is empty
  while (!queue.empty()) {
    // Get the lowest cost pair from the priority queue
    Point current = queue.top().destination;
    queue.pop();

    // Check if we've reached our target
    if (current == end) {
      // Backtrack through neighbours to get the path
      while (!(neighbours[current].destination == current)) {
        // Add the current pair to the result list
        result.push_back(current);

        // Get the next pair in the path
        current = neighbours[current].destination;
      }

      // Add the start point and exit out of the loop
      result.push_back(start);
      break;
    }

    // Add all the neighbours to the heap with their cost being f = g + h:
    //   f - The total cost of traversing the neighbour.
    //   g - The distance between the start pair and the neighbour pair.
    //   h - The estimated distance from the neighbour pair to the end pair.
    //   We're using the Chebyshev distance for this.
    for (Point offset : INTERCARDINAL_OFFSETS) {
      // Calculate the neighbour's position and check if its valid excluding the
      // boundaries as that produces weird paths
      Point neighbour = current + offset;
      if (neighbour.x < 1 || neighbour.x >= grid.width - 1 || neighbour.y < 1 || neighbour.y >= grid.height - 1) {
        continue;
      }

      // Test if the neighbour is an obstacle or not. If so, skip to the next
      // neighbour as we want to move around it
      if (grid.get_value(neighbour) == TileType::Obstacle) {
        continue;
      }

      // Calculate the distance from the start
      int distance = neighbours[current].cost + 1;

      // Check if we need to add a new neighbour to the heap
      if ((!neighbours.contains(neighbour)) || distance < neighbours[neighbour].cost) {
        neighbours[neighbour] = {distance, current};

        // Add the neighbour to the priority queue
        Point diff = end - neighbour;
        queue.emplace(distance + std::max(diff.x, diff.y), neighbour);
      }
    }
  }

  // Return result
  return result;
}
