// Std includes
#include <queue>
#include <stdexcept>
#include <unordered_map>

// Custom includes
#include "astar.hpp"

// ----- FUNCTIONS ------------------------------
std::vector<Point> grid_bfs(Point &target, int height, int width) {
  // Create a vector to store the neighbours
  std::vector<Point> result;

  // Iterate over each offset and check if it's a valid neighbour
  for (Point offset : INTERCARDINAL_OFFSETS) {
    int x = target.x + offset.x;
    int y = target.y + offset.y;
    if ((x >= 0 && x < width) && (y >= 0 && y < height)) {
      result.emplace_back(x, y);
    }
  }

  // Return the result
  return result;
}

std::vector<Point>
calculate_astar_path(std::vector<std::vector<TileType>> &grid, Point &start,
                     Point &end) {
  // Check if the grid size is not zero, if not, set up a few variables needed
  // for the pathfinding
  if (grid.empty()) {
    throw std::length_error("Grid size must be bigger than 0.");
  }
  std::vector<Point> result;
  std::priority_queue<Neighbour> queue;
  std::unordered_map<Point, Point> came_from = {{start, start}};
  std::unordered_map<Point, int> distances = {{start, 0}};
  int height = (int) grid.capacity();
  int width = (int) grid[0].capacity();
  queue.push({0, start});

  // Loop until the priority queue is empty
  while (!queue.empty()) {
    // Get the lowest cost pair from the priority queue
    Point current = queue.top().pair;
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
    for (Point neighbour : grid_bfs(current, height, width)) {
      // Test if the neighbour is an obstacle or not. If so, skip to the next
      // neighbour as we want to move around it
      if (grid[neighbour.y][neighbour.x] == TileType::Obstacle) {
        continue;
      }

      // Calculate the distance from the start
      int distance = distances[current] + 1;

      // Check if we need to add a new neighbour to the heap
      if ((!came_from.contains(neighbour)) || distance < distances.at(neighbour)) {
        came_from[neighbour] = current;
        distances[neighbour] = distance;

        // Add the neighbour to the priority queue
        queue.push({distance + std::max(abs(end.x - neighbour.x), abs(end.y - neighbour.y)), neighbour});
      }
    }
  }

  // Return result
  return result;
}
