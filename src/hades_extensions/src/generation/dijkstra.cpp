// Related header
#include "generation/dijkstra.hpp"

// Std headers
#include <functional>
#include <queue>

namespace {
/// Perform pathfinding on a grid using a specific heuristic and cost calculation.
///
/// @param grid - The 2D grid which represents the dungeon.
/// @param start - The start position for the algorithm.
/// @param end - The end position for the algorithm.
/// @param is_not_traversable - A function which checks if a position is not traversable.
/// @param heuristic_function - A function which calculates the heuristic cost between two positions.
/// @return A map of positions and their neighbours.
auto pathfind(const Grid &grid, const Position &start, const Position &end,
              const std::function<bool(const Position &)> &is_not_traversable,
              const std::function<int(const Position &, int)> &heuristic_function)
    -> std::unordered_map<Position, Connection> {
  // Check if the grid size is not zero
  if (grid.width == 0 || grid.height == 0) {
    return {};
  }

  // Initialise the result vector, priority queue and neighbours map which will be used during the algorithm
  std::priority_queue<Connection> queue;
  std::unordered_map<Position, Connection> neighbours{{start, {.cost = 0, .source = start, .destination = start}}};
  queue.emplace(0, start, start);

  // Loop until we have explored every neighbour or until we've reached the end
  while (!queue.empty()) {
    const Position current{queue.top().destination};
    queue.pop();

    // Check if we've reached the end. If so, do an early exit
    if (current == end) {
      return neighbours;
    }

    // Add all the neighbours to the priority queue with their cost being f = g + h:
    //   f - The total cost of traversing the neighbour.
    //   g - The distance between the start pair and the neighbour pair.
    //   h - The estimated distance between the neighbour and the end position.
    for (const Position &neighbour : grid.get_neighbours(current)) {
      // Move around the neighbour if it's not valid
      if (is_not_traversable(neighbour)) {
        continue;
      }

      // Check if we've found a more efficient path to the neighbour
      const auto distance{neighbours.at(current).cost + 1};
      if (const auto neighbour_it{neighbours.find(neighbour)};
          neighbour_it != neighbours.end() && distance >= neighbour_it->second.cost) {
        continue;
      }

      // Add the neighbour to the queue and neighbours map
      queue.emplace(heuristic_function(neighbour, distance), current, neighbour);
      neighbours[neighbour] = {.cost = distance, .source = current, .destination = neighbour};
    }
  }
  return neighbours;
}
}  // namespace

auto calculate_astar_path(const Grid &grid, const Position &start, const Position &end) -> std::vector<Position> {
  // Explore the grid using the A* algorithm with the Chebyshev distance heuristic and then check if we've reached the
  // end
  const auto obstacle_check{
      [&grid](const Position &neighbour) { return grid.get_value(neighbour) == GameObjectType::Obstacle; }};
  const auto chebyshev_heuristic{
      [&end](const Position &neighbour, const int cost) { return cost + neighbour.get_distance_to(end); }};
  const auto result{pathfind(grid, start, end, obstacle_check, chebyshev_heuristic)};
  if (!result.contains(end)) {
    return {};
  }

  // Backtrack through the neighbours to get the resultant path since we've reached the end
  std::vector<Position> path;
  for (Position current{end}; current != start; current = result.at(current).source) {
    path.push_back(current);
  }
  path.push_back(start);
  return path;
}

auto get_furthest_position(const Grid &grid, const Position &start) -> Position {
  // Initialise some variables needed to find the furthest position
  Position furthest_position{.x = -1, .y = -1};
  int max_distance{-1};

  // Explore the grid using the Dijkstra algorithm to find the furthest position from the start
  const auto floor_check{
      [&grid](const Position &neighbour) { return grid.get_value(neighbour) != GameObjectType::Floor; }};
  const auto dijkstra_heuristic{[&max_distance, &furthest_position](const Position &neighbour, const int cost) {
    if (cost > max_distance) {
      max_distance = cost;
      furthest_position = neighbour;
    }
    return cost;
  }};
  pathfind(grid, start, {.x = -1, .y = -1}, floor_check, dijkstra_heuristic);
  return furthest_position;
}
