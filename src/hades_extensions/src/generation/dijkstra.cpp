// Related header
#include "generation/dijkstra.hpp"

// Std headers
#include <algorithm>
#include <functional>
#include <queue>
#include <ranges>
#include <unordered_map>

// Local headers
#include "generation/primitives.hpp"

/// Represents a grid position and its distance from the start position.
struct Neighbour {
  // std::priority_queue uses a max heap, but we want a min heap, so the operator needs to be reversed
  auto operator<(const Neighbour &neighbour) const -> bool { return cost > neighbour.cost; }

  /// The cost to traverse to this neighbour.
  int cost;

  /// The destination position in the grid.
  Position destination;
};

namespace {
auto explore_grid(const Grid &grid, const Position &start, const Position &end,
                  const std::function<bool(const Position &)> &neighbour_check,
                  const std::function<int(const Position &, int)> &cost_calc)
    -> std::unordered_map<Position, Neighbour> {
  // Check if the grid size is not zero
  if (grid.width == 0 || grid.height == 0) {
    throw std::length_error("Grid size must be bigger than 0.");
  }

  // Initialise the result vector, priority queue and neighbours map which will be used during the algorithm
  std::priority_queue<Neighbour> queue;
  std::unordered_map<Position, Neighbour> neighbours{{start, {.cost = 0, .destination = start}}};
  queue.emplace(0, start);

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
    //   h - The estimated distance between the neighbour and the end position using the cost_calc function.
    for (const Position &neighbour : grid.get_neighbours(current)) {
      // Move around the neighbour if it's not valid
      if (neighbour_check(neighbour)) {
        continue;
      }

      // Check if we've found a more efficient path to the neighbour
      const auto distance{neighbours.at(current).cost + 1};
      if (neighbours.contains(neighbour) && distance >= neighbours.at(neighbour).cost) {
        continue;
      }

      // Add the neighbour to the queue and neighbours map
      queue.emplace(cost_calc(neighbour, distance), neighbour);
      neighbours[neighbour] = {.cost = distance, .destination = current};
    }
  }
  return neighbours;
}
}  // namespace

auto calculate_astar_path(const Grid &grid, const Position &start, const Position &end) -> std::vector<Position> {
  // Explore the grid using the A* algorithm with the Chebyshev distance
  // heuristic and then check if we've reached the end
  const auto result{explore_grid(
      grid, start, end, [&grid](const Position &neighbour) { return grid.get_value(neighbour) == TileType::Obstacle; },
      [&end](const Position &neighbour, const int cost) {
        return cost + std::max(abs(end.x - neighbour.x), abs(end.y - neighbour.y));
      })};
  if (!result.contains(end)) {
    return {};
  }

  // Backtrack through the neighbours to get the resultant path since we've
  // reached the end
  std::vector<Position> path;
  Position current{end};
  while (result.at(current).destination != current) {
    path.push_back(current);
    current = result.at(current).destination;
  }
  path.push_back(start);
  return path;
}

auto get_furthest_position(const Grid &grid, const Position &start) -> Position {
  // Explore the grid using the Dijkstra algorithm
  const auto result{explore_grid(
      grid, start, {.x = -1, .y = -1},
      [&grid](const Position &neighbour) { return grid.get_value(neighbour) != TileType::Floor; },
      [](const Position & /*neighbour*/, const int cost) { return cost; })};

  // Find the position with the highest cost that isn't touching a wall
  std::pair furthest_position{std::make_pair(Position{.x = -1, .y = -1}, Neighbour{.cost = -1, .destination = start})};
  for (const auto [position, neighbour] : result) {
    const auto neighbours{grid.get_neighbours(neighbour.destination)};
    if (const bool is_next_to_wall{std::ranges::any_of(
            neighbours,
            [&grid](const Position &neighbour_pos) { return grid.get_value(neighbour_pos) == TileType::Wall; })};
        neighbour.cost > furthest_position.second.cost && !is_next_to_wall) {
      furthest_position = {position, neighbour};
    }
  }
  return furthest_position.first;
}
