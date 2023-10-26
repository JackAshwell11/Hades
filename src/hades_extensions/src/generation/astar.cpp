// Std includes
#include <array>
#include <queue>
#include <unordered_map>

// Custom includes
#include "generation/astar.hpp"

// ----- STRUCTURES ------------------------------
/// Represents a grid position and its distance from the start position.
struct Neighbour {
  // std::priority_queue uses a max heap, but we want a min heap, so the operator needs to be reversed
  inline bool operator<(const Neighbour &neighbour) const { return cost > neighbour.cost; }

  /// The cost to traverse to this neighbour.
  int cost;

  /// The destination position in the grid.
  Position destination;

  /// Initialise the object.
  Neighbour() = default;

  /// Initialise the object.
  ///
  /// @param cost - The cost to traverse to this neighbour.
  /// @param destination - The destination position in the grid.
  Neighbour(int cost, const Position &destination) : cost(cost), destination(destination) {}
};

// ----- CONSTANTS ------------------------------
// Represents the north, south, east, west, north-east, north-west, south-east and south-west directions on a compass
const std::array<Position, 8> INTERCARDINAL_OFFSETS = {Position{-1, -1}, Position{0, -1}, Position{1, -1},
                                                       Position{-1, 0},  Position{1, 0},  Position{-1, 1},
                                                       Position{0, 1},   Position{1, 1}};

// ----- FUNCTIONS ------------------------------
std::vector<Position> calculate_astar_path(const Grid &grid, const Position &start, const Position &end) {
  // Check if the grid size is not zero
  if (!grid.width) {
    throw std::length_error("Grid size must be bigger than 0.");
  }

  // Initialise the result vector, priority queue and neighbours map which will be used during the algorithm
  std::vector<Position> result;
  std::priority_queue<Neighbour> queue;
  std::unordered_map<Position, Neighbour> neighbours{{start, {0, start}}};
  queue.emplace(0, start);

  // Loop until we have explored every neighbour or until we've reached the end
  while (!queue.empty()) {
    Position current = queue.top().destination;
    queue.pop();

    // Check if we've reached the end. If so, backtrack through the neighbours to get the resultant path
    if (current == end) {
      while (!(neighbours.at(current).destination == current)) {
        result.push_back(current);
        current = neighbours.at(current).destination;
      }

      // Add the start position to the result and break out of the loop
      result.push_back(start);
      break;
    }

    // Add all the neighbours to the heap with their cost being f = g + h:
    //   f - The total cost of traversing the neighbour
    //   g - The distance between the start pair and the neighbour pair
    //   h - The estimated distance from the neighbour pair to the end pair (this uses the Chebyshev distance)
    for (const Position &offset : INTERCARDINAL_OFFSETS) {
      // Move around the neighbour if it is an obstacle as they have an infinite cost
      Position neighbour = current + offset;
      if (neighbour.x < 1 || neighbour.x >= grid.width - 1 || neighbour.y < 1 || neighbour.y >= grid.height - 1) {
        continue;
      }

      // Move around the neighbour if it is an obstacle as they have an infinite cost
      if (grid.get_value(neighbour) == TileType::Obstacle) {
        continue;
      }

      // Check if we've found a more efficient path to the neighbour and if so, add all of its neighbours to the queue
      int distance = neighbours.at(current).cost + 1;
      if (!neighbours.contains(neighbour) || distance < neighbours.at(neighbour).cost) {
        neighbours[neighbour] = {distance, current};

        // Add the neighbour to the priority queue
        queue.emplace(distance + std::max(abs(end.x - neighbour.x), abs(end.y - neighbour.y)), neighbour);
      }
    }
  }

  // Return the most efficient path
  return result;
}
