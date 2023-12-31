// Related header
#include "generation/searching.hpp"

// Std headers
#include <queue>

// ----- STRUCTURES ------------------------------
/// Represents a grid position and its distance from the start position.
struct Neighbour {
  // std::priority_queue uses a max heap, but we want a min heap, so the operator needs to be reversed
  auto operator<(const Neighbour &neighbour) const -> bool { return cost > neighbour.cost; }

  /// The cost to traverse to this neighbour.
  int cost;

  /// The destination position in the grid.
  Position destination;
};

// ----- CONSTANTS ------------------------------
constexpr int TILE_MIN_DISTANCE{5};

// ----- FUNCTIONS ------------------------------
auto calculate_astar_path(const Grid &grid, const Position &start, const Position &end) -> std::vector<Position> {
  // Check if the grid size is not zero
  if (grid.width == 0 || grid.height == 0) {
    throw std::length_error("Grid size must be bigger than 0.");
  }

  // Initialise the result vector, priority queue and neighbours map which will be used during the algorithm
  std::vector<Position> result;
  std::priority_queue<Neighbour> queue;
  std::unordered_map<Position, Neighbour> neighbours{{start, {0, start}}};
  queue.emplace(0, start);

  // Loop until we have explored every neighbour or until we've reached the end
  while (!queue.empty()) {
    Position current{queue.top().destination};
    queue.pop();

    // Check if we've reached the end. If so, backtrack through the neighbours to get the resultant path
    if (current == end) {
      while (neighbours.at(current).destination != current) {
        result.push_back(current);
        current = neighbours.at(current).destination;
      }

      // Add the start position to the result and break out of the loop
      result.push_back(start);
      break;
    }

    // Add all the neighbours to the priority queue with their cost being f = g + h:
    //   f - The total cost of traversing the neighbour
    //   g - The distance between the start pair and the neighbour pair
    //   h - The estimated distance from the neighbour pair to the end pair (this uses the Chebyshev distance)
    for (const Position &neighbour : grid.get_neighbours(current)) {
      // Move around the neighbour if it is an obstacle as they have an infinite cost
      if (grid.get_value(neighbour) == TileType::Obstacle) {
        continue;
      }

      // Check if we've found a more efficient path to the neighbour and if so, add all of its neighbours to the queue
      if (const int distance{neighbours.at(current).cost + 1};
          !neighbours.contains(neighbour) || distance < neighbours.at(neighbour).cost) {
        neighbours[neighbour] = {distance, current};

        // Add the neighbour to the priority queue
        queue.emplace(distance + std::max(abs(end.x - neighbour.x), abs(end.y - neighbour.y)), neighbour);
      }
    }
  }

  // Return the most efficient path
  return result;
}

auto generate_item_position(const Grid &grid, const std::unordered_set<Position> &item_positions, const bool within)
    -> Position {
  // Check if the grid size is not zero
  if (grid.width == 0 || grid.height == 0) {
    throw std::length_error("Grid size must be bigger than 0.");
  }

  // Set up a few variables needed for the algorithm
  std::unordered_set<Position> within_positions;
  std::unordered_set<Position> outside_positions;
  std::priority_queue<Neighbour> queue;
  for (const Position &item_position : item_positions) {
    queue.emplace(0, item_position);
  }

  // Loop until we have explored every neighbour
  while (!queue.empty()) {
    const int current_cost{queue.top().cost};
    Position current{queue.top().destination};
    queue.pop();

    // Add all the neighbours to the priority queue
    for (const Position &neighbour : grid.get_neighbours(current)) {
      // Check if the neighbour neighbour position is already explored or not
      if (within_positions.contains(neighbour) || outside_positions.contains(neighbour)) {
        continue;
      }

      // Only add the neighbour if it is a floor tile
      if (grid.get_value(neighbour) != TileType::Floor) {
        continue;
      }

      // Add it to the within_positions if it is within the minimum distance, otherwise add it to the outside_positions
      if (current_cost + 1 <= TILE_MIN_DISTANCE) {
        within_positions.emplace(neighbour);
      } else {
        outside_positions.emplace(neighbour);
      }

      // Add the neighbour to the priority queue
      queue.emplace(current_cost + 1, neighbour);
    }
  }

  // Return a position based on whether we want a position within or outside the minimum distance
  if (within && !within_positions.empty()) {
    return *within_positions.begin();
  }
  if (!within && !outside_positions.empty()) {
    return *outside_positions.begin();
  }
  return {-1, -1};
}
