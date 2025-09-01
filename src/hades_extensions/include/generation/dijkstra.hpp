// Ensure this file is only included once
#pragma once

// Local headers
#include "generation/primitives.hpp"

/// Represents an undirected weighted connection in a graph.
struct Connection {
  /// The less than operator.
  auto operator<(const Connection& connection) const -> bool {
    // std::priority_queue uses a max heap, but we want a min heap, so the operator needs to be reversed
    return cost > connection.cost;
  }

  /// The equality operator.
  auto operator==(const Connection& connection) const -> bool {
    return cost == connection.cost && source == connection.source && destination == connection.destination;
  }

  /// The cost of the connection.
  int cost;

  /// The source position.
  Position source;

  /// The destination position.
  Position destination;
};

/// Calculate the shortest path in a grid from one pair to another using the A* algorithm.
///
/// @details https://en.wikipedia.org/wiki/A%2A_search_algorithm
/// @param grid - The 2D grid which represents the dungeon.
/// @param start - The start position for the algorithm.
/// @param end - The end position for the algorithm.
/// @throws std::length_error - If the grid size is less than 0.
/// @return A vector of positions mapping out the shortest path from start to end.
auto calculate_astar_path(const Grid& grid, const Position& start, const Position& end) -> std::vector<Position>;

/// Get the furthest position from the start position in the grid using a Dijkstra map.
///
/// @param grid - The 2D grid which represents the dungeon.
/// @param start - The start position for the algorithm.
/// @throws std::length_error - If the grid size is less than 0.
/// @return A position which is the furthest from the start position.
auto get_furthest_position(const Grid& grid, const Position& start) -> Position;
