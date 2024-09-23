// Ensure this file is only included once
#pragma once

// Std headers
#include <random>
#include <unordered_set>

// Local headers
#include "primitives.hpp"

/// Represents an undirected weighted edge in a graph.
struct Edge {
  // std::priority_queue uses a max heap, but we want a min heap, so the operator needs to be reversed
  auto operator<(const Edge &edge) const -> bool { return cost > edge.cost; }

  auto operator==(const Edge &edge) const -> bool {
    return cost == edge.cost && source == edge.source && destination == edge.destination;
  }

  /// The cost of the edge.
  int cost;

  /// The source rect.
  Rect source;

  /// The destination rect.
  Rect destination;
};

template <>
struct std::hash<Edge> {
  auto operator()(const Edge &edge) const noexcept -> size_t {
    size_t res{0};
    hash_combine(res, edge.cost);
    hash_combine(res, edge.source);
    hash_combine(res, edge.destination);
    return res;
  }
};

/// Place a given amount of tiles in the 2D grid.
///
/// @param grid - The 2D grid which represents the dungeon.
/// @param random_generator - The random generator used to pick the position.
/// @param target_tile - The tile to place in the 2D grid.
/// @param probability - The probability of placing the tile.
/// @param count - The number of tiles to place.
void place_tiles(const Grid &grid, std::mt19937 &random_generator, TileType target_tile, double probability,
                 int count = std::numeric_limits<int>::max());

/// Create a minimum spanning tree from a given complete graph.
///
/// @details https://en.wikipedia.org/wiki/Prim%27s_algorithm
/// @param rooms - The rooms to create connections between.
/// @throws std::length_error - If rooms is empty.
/// @return A set of edges which form the connections between rects.
auto create_connections(const std::vector<Rect> &rooms) -> std::unordered_set<Edge>;

/// Create the hallways by using A* to pathfind between the rooms.
///
/// @param grid - The 2D grid which represents the dungeon.
/// @param connections - The connections to pathfind using the A* algorithm.
void create_hallways(const Grid &grid, const std::unordered_set<Edge> &connections);

/// Perform a cellular automata simulation on the grid.
///
/// @param grid - The 2D grid which represents the dungeon.
void run_cellular_automata(Grid &grid);
