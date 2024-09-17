// Ensure this file is only included once
#pragma once

// Std headers
#include <optional>
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

/// Holds the constants for a specific level.
struct LevelConstants {
  /// The level of this game.
  int level;

  /// The width of the dungeon.
  int width;

  /// The height of the dungeon.
  int height;

  /// The total number of enemies that should exist in the dungeon.
  int enemy_limit;
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

/// Generate the game map for a given game level.
///
/// @param level - The game level to generate a map for.
/// @param seed - The seed to initialise the random generator. If this is empty then one will be generated.
/// @throws std::invalid_argument - If the level is less than 0.
/// @return A tuple containing the generated map and the level constants.
auto create_map(int level, std::optional<unsigned int> seed = std::nullopt)
    -> std::pair<std::vector<TileType>, LevelConstants>;
