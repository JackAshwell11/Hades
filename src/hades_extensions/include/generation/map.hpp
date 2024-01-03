// Ensure this file is only included once
#pragma once

// Std headers
#include <optional>
#include <unordered_set>

// Local headers
#include "bsp.hpp"
#include "searching.hpp"

// ----- STRUCTURES ------------------------------
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
};

// ----- HASHES ------------------------------
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

// ----- FUNCTIONS ------------------------------
/// Place a random tile in the 2D grid.
///
/// @param grid - The 2D grid which represents the dungeon.
/// @param random_generator - The random generator used to pick the position.
/// @param replaceable_tile - The tile to replace in the 2D grid.
/// @param target_tile - The tile to place in the 2D grid.
/// @param count - The number of tiles to place.
/// @throws std::length_error - If there are not enough replaceable tiles to place the target tiles.
[[maybe_unused]] auto place_random_tiles(const Grid &grid, std::mt19937 &random_generator, TileType replaceable_tile,
                                         TileType target_tile, int count = 1) -> std::unordered_set<Position>;

/// Places a tile in the 2D grid using the Dijkstra map algorithm.
///
/// @param grid - The 2D grid which represents the dungeon.
/// @param random_generator - The random generator used to pick the position.
/// @param item_positions - The positions of all the items in the 2D grid.
/// @param target_tile - The tile to place in the 2D grid.
/// @param count - The number of tiles to place.
void place_dijkstra_tiles(const Grid &grid, std::mt19937 &random_generator,
                          std::unordered_set<Position> &item_positions, TileType target_tile, int count);

/// Create a complete graph from a given list of rooms.
///
/// @param rooms - The rooms to create connections between.
/// @throws std::length_error - If rooms is empty.
/// @return A adjacency list of all the rooms and their neighbours.
auto create_complete_graph(const std::vector<Rect> &rooms) -> std::unordered_map<Rect, std::vector<Rect>>;

/// Create a minimum spanning tree from a given complete graph.
///
/// @details https://en.wikipedia.org/wiki/Prim%27s_algorithm
/// @param complete_graph - An adjacency list which represents a complete graph. This should not be empty.
/// @throws std::length_error - If complete_graph is empty.
/// @return A set of edges which form the connections between rects.
auto create_connections(const std::unordered_map<Rect, std::vector<Rect>> &complete_graph) -> std::unordered_set<Edge>;

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
