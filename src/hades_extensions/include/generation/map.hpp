// Ensure this file is only included once
#pragma once

// Std headers
#include <optional>
#include <unordered_set>

// Local headers
#include "bsp.hpp"

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
/// Collect all positions in a given grid that match the target.
///
/// @param grid - The 2D grid which represents the dungeon.
/// @param target - The TileType to test for.
/// @return A vector of positions which match the target.
auto collect_positions(const Grid &grid, TileType target) -> std::vector<Position>;

/// Places a given tile in the 2D grid.
///
/// @param grid - The 2D grid which represents the dungeon.
/// @param random_generator - The random generator used to pick the position.
/// @param target_tile - The tile to place in the 2D grid.
/// @param possible_tiles - The possible tiles that the tile can be placed into.
/// @throws std::length_error - Possible tiles size must be bigger than 0.
void place_tile(const Grid &grid, std::mt19937 &random_generator, TileType target_tile,
                std::vector<Position> &possible_tiles);

/// Create a complete graph from a given list of rooms.
///
/// @param rooms - The rooms to create connections between.
/// @throws std::length_error - Rooms size must be bigger than 0.
/// @return A adjacency list of all the rooms and their neighbours.
auto create_complete_graph(const std::vector<Rect> &rooms) -> std::unordered_map<Rect, std::vector<Rect>>;

/// Create a minimum spanning tree from a given complete graph.
///
/// @details https://en.wikipedia.org/wiki/Prim%27s_algorithm
/// @param complete_graph - An adjacency list which represents a complete graph. This should not be empty.
/// @return A set of edges which form the connections between rects.
auto create_connections(const std::unordered_map<Rect, std::vector<Rect>> &complete_graph) -> std::unordered_set<Edge>;

/// Create the hallways by placing random obstacles and pathfinding around them.
///
/// @param grid - The 2D grid which represents the dungeon.
/// @param random_generator - The random generator used to pick the obstacle positions.
/// @param connections - The connections to pathfind using the A* algorithm.
/// @param obstacle_count - The number of obstacles to place in the 2D grid.
void create_hallways(Grid &grid, std::mt19937 &random_generator, const std::unordered_set<Edge> &connections,
                     int obstacle_count);

/// Generate the game map for a given game level.
///
/// @param level - The game level to generate a map for.
/// @param seed - The seed to initialise the random generator. If this is empty then one will be generated.
/// @return A tuple containing the generated map and the level constants.
auto create_map(int level, std::optional<unsigned int> seed = std::nullopt)
    -> std::pair<std::vector<TileType>, std::tuple<int, int, int>>;
