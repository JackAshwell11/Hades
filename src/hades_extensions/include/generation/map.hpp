// Std includes
#include <unordered_map>
#include <unordered_set>

// Custom includes
#include "bsp.hpp"
#include "primitives.hpp"

// ----- STRUCTURES ------------------------------
/// Represents an undirected weighted edge in a graph.
struct Edge {
  inline bool operator<(const Edge edg) const {
    // The priority_queue data structure gets the maximum priority, so we need
    // to override that functionality to get the minimum priority
    return cost > edg.cost;
  }

  inline bool operator==(const Edge edg) const {
    return cost == edg.cost && source == edg.source &&
        destination == edg.destination;
  }

  /// The cost of the edge.
  int cost;

  /// The source rect.
  Rect source;

  /// The destination rect.
  Rect destination;

  /// Initialise the object.
  ///
  /// @param cost_val - The cost of the edge.
  /// @param source_val - The source rect.
  /// @param destination_val - The destination rect.
  Edge(int cost_val, Rect source_val, Rect destination_val)
      : cost(cost_val), source(source_val), destination(destination_val) {}
};

// ----- HASHES ------------------------------
template<>
struct std::hash<Edge> {
  size_t operator()(const Edge &edg) const {
    size_t res = 0;
    hash_combine(res, edg.cost);
    hash_combine(res, edg.source);
    hash_combine(res, edg.destination);
    return res;
  }
};

// ----- FUNCTIONS ------------------------------
/// Collect all positions in a given grid that match the target.
///
/// @param grid - The 2D grid which represents the dungeon.
/// @param target - The TileType to test for.
/// @return A vector of positions which match the target.
std::vector<Position> collect_positions(Grid &grid, TileType target);

/// Split the bsp based on the generated constants.
///
/// @param bsp - The root leaf for the binary space partition.
/// @param random_generator - The random generator used to generate the bsp.
/// @param split_iteration - The number of splits to perform.
void split_bsp(Leaf &bsp, std::mt19937 &random_generator, int split_iteration);

/// Generate the rooms for a given game level using the bsp.
///
/// @param bsp - The root leaf for the binary space partition.
/// @param grid - The 2D grid which represents the dungeon.
/// @param random_generator - The random generator used to generate the bsp.
/// @return The generated rooms.
std::vector<Rect> generate_rooms(Leaf &bsp, Grid &grid, std::mt19937 &random_generator);

/// Create a set of connections between all the rects ensuring that every rect
/// is reachable.
///
/// @details https://en.wikipedia.org/wiki/Prim's_algorithm
///
/// @param complete_graph - An adjacency list which represents a complete
/// graph.
/// @throws std::length_error - Complete graph size must be bigger than 0.
/// @return A set of edges which form the connections between rects.
std::unordered_set<Edge> create_connections(std::unordered_map<Rect, std::vector<Rect>> &complete_graph);

/// Places a given tile in the 2D grid.
///
/// @param grid - The 2D grid which represents the dungeon.
/// @param random_generator - The random generator used to pick the position.
/// @param target_tile - The tile to place in the 2D grid.
/// @param possible_tiles - The possible tiles that the tile can be placed
/// into.
/// @throws std::length_error - Possible tiles size must be bigger than 0.
void place_tile(Grid &grid,
                std::mt19937 &random_generator,
                TileType target_tile,
                std::vector<Position> &possible_tiles);

/// Create the hallways by placing random obstacles and pathfinding around
/// them.
///
/// @param grid - The 2D grid which represents the dungeon.
/// @param random_generator - The random generator used to pick the obstacle
/// positions.
/// @param connections - The connections to pathfind using the A* algorithm.
/// @param obstacle_count - The number of obstacles to place in the 2D grid.
void create_hallways(Grid &grid,
                     std::mt19937 &random_generator,
                     std::unordered_set<Edge> &connections,
                     int obstacle_count);

/// Generate the game map for a given game level.
///
/// @param level - The game level to generate a map for.
/// @param seed - The seed to initialise the random generator. If this is
/// empty, then one will be generated.
/// @return A tuple containing the generated map and the level constants.
std::pair<std::vector<TileType>, std::tuple<int, int, int>> create_map(int level,
                                                                       std::optional<unsigned int> seed = std::nullopt);
