// Std includes
#include <unordered_map>
#include <unordered_set>

// Custom includes
#include "bsp.hpp"
#include "primitives.hpp"

// ----- STRUCTURES ------------------------------
/// Represents an undirected weighted edge in a graph.
struct Edge {
  int cost;
  Rect source, destination;

  /// Construct an Edge object.
  ///
  /// Parameters
  /// ----------
  /// cost - The cost to traverse this edge.
  /// source - The starting node.
  /// destination - The ending node.
  ///
  /// Returns
  /// -------
  /// An Edge object.
  Edge(int cost_val, Rect source_val, Rect destination_val) {
    cost = cost_val;
    source = source_val;
    destination = destination_val;
  }

  inline bool operator<(const Edge edg) const {
    // The priority_queue data structure gets the maximum priority, so we need
    // to override that functionality to get the minimum priority
    return cost > edg.cost;
  }

  inline bool operator==(const Edge edg) const {
    return cost == edg.cost && source == edg.source &&
        destination == edg.destination;
  }
};

// ----- FUNCTIONS ------------------------------
/// Allows the edge struct to be hashed in a map.
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

// ----- DEFINITIONS ------------------------------
/// Collect all points in a given grid that match the target.
///
/// Parameters
/// ----------
/// grid - The 2D grid which represents the dungeon.
/// target - The TileType to test for.
///
/// Returns
/// -------
/// A vector of points which match the target.
std::vector<Point> collect_positions(Grid &grid, TileType target);

/// Split the bsp based on the generated constants.
///
/// Parameters
/// ----------
/// bsp - The root leaf for the binary space partition.
/// grid - The 2D grid which represents the dungeon.
/// random_generator - The random generator used to generate the bsp.
/// split_iteration - The number of splits to perform.
void split_bsp(Leaf &bsp, Grid &grid, std::mt19937 &random_generator, int split_iteration);

/// Generate the rooms for a given game level using the bsp.
///
/// Parameters
/// ----------
/// bsp - The root leaf for the binary space partition.
/// grid - The 2D grid which represents the dungeon.
/// random_generator - The random generator used to generate the bsp.
///
/// Returns
/// -------
/// The generated rooms.
std::vector<Rect> generate_rooms(Leaf &bsp, Grid &grid, std::mt19937 &random_generator);

/// Create a set of connections between all the rects ensuring that every rect
/// is reachable.
///
/// Further reading which may be useful:
/// `Prim's algorithm <https://en.wikipedia.org/wiki/Prim's_algorithm>`_
///
/// Parameters
/// ----------
/// complete_graph - An adjacency list which represents a complete graph.
///
/// Throws
/// ------
/// std::length_error - Complete graph size must be bigger than 0.
///
/// Returns
/// -------
/// A set of edges which form the connections between rects.
std::unordered_set<Edge> create_connections(std::unordered_map<Rect, std::vector<Rect>> &complete_graph);

/// Places a given tile in the 2D grid.
///
/// Parameters
/// ----------
/// grid - The 2D grid which represents the dungeon.
/// random_generator - The random generator used to pick the position.
/// target_tile - The tile to place in the 2D grid.
/// possible_tiles - The possible tiles that the tile can be placed into.
///
/// Throws
/// ------
/// std::length_error - Possible tiles size must be bigger than 0.
void place_tile(Grid &grid, std::mt19937 &random_generator, TileType target_tile, std::vector<Point> &possible_tiles);

/// Create the hallways by placing random obstacles and pathfinding around them.
///
/// Parameters
/// ----------
/// grid - The 2D grid which represents the dungeon.
/// random_generator - The random generator used to pick the obstacle positions.
/// connections - The connections to pathfind using the A* algorithm.
/// obstacle_count - The number of obstacles to place in the 2D grid.
void create_hallways(Grid &grid,
                     std::mt19937 &random_generator,
                     std::unordered_set<Edge> &connections,
                     int obstacle_count);

/// Generate the game map for a given game level.
///
/// Parameters
/// ----------
/// level - The game level to generate a map for.
/// seed - The seed to initialise the random generator. If this is empty, then
/// one will be generated.
///
/// Returns
/// -------
/// A tuple containing the generated map and the level constants.
std::pair<std::vector<TileType>, std::tuple<int, int, int>> create_map(int level,
                                                                       std::optional<unsigned int> seed = std::nullopt);
