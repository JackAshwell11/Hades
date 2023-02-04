// Std includes
#include <unordered_map>
#include <unordered_set>

// Custom includes
#include "bsp.hpp"
#include "primitives.hpp"

// ----- STRUCTURES ------------------------------
/// Represents an undirected weighted edge in a graph.
struct Edge {
  int cost{};
  Rect source, destination;

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
inline std::vector<Point> collect_positions(std::vector<std::vector<TileType>> &grid, TileType target);

void split_bsp(Leaf &bsp,
               std::vector<std::vector<TileType>> &grid,
               std::mt19937 &random_generator,
               int split_iteration);

std::vector<Rect> generate_rooms(Leaf &bsp, std::vector<std::vector<TileType>> &grid, std::mt19937 &random_generator);

std::unordered_set<Edge> create_connections(std::unordered_map<Rect, std::vector<Rect>> &complete_graph);

void place_tile(std::vector<std::vector<TileType>> &grid,
                std::mt19937 &random_generator,
                TileType target_tile,
                std::vector<Point> &possible_tiles);

std::pair<std::vector<std::vector<TileType>>, std::tuple<int, int, int>> create_map(int level, unsigned int seed);

void create_hallways(std::vector<std::vector<TileType>> &grid,
                     std::mt19937 &random_generator,
                     std::unordered_set<Edge> &connections,
                     int obstacle_count);
