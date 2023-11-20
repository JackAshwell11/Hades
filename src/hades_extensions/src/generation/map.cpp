// Related header
#include "generation/map.hpp"

// Std headers
#include <execution>
#include <queue>

// Local headers
#include "generation/astar.hpp"

// ----- STRUCTURES ------------------------------
/// Stores a map generation constant which can be calculated.
///
/// @param base_value - The base value for the exponential calculation.
/// @param increase - The percentage increase for the constant.
/// @param max_value - The max value for the exponential calculation.
struct MapGenerationConstant {
  /// The base value for the exponential calculation.
  double base_value;

  /// The percentage increase for the constant.
  double increase;

  /// The max value for the exponential calculation.
  double max_value;
};

// ----- CONSTANTS ------------------------------
constexpr int HALLWAY_SIZE{5};
constexpr int HALF_HALLWAY_SIZE{HALLWAY_SIZE / 2};
const MapGenerationConstant WIDTH{30, 1.2, 150};
const MapGenerationConstant HEIGHT{20, 1.2, 100};
const MapGenerationConstant OBSTACLE_COUNT{20, 1.3, 200};
const MapGenerationConstant ITEM_COUNT{5, 1.1, 30};

// ----- FUNCTIONS ------------------------------
/// Generate a value based on the exponential equation.
///
/// @param level - The game level to generate a value for.
/// @return The generated value.
[[nodiscard]] inline auto generate_value(const MapGenerationConstant &map_generation_constant, const int level) -> int {
  return static_cast<int>(
      std::min(round(map_generation_constant.base_value * pow(map_generation_constant.increase, level)),
               map_generation_constant.max_value));
}

auto collect_positions(const Grid &grid, const TileType target) -> std::vector<Position> {
  // Get all positions in the grid that match the target
  std::vector<Position> result;
  for (int y = 0; y < grid.height; y++) {
    for (int x = 0; x < grid.width; x++) {
      if (grid.get_value({x, y}) == target) {
        result.emplace_back(x, y);
      }
    }
  }
  return result;
}

void place_tile(Grid &grid, std::mt19937 &random_generator, const TileType target_tile,
                std::vector<Position> &possible_tiles) {
  // Check if at least one tile exists
  if (possible_tiles.empty()) {
    throw std::length_error("Possible tiles size must be bigger than 0.");
  }

  // Get a random tile and place the target tile there
  const std::size_t tile_index{
      std::uniform_int_distribution<std::size_t>{0, possible_tiles.size() - 1}(random_generator)};
  const Position possible_tile{possible_tiles[tile_index]};
  possible_tiles[tile_index] = possible_tiles.back();
  possible_tiles.pop_back();
  grid.set_value(possible_tile, target_tile);
}

auto create_complete_graph(const std::vector<Rect> &rooms) -> std::unordered_map<Rect, std::vector<Rect>> {
  // Check if the rooms vector is empty
  if (rooms.empty()) {
    throw std::length_error("Rooms size must be bigger than 0.");
  }

  // Create the complete graph of all rooms
  std::unordered_map<Rect, std::vector<Rect>> complete_graph;
  for (const Rect &room : rooms) {
    std::vector<Rect> temp;
    std::copy_if(rooms.begin(), rooms.end(), std::back_inserter(temp),
                 [&room](const Rect &neighbour) { return neighbour != room; });
    complete_graph.insert({room, temp});
  }
  return complete_graph;
}

auto create_connections(const std::unordered_map<Rect, std::vector<Rect>> &complete_graph) -> std::unordered_set<Edge> {
  // Check if the complete_graph is empty
  if (complete_graph.empty()) {
    throw std::length_error("Complete graph size must be bigger than 0.");
  }

  // Use Prim's algorithm to construct a minimum spanning tree from complete_graph
  const Rect start{complete_graph.begin()->first};
  std::priority_queue<Edge> unexplored;
  std::unordered_set<Rect> visited;
  std::unordered_set<Edge> mst;
  unexplored.emplace(0, start, start);
  while (mst.size() < complete_graph.size() && !unexplored.empty()) {
    // Get the neighbour with the lowest cost
    const Edge lowest{unexplored.top()};
    unexplored.pop();

    // Check if the neighbour is already visited or not
    if (visited.contains(lowest.destination)) {
      continue;
    }

    // Neighbour isn't visited so mark it as visited and add its neighbours to the heap
    visited.emplace(lowest.destination);
    for (const auto &neighbour : complete_graph.at(lowest.destination)) {
      unexplored.emplace(lowest.destination.get_distance_to(neighbour), lowest.destination, neighbour);
    }

    // Add a new edge towards the lowest cost neighbour onto the mst
    if (lowest.source != lowest.destination) {
      // Save the connection
      mst.emplace(lowest);
    }
  }

  // Return the constructed minimum-spanning tree
  return mst;
}

void create_hallways(Grid &grid, std::mt19937 &random_generator, const std::unordered_set<Edge> &connections,
                     const int obstacle_count) {
  // Place random obstacles in the grid
  std::vector<Position> obstacle_positions{collect_positions(grid, TileType::Empty)};
  for (int _ = 0; _ < obstacle_count; _++) {
    place_tile(grid, random_generator, TileType::Obstacle, obstacle_positions);
  }

  // Use the A* algorithm to connect each pair of rooms avoiding the obstacles
  std::vector<std::vector<Position>> path_positions(connections.size());
  std::transform(std::execution::par, connections.begin(), connections.end(), path_positions.begin(),
                 [&grid](Edge connection) {
                   return calculate_astar_path(grid, connection.source.centre, connection.destination.centre);
                 });

  // Place a rect box around each path_position to create the hallways
  for (const std::vector<Position> &path : path_positions) {
    for (const Position &path_position : path) {
      Rect{{path_position.x - HALF_HALLWAY_SIZE, path_position.y - HALF_HALLWAY_SIZE},
           {path_position.x + HALF_HALLWAY_SIZE, path_position.y + HALF_HALLWAY_SIZE}}
          .place_rect(grid);
    }
  }
}

auto create_map(const int level, std::optional<unsigned int> seed)
    -> std::pair<std::vector<TileType>, std::tuple<int, int, int>> {
  // Check that the level number is valid
  if (level < 0) {
    throw std::length_error("Level must be bigger than or equal to 0.");
  }

  // Create the random generator generating a seed if one isn't provided
  if (!seed.has_value()) {
    std::random_device random_device;
    std::mt19937_64 seed_generator{random_device()};
    seed = std::uniform_int_distribution<unsigned int>{}(seed_generator);
  }
  std::mt19937 random_generator{seed.value()};

  // Initialise a few variables needed for the map generation
  const int grid_width{generate_value(WIDTH, level)};
  const int grid_height{generate_value(HEIGHT, level)};
  Grid grid{grid_width, grid_height};
  Leaf bsp{{{0, 0}, {grid_width - 1, grid_height - 1}}};

  // Split the bsp, create the rooms, and create the hallways between the rooms
  std::vector<Rect> rooms;
  split(bsp, random_generator);
  create_room(bsp, grid, random_generator, rooms);
  create_hallways(grid, random_generator, create_connections(create_complete_graph(rooms)),
                  generate_value(OBSTACLE_COUNT, level));

  // Place the player tile as well as the items in the grid
  std::vector<Position> possible_tiles{collect_positions(grid, TileType::Floor)};
  place_tile(grid, random_generator, TileType::Player, possible_tiles);
  for (int _ = 0; _ < generate_value(ITEM_COUNT, level); _++) {
    place_tile(grid, random_generator, TileType::Potion, possible_tiles);
  }

  // Return the grid and a LevelConstants object
  return std::make_pair(*grid.grid, std::make_tuple(level, grid_width, grid_height));
}
