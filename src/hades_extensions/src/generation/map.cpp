// Related header
#include "generation/map.hpp"

// Std headers
#include <execution>
#include <queue>

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
constexpr double WITHIN_MIN_DISTANCE_CHANCE{0.3};
constexpr MapGenerationConstant WIDTH{30, 1.2, 150};
constexpr MapGenerationConstant HEIGHT{20, 1.2, 100};
constexpr MapGenerationConstant OBSTACLE_COUNT{20, 1.3, 200};
constexpr MapGenerationConstant ITEM_COUNT{5, 1.1, 30};

// ----- FUNCTIONS ------------------------------
/// Generate a value based on the exponential equation.
///
/// @param map_generation_constant - The map generation constant to use.
/// @param level - The game level to generate a value for.
/// @return The generated value.
[[nodiscard]] inline auto generate_value(const MapGenerationConstant &map_generation_constant, const int level) -> int {
  return static_cast<int>(
      std::min(round(map_generation_constant.base_value * pow(map_generation_constant.increase, level)),
               map_generation_constant.max_value));
}

void place_tiles(const Grid &grid, std::mt19937 &random_generator, std::unordered_set<Position> &item_positions,
                 const TileType target_tile, const int count) {
  // Place each tile using the Dijkstra map
  for (int _ = 0; _ < count; _++) {
    // Determine if we should select a position within or outside the minimum distance
    const bool within_min_distance{std::uniform_real_distribution<>{0, 1}(random_generator) <
                                   WITHIN_MIN_DISTANCE_CHANCE};

    // Generate the Dijkstra map for the grid and place the tile in a random position
    const Position possible_tile{generate_dijkstra_map_position(grid, item_positions, within_min_distance)};
    grid.set_value(possible_tile, target_tile);
    item_positions.emplace(possible_tile);
  }
}

auto place_tiles(const Grid &grid, std::mt19937 &random_generator, const TileType replaceable_tile,
                 const TileType target_tile, const int count) -> std::unordered_set<Position> {
  // Get all the positions that match the replaceable tile
  std::vector<Position> replaceable_tiles;
  for (int y = 0; y < grid.height; y++) {
    for (int x = 0; x < grid.width; x++) {
      if (grid.get_value({x, y}) == replaceable_tile) {
        replaceable_tiles.emplace_back(x, y);
      }
    }
  }

  // Create a collection to store the item positions then place each tile
  std::unordered_set<Position> item_positions;
  for (int _ = 0; _ < count; _++) {
    const std::size_t tile_index{
        std::uniform_int_distribution<std::size_t>{0, replaceable_tiles.size() - 1}(random_generator)};
    const Position possible_tile{replaceable_tiles[tile_index]};
    replaceable_tiles[tile_index] = replaceable_tiles.back();
    grid.set_value(possible_tile, target_tile);
    replaceable_tiles.pop_back();
    item_positions.emplace(possible_tile);
  }
  return item_positions;
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
    std::ranges::copy_if(rooms.begin(), rooms.end(), std::back_inserter(temp),
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

void create_hallways(Grid &grid, const std::unordered_set<Edge> &connections) {
  // Use the A* algorithm to connect each pair of rooms avoiding the obstacles
  std::vector<std::vector<Position>> path_positions(connections.size());
  std::transform(std::execution::par, connections.begin(), connections.end(), path_positions.begin(),
                 [&grid](const Edge &connection) {
                   return calculate_astar_path(grid, connection.source.centre, connection.destination.centre);
                 });

  // Place a rect box around each path_position to create the hallways
  for (const std::vector<Position> &path : path_positions) {
    for (const auto &[x_pos, y_pos] : path) {
      grid.place_rect({{x_pos - HALF_HALLWAY_SIZE, y_pos - HALF_HALLWAY_SIZE},
                       {x_pos + HALF_HALLWAY_SIZE, y_pos + HALF_HALLWAY_SIZE}});
    }
  }
}

#include <iostream>

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

  // Place random obstacles in the grid then create hallways between the rooms
  place_tiles(grid, random_generator, TileType::Empty, TileType::Obstacle, generate_value(OBSTACLE_COUNT, level));
  create_hallways(grid, create_connections(create_complete_graph(rooms)));

  // Place the player as well as the item tiles in the grid
  auto item_positions{place_tiles(grid, random_generator, TileType::Floor, TileType::Player)};
  place_tiles(grid, random_generator, item_positions, TileType::Potion, generate_value(ITEM_COUNT, level));

  // Return the grid and a LevelConstants object
  return std::make_pair(*grid.grid, std::make_tuple(level, grid_width, grid_height));
}

// TODO: Optimise/simplify this whole file and maybe combine place_tiles with a boolean
