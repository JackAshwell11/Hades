// Related header
#include "generation/map.hpp"

// Std headers
#include <execution>
#include <queue>
#include <unordered_map>

// Local headers
#include "generation/bsp.hpp"
#include "generation/dijkstra.hpp"

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

  /// Generate a value based on the exponential equation.
  ///
  /// @param level - The game level to generate a value for.
  /// @return The generated value.
  [[nodiscard]] auto generate_value(const int level) const -> int {
    return static_cast<int>(std::min(round(base_value * pow(increase, level)), max_value));
  }
};

namespace {
// The width of the floor tiles in the hallway.
constexpr int HALLWAY_SIZE{3};

// The minimum distance between tiles of the same type.
constexpr int MIN_TILE_DISTANCE{5};

// The number of cellular automata runs to perform.
constexpr int CELLULAR_AUTOMATA_SIMULATIONS{3};

// The number of neighbours a tile must have to remain alive.
constexpr int MIN_NEIGHBOUR_DISTANCE{4};

// The width of the grid.
constexpr MapGenerationConstant WIDTH{.base_value = 30, .increase = 1.2, .max_value = 150};

// The height of the grid.
constexpr MapGenerationConstant HEIGHT{.base_value = 20, .increase = 1.2, .max_value = 100};

// The number of obstacles to place in the grid.
constexpr MapGenerationConstant OBSTACLE_COUNT{.base_value = 20, .increase = 1.3, .max_value = 200};

// The total number of enemies that should exist for a given level.
constexpr MapGenerationConstant ENEMY_LIMIT{.base_value = 5, .increase = 1.2, .max_value = 50};

// The chances of placing an item tile in the grid.
constexpr std::array<std::pair<TileType, double>, 2> ITEM_CHANCES{
    {{TileType::HealthPotion, 0.75}, {TileType::Chest, 0.25}}};
}  // namespace

auto place_tiles(const Grid &grid, std::mt19937 &random_generator, const TileType target_tile, const double probability,
                 const int count) -> std::vector<Position> {
  // Get all the possible positions for the target tile
  std::vector<Position> valid_positions;
  for (int y{0}; y < grid.height; y++) {
    for (int x{0}; x < grid.width; x++) {
      // If the target tile is not an obstacle, only pick positions which are
      // not next to a wall. Otherwise, only pick empty positions
      bool should_add{false};
      const Position pos{.x = x, .y = y};
      if (target_tile != TileType::Obstacle) {
        const auto neighbours{grid.get_neighbours(pos)};
        should_add = std::ranges::all_of(
                         neighbours.begin(), neighbours.end(),
                         [&grid](const Position &neighbour) { return grid.get_value(neighbour) != TileType::Wall; }) &&
                     grid.get_value(pos) == TileType::Floor;
      } else {
        should_add = grid.get_value(pos) == TileType::Empty;
      }

      // Add the position if it is valid
      if (should_add) {
        valid_positions.push_back(pos);
      }
    }
  }

  // Place the target tile in random positions and remove surrounding positions
  std::vector<Position> positions;
  std::ranges::shuffle(valid_positions.begin(), valid_positions.end(), random_generator);
  std::uniform_real_distribution distribution(0.0, 1.0);
  for (int _{0}; _ < count && !valid_positions.empty(); _++) {
    const Position possible_tile{valid_positions.back()};
    valid_positions.pop_back();
    if (distribution(random_generator) <= probability) {
      grid.set_value(possible_tile, target_tile);
      positions.push_back(possible_tile);
    }

    // Remove all tiles from valid_positions within MIN_TILE_DISTANCE of the
    // placed tile
    std::erase_if(valid_positions, [&possible_tile](const Position &pos) {
      return std::abs(pos.x - possible_tile.x) <= MIN_TILE_DISTANCE ||
             std::abs(pos.y - possible_tile.y) <= MIN_TILE_DISTANCE;
    });
  }
  return positions;
}

auto create_connections(const std::vector<Rect> &rooms) -> std::unordered_set<Edge> {
  // Check if the rooms vector is empty
  if (rooms.empty()) {
    throw std::length_error("Rooms size must be bigger than 0.");
  }

  // Use Prim's algorithm to construct a minimum spanning tree from complete_graph
  std::priority_queue<Edge> unexplored;
  std::unordered_set<Rect> visited;
  std::unordered_set<Edge> mst;

  // Start with the first room
  visited.emplace(rooms[0]);
  for (const auto &room : rooms) {
    if (room != rooms[0]) {
      unexplored.emplace(rooms[0].get_distance_to(room), rooms[0], room);
    }
  }

  // Construct the minimum spanning tree
  while (mst.size() < rooms.size() - 1 && !unexplored.empty()) {
    // Get the neighbour with the lowest cost
    const Edge lowest{unexplored.top()};
    unexplored.pop();

    // Check if the neighbour is already visited or not
    if (visited.contains(lowest.destination)) {
      continue;
    }

    // Mark the destination room as visited
    visited.emplace(lowest.destination);
    mst.emplace(lowest);

    // Add edges from the newly visited room to all other unvisited rooms
    for (const auto &room : rooms) {
      if (!visited.contains(room)) {
        unexplored.emplace(lowest.destination.get_distance_to(room), lowest.destination, room);
      }
    }
  }

  // Return the constructed minimum-spanning tree
  return mst;
}

void create_hallways(const Grid &grid, const std::unordered_set<Edge> &connections) {
  // Use the A* algorithm to connect each pair of rooms avoiding the obstacles
  std::vector<std::vector<Position>> path_positions(connections.size());
  std::transform(std::execution::par, connections.begin(), connections.end(), path_positions.begin(),
                 [&grid](const Edge &connection) {
                   return calculate_astar_path(grid, connection.source.centre, connection.destination.centre);
                 });

  // Place a rect box around each path_position to create the hallways
  constexpr int HALF_HALLWAY_SIZE{HALLWAY_SIZE / 2};
  for (const std::vector<Position> &path : path_positions) {
    for (const auto &[x_pos, y_pos] : path) {
      grid.place_rect({{.x = x_pos - HALF_HALLWAY_SIZE, .y = y_pos - HALF_HALLWAY_SIZE},
                       {.x = x_pos + HALF_HALLWAY_SIZE, .y = y_pos + HALF_HALLWAY_SIZE}});
    }
  }
}

void run_cellular_automata(Grid &grid) {
  // Create a temporary grid to store the next generation then perform the cellular automata simulation
  auto temp_grid{std::make_unique<std::vector<TileType>>(*grid.grid)};
  for (int i{0}; i < grid.width * grid.height; i++) {
    // Get the number of alive neighbours and check if the tile should be alive or dead
    const auto alive_neighbours{std::ranges::count_if(
        grid.get_neighbours({.x = i % grid.width, .y = i / grid.width}),
        [&grid](const Position &neighbour) { return grid.get_value(neighbour) == TileType::Floor; })};
    temp_grid->at(i) = alive_neighbours >= MIN_NEIGHBOUR_DISTANCE ? TileType::Floor : TileType::Empty;
  }
  grid.grid = std::move(temp_grid);

  // Place walls around the floor tiles
  for (int y{0}; y < grid.height; y++) {
    for (int x{0}; x < grid.width; x++) {
      // Check if the tile is on the edge of the grid or if it has a floor neighbour (while not being a floor tile)
      const Position position{.x = x, .y = y};
      if (const auto floor_neighbours{std::ranges::count_if(
              grid.get_neighbours(position),
              [&grid](const Position &neighbour) { return grid.get_value(neighbour) == TileType::Floor; })};
          (x == 0 || y == 0 || x == grid.width - 1 || y == grid.height - 1 ||
           grid.get_value(position) != TileType::Floor) &&
          floor_neighbours > 0) {
        grid.set_value(position, TileType::Wall);
      }
    }
  }
}

auto create_map(const int level, std::optional<unsigned int> seed) -> std::pair<std::vector<TileType>, LevelConstants> {
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
  const LevelConstants constants{.level = level,
                                 .width = WIDTH.generate_value(level),
                                 .height = HEIGHT.generate_value(level),
                                 .enemy_limit = ENEMY_LIMIT.generate_value(level)};
  Grid grid{constants.width, constants.height};

  // Split the BSP tree to create the containers
  Leaf bsp{{{.x = 0, .y = 0}, {.x = constants.width - 1, .y = constants.height - 1}}};
  bsp.split(random_generator);

  // Create the rooms inside the grid using the BSP
  std::vector<Rect> rooms;
  bsp.create_room(grid, random_generator, rooms);

  // Place random obstacles in the grid and create the hallways between the rooms
  place_tiles(grid, random_generator, TileType::Obstacle, 1, OBSTACLE_COUNT.generate_value(level));
  create_hallways(grid, create_connections(rooms));

  // Run some cellular automata simulations on the grid then place the walls around the floor tiles
  for (int _{0}; _ < CELLULAR_AUTOMATA_SIMULATIONS; _++) {
    run_cellular_automata(grid);
  }

  // Place the player as well as the item tiles in the grid then place the end
  // goal at the furthest position from the player
  const auto player_position{place_tiles(grid, random_generator, TileType::Player, 1, 1)};
  for (const auto &[tile, probability] : ITEM_CHANCES) {
    place_tiles(grid, random_generator, tile, probability);
  }
  grid.set_value(get_furthest_position(grid, player_position.front()), TileType::Goal);

  // Return the grid and the level constants
  return std::make_pair(*grid.grid, constants);
}
