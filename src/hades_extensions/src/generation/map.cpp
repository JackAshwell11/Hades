// Related header
#include "generation/map.hpp"

// Std headers
#include <array>
#include <execution>
#include <queue>
#include <unordered_set>

// Local headers
#include "generation/bsp.hpp"

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
/// The width of the floor tiles in the hallway.
constexpr int HALLWAY_SIZE{3};

/// The minimum distance between tiles of the same type.
constexpr int MIN_TILE_DISTANCE{5};

/// The number of neighbours a tile must have to remain alive.
constexpr int MIN_NEIGHBOUR_DISTANCE{4};

/// The width of the grid.
constexpr MapGenerationConstant WIDTH{.base_value = 30, .increase = 1.2, .max_value = 150};

/// The height of the grid.
constexpr MapGenerationConstant HEIGHT{.base_value = 20, .increase = 1.2, .max_value = 100};

/// The number of obstacles to place in the grid.
constexpr MapGenerationConstant OBSTACLE_COUNT{.base_value = 20, .increase = 1.3, .max_value = 200};

/// The total number of enemies that should exist for a given level.
constexpr MapGenerationConstant ENEMY_LIMIT{.base_value = 5, .increase = 1.2, .max_value = 50};

/// The chances of placing an item tile in the grid.
constexpr std::array<std::pair<GameObjectType, double>, 2> ITEM_CHANCES{
    {{GameObjectType::HealthPotion, 0.75}, {GameObjectType::Chest, 0.25}}};

/// The width of the dungeon lobby.
constexpr int LOBBY_WIDTH{15};

/// The height of the dungeon lobby.
constexpr int LOBBY_HEIGHT{11};

/// The position of the player in the dungeon lobby.
constexpr Position LOBBY_PLAYER_POSITION{.x = 7, .y = 5};

/// The position of the goal in the dungeon lobby.
constexpr Position LOBBY_GOAL_POSITION{.x = 7, .y = 9};

/// The position of the shop in the dungeon lobby.
constexpr Position LOBBY_SHOP_POSITION{.x = 7, .y = 1};

/// Count the number of floor neighbours for a given position.
///
/// @param grid - The 2D grid which represents the dungeon.
/// @param position - The position to check the neighbours for.
/// @return The number of floor neighbours.
const auto count_floor_neighbours{[](const Grid &grid, const Position &position) {
  return std::ranges::count_if(grid.get_neighbours(position), [&grid](const Position &neighbour) {
    return grid.get_value(neighbour) == GameObjectType::Floor;
  });
}};

/// Place a given amount of tiles in the 2D grid.
///
/// @param grid - The 2D grid which represents the dungeon.
/// @param random_generator - The random generator used to pick the position.
/// @param target_tile - The tile to place in the 2D grid.
/// @param probability - The probability of placing the tile.
/// @param count - The number of tiles to place.
void place_tiles(Grid &grid, std::mt19937 &random_generator, const GameObjectType target_tile, const double probability,
                 const int count = std::numeric_limits<int>::max()) {
  const auto is_next_to_wall{[&grid](const Position &position) {
    return std::ranges::any_of(grid.get_neighbours(position), [&grid](const Position &neighbour) {
      return grid.get_value(neighbour) == GameObjectType::Wall;
    });
  }};

  // Get all the possible positions for the target tile
  std::vector<Position> valid_positions;
  for (auto i{0}; i < grid.width * grid.height; i++) {
    const Position position{grid.convert_position(i)};
    if (target_tile != GameObjectType::Obstacle) {
      if (grid.get_value(position) == GameObjectType::Floor && !is_next_to_wall(position)) {
        valid_positions.push_back(position);
      }
    } else if (grid.get_value(position) == GameObjectType::Empty) {
      valid_positions.push_back(position);
    }
  }

  // Place the target tile in random positions and remove surrounding positions
  std::ranges::shuffle(valid_positions.begin(), valid_positions.end(), random_generator);
  const auto tile_count{std::max(1, static_cast<int>(count * probability))};
  for (auto _{0}; _ < tile_count && !valid_positions.empty(); _++) {
    const Position possible_tile{valid_positions.back()};
    valid_positions.pop_back();
    grid.set_value(possible_tile, target_tile);

    // Remove all tiles from valid_positions within MIN_TILE_DISTANCE of the placed tile
    std::erase_if(valid_positions, [&possible_tile](const Position &pos) {
      return std::abs(pos.x - possible_tile.x) <= MIN_TILE_DISTANCE ||
             std::abs(pos.y - possible_tile.y) <= MIN_TILE_DISTANCE;
    });
  }
}
}  // namespace

MapGenerator::MapGenerator() : level_{0}, grid_{LOBBY_WIDTH, LOBBY_HEIGHT}, random_generator_{std::random_device{}()} {}

MapGenerator::MapGenerator(const int level, const std::mt19937 &random_generator)
    : level_(level),
      grid_{WIDTH.generate_value(level), HEIGHT.generate_value(level)},
      random_generator_{random_generator} {}

auto MapGenerator::generate_rooms() -> MapGenerator & {
  Leaf bsp{{{.x = 0, .y = 0}, {.x = grid_.width - 1, .y = grid_.height - 1}}};
  bsp.split(random_generator_);
  bsp.create_room(grid_, random_generator_, rooms_);
  return *this;
}

auto MapGenerator::create_connections() -> MapGenerator & {
  if (rooms_.empty()) {
    throw std::length_error("Rooms size must be bigger than 0.");
  }
  std::priority_queue<Connection> unexplored;
  std::unordered_set visited{rooms_.front()};

  // Add all the rooms to the unexplored queue
  for (auto i{1}; std::cmp_less(i, rooms_.size()); i++) {
    unexplored.emplace(rooms_[0].get_distance_to(rooms_[i]), rooms_[0], rooms_[i]);
  }

  // Construct the minimum spanning tree
  while (connections_.size() < rooms_.size() - 1 && !unexplored.empty()) {
    // Get the neighbour with the lowest cost that has not been visited
    const Connection lowest{unexplored.top()};
    unexplored.pop();
    if (visited.contains(lowest.destination)) {
      continue;
    }

    // Add connections from the newly visited room to all other unvisited rooms
    connections_.push_back(lowest);
    visited.emplace(lowest.destination);
    for (const auto &room : rooms_) {
      if (!visited.contains(room)) {
        unexplored.emplace(lowest.destination.get_distance_to(room), lowest.destination, room);
      }
    }
  }
  return *this;
}

auto MapGenerator::generate_hallways() -> MapGenerator & {
  // Use the A* algorithm to connect each pair of rooms avoiding the obstacles
  constexpr int HALF_HALLWAY_SIZE{HALLWAY_SIZE / 2};
  std::vector<std::vector<Position>> path_positions(connections_.size());
  std::transform(std::execution::par, connections_.begin(), connections_.end(), path_positions.begin(),
                 [this](const Connection &connection) {
                   return calculate_astar_path(grid_, connection.source, connection.destination);
                 });

  // Place a rect box around each path_position to create the hallways
  for (const std::vector<Position> &path : path_positions) {
    for (const auto &[x_pos, y_pos] : path) {
      grid_.place_rect({{.x = x_pos - HALF_HALLWAY_SIZE, .y = y_pos - HALF_HALLWAY_SIZE},
                        {.x = x_pos + HALF_HALLWAY_SIZE, .y = y_pos + HALF_HALLWAY_SIZE}});
    }
  }
  return *this;
}

auto MapGenerator::cellular_automata(const int generations) -> MapGenerator & {
  for (int _{0}; _ < generations; _++) {
    auto temp_grid{std::vector(grid_.grid)};
    for (auto i{0}; i < grid_.width * grid_.height; i++) {
      const auto floor_neighbours{count_floor_neighbours(grid_, grid_.convert_position(i))};
      temp_grid.at(i) = floor_neighbours >= MIN_NEIGHBOUR_DISTANCE ? GameObjectType::Floor : GameObjectType::Empty;
    }
    grid_.grid = std::move(temp_grid);
  }
  return *this;
}

auto MapGenerator::generate_walls() -> MapGenerator & {
  auto is_edge_or_non_floor{[this](const Position &position) {
    return position.x == 0 || position.y == 0 || position.x == grid_.width - 1 || position.y == grid_.height - 1 ||
           grid_.get_value(position) != GameObjectType::Floor;
  }};

  for (auto i{0}; i < grid_.width * grid_.height; i++) {
    if (const Position position{grid_.convert_position(i)};
        is_edge_or_non_floor(position) && count_floor_neighbours(grid_, position) > 0) {
      grid_.set_value(position, GameObjectType::Wall);
    }
  }
  return *this;
}

auto MapGenerator::place_obstacles() -> MapGenerator & {
  place_tiles(grid_, random_generator_, GameObjectType::Obstacle, 1.0, OBSTACLE_COUNT.generate_value(level_));
  return *this;
}

auto MapGenerator::place_player() -> MapGenerator & {
  place_tiles(grid_, random_generator_, GameObjectType::Player, 1.0, 1);
  return *this;
}

auto MapGenerator::place_items() -> MapGenerator & {
  for (const auto &[tile, probability] : ITEM_CHANCES) {
    place_tiles(grid_, random_generator_, tile, probability);
  }
  return *this;
}

auto MapGenerator::place_goal() -> MapGenerator & {
  const auto player_iter{std::ranges::find(grid_.grid.begin(), grid_.grid.end(), GameObjectType::Player)};
  const auto player_index{static_cast<int>(std::distance(grid_.grid.begin(), player_iter))};
  grid_.set_value(get_furthest_position(grid_, grid_.convert_position(player_index)), GameObjectType::Goal);
  return *this;
}

auto MapGenerator::place_lobby() -> MapGenerator & {
  grid_.place_rect({{.x = 0, .y = 0}, {.x = LOBBY_WIDTH - 1, .y = LOBBY_HEIGHT - 1}});
  *this = generate_walls();
  grid_.set_value(LOBBY_PLAYER_POSITION, GameObjectType::Player);
  grid_.set_value(LOBBY_GOAL_POSITION, GameObjectType::Goal);
  grid_.set_value(LOBBY_SHOP_POSITION, GameObjectType::Shop);
  return *this;
}

auto MapGenerator::get_enemy_limit(const int level) -> int { return ENEMY_LIMIT.generate_value(level); }
