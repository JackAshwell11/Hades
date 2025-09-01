// Std headers
#include <utility>

// Related header
#include "game_state.hpp"

// Local headers
#include "ecs/registry.hpp"
#include "ecs/systems/inventory.hpp"
#include "ecs/systems/level.hpp"
#include "ecs/systems/movements.hpp"
#include "ecs/systems/physics.hpp"
#include "events.hpp"
#include "factories.hpp"
#include "generation/map.hpp"
#include "generation/primitives.hpp"

namespace {
/// The deviation of the level distribution.
constexpr int LEVEL_DISTRIBUTION_DEVIATION{2};

/// The number of cellular automata runs to perform.
constexpr int CELLULAR_AUTOMATA_SIMULATIONS{3};

/// The distance from the player to generate an enemy.
constexpr double ENEMY_GENERATION_DISTANCE{5 * SPRITE_SIZE};

/// The hasher for the seed string.
constexpr std::hash<std::string> seed_hasher;
}  // namespace

GameState::GameState(const std::shared_ptr<Registry>& registry) : registry_(registry) {
  if (get_player_id() == -1) {
    game_state_.game.player_id = registry_->create_game_object(GameObjectType::Player, cpvzero,
                                                               get_game_object_components(GameObjectType::Player));
  }
  initialise_dungeon_run();
}

auto GameState::get_player_id() const -> GameObjectID { return game_state_.game.player_id; }

auto GameState::get_nearest_item() const -> GameObjectID { return game_state_.current_level.nearest_item; }

void GameState::set_nearest_item(const GameObjectID item_id) { game_state_.current_level.nearest_item = item_id; }

auto GameState::get_dungeon_level() const -> LevelType { return game_state_.current_level.dungeon_level; }

auto GameState::get_game_level() const -> int { return game_state_.dungeon_run.game_level; }

auto GameState::is_lobby() const -> bool { return game_state_.current_level.dungeon_level == LevelType::Lobby; }

auto GameState::is_boss() const -> bool { return game_state_.current_level.dungeon_level == LevelType::Boss; }

auto GameState::get_enemy_generation_timer() const -> double {
  return game_state_.current_level.enemy_generation_timer;
}

void GameState::set_enemy_generation_timer(const double value) {
  game_state_.current_level.enemy_generation_timer = value;
}

auto GameState::is_player_touching_type(const GameObjectType game_object_type) const -> bool {
  return get_nearest_item() != -1 && registry_->get_game_object_type(get_nearest_item()) == game_object_type;
}

void GameState::set_seed(const std::string& seed) {
  game_state_.dungeon_run.random_generator.seed(static_cast<unsigned int>(seed_hasher(seed)));
}

void GameState::initialise_dungeon_run() {
  const auto player_level{registry_->get_component<PlayerLevel>(get_player_id())->level};
  game_state_.dungeon_run = {
      .game_level = player_level,
      .level_distribution = std::normal_distribution<>(player_level, LEVEL_DISTRIBUTION_DEVIATION)};
}

void GameState::reset_level(const LevelType level_type) {
  // Reset the game state while preserving inventory items if necessary
  std::unordered_set preserved_ids{get_player_id()};
  if (level_type != LevelType::Lobby) {
    const auto inventory{registry_->get_component<Inventory>(get_player_id())};
    preserved_ids.insert(inventory->items.begin(), inventory->items.end());
  } else {
    for (const auto& component : registry_->get_game_object_components(get_player_id())) {
      component->reset();
    }
  }
  registry_->clear_game_objects(preserved_ids);

  // Set up the player and game state
  game_state_.current_level = {.floor_positions = {}, .dungeon_level = level_type};
  if (level_type == LevelType::Lobby) {
    initialise_dungeon_run();
  }

  // Create the game objects for the level
  Grid grid(0, 0);
  if (level_type == LevelType::Lobby) {
    grid = MapGenerator{}.place_lobby().get_grid();
  } else {
    grid = MapGenerator{get_game_level(), game_state_.dungeon_run.random_generator}
               .generate_rooms()
               .place_obstacles()
               .create_connections()
               .generate_hallways()
               .cellular_automata(CELLULAR_AUTOMATA_SIMULATIONS)
               .generate_walls()
               .place_player()
               .place_items()
               .place_goal()
               .get_grid();
  }
  create_game_objects(grid, level_type != LevelType::Lobby);

  // Notify the Python module of the level reset
  if (level_type == LevelType::Lobby) {
    notify<EventType::InventoryUpdate>(std::vector<GameObjectID>{});
    notify<EventType::RangedAttackSwitch>(0);
    notify<EventType::AttackCooldownUpdate>(get_player_id(), 0.0, 0.0, 0.0);
    notify<EventType::StatusEffectUpdate>(std::unordered_map<EffectType, double>{});
  }
  notify<EventType::GameOpen>();
}

void GameState::generate_enemy() {
  // Get a random floor position and check if it is valid for enemy generation
  const auto& current_level{game_state_.current_level};
  if (current_level.floor_positions.empty()) {
    return;
  }
  auto dist{std::uniform_int_distribution<size_t>(0, current_level.floor_positions.size() - 1)};
  const auto grid_position{current_level.floor_positions.at(dist(game_state_.dungeon_run.random_generator))};
  const auto world_position{grid_pos_to_pixel(grid_position)};
  const bool intersecting_enemies{
      cpSpacePointQueryNearest(registry_->get_space(), world_position, 0.0,
                               {CP_NO_GROUP, CP_ALL_CATEGORIES, static_cast<cpBitmask>(GameObjectType::Enemy)},
                               nullptr) != nullptr};
  if (const auto player_position{
          cpBodyGetPosition(*registry_->get_component<KinematicComponent>(get_player_id())->body)};
      intersecting_enemies || cpvdist(world_position, player_position) < ENEMY_GENERATION_DISTANCE) {
    return;
  }

  // Generate the enemy at the position
  const int enemy_level{
      static_cast<int>(game_state_.dungeon_run.level_distribution(game_state_.dungeon_run.random_generator))};
  const auto enemy_id{
      registry_->create_game_object(GameObjectType::Enemy, grid_position,
                                    get_game_object_components(GameObjectType::Enemy, std::max(0, enemy_level)))};
  registry_->get_component<SteeringMovement>(enemy_id)->target_id = get_player_id();
}

void GameState::create_game_objects(const Grid& grid, const bool store_floor_positions) {
  for (auto i{0}; std::cmp_less(i, grid.grid.size()); i++) {
    const auto game_object_type{grid.grid[i]};
    if (game_object_type == GameObjectType::Empty || game_object_type == GameObjectType::Obstacle) {
      continue;
    }

    // Get the game object's position
    const auto [x, y]{grid.convert_position(i)};
    const auto position{cpv(x, y)};

    // Store the floor position for enemy generation
    if (game_object_type != GameObjectType::Wall) {
      if (store_floor_positions) {
        game_state_.current_level.floor_positions.emplace_back(position);
      }
      if (game_object_type != GameObjectType::Floor) {
        registry_->create_game_object(GameObjectType::Floor, position,
                                      get_game_object_components(GameObjectType::Floor));
      }
    }

    // Handle game object creation
    if (game_object_type == GameObjectType::Player) {
      cpBodySetPosition(*registry_->get_component<KinematicComponent>(get_player_id())->body,
                        grid_pos_to_pixel(position));
    } else {
      registry_->create_game_object(game_object_type, position, get_game_object_components(game_object_type));
    }
  }
}
