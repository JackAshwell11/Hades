// Related header
#include "game_engine.hpp"

// Std headers
#include <fstream>

// External headers
#include <nlohmann/json.hpp>

// Local headers
#include "ecs/systems/armour_regen.hpp"
#include "ecs/systems/attacks.hpp"
#include "ecs/systems/effects.hpp"
#include "ecs/systems/inventory.hpp"
#include "ecs/systems/movements.hpp"
#include "ecs/systems/physics.hpp"
#include "ecs/systems/shop.hpp"
#include "generation/map.hpp"

namespace {
/// The interval in seconds between enemy generations.
constexpr double ENEMY_GENERATION_INTERVAL{1.0};
}  // namespace

GameEngine::GameEngine()
    : registry_(std::make_shared<Registry>()),
      game_state_(std::make_shared<GameState>(registry_)),
      input_handler_(std::make_shared<InputHandler>(registry_, game_state_)),
      save_manager_(std::make_shared<SaveManager>(registry_, game_state_)) {
  registry_->add_system<ArmourRegenSystem>();
  registry_->add_system<AttackSystem>();
  registry_->add_system<DamageSystem>();
  registry_->add_system<EffectSystem>();
  registry_->add_system<FootprintSystem>();
  registry_->add_system<InventorySystem>();
  registry_->add_system<KeyboardMovementSystem>();
  registry_->add_system<PhysicsSystem>();
  registry_->add_system<ShopSystem>();
  registry_->add_system<SteeringMovementSystem>();
}

void GameEngine::on_update(const double delta_time) const {
  game_state_->set_nearest_item(registry_->get_system<PhysicsSystem>()->get_nearest_item(game_state_->get_player_id()));
  if (game_state_->is_player_touching_type(GameObjectType::Goal) && !game_state_->is_lobby()) {
    if (const auto dungeon_level{game_state_->get_dungeon_level()}; dungeon_level == LevelType::FirstDungeon) {
      game_state_->reset_level(LevelType::SecondDungeon);
    } else if (dungeon_level == LevelType::SecondDungeon) {
      game_state_->reset_level(LevelType::Boss);
    } else {
      game_state_->reset_level(LevelType::Lobby);
    }
    return;
  }
  game_state_->set_enemy_generation_timer(game_state_->get_enemy_generation_timer() + delta_time);
  if (game_state_->get_enemy_generation_timer() >= ENEMY_GENERATION_INTERVAL &&
      std::cmp_less(registry_->get_game_object_ids(GameObjectType::Enemy).size(),
                    MapGenerator::get_enemy_limit(game_state_->get_game_level()))) {
    game_state_->generate_enemy();
    game_state_->set_enemy_generation_timer(0.0);
  }
}

void GameEngine::on_fixed_update(const double delta_time) const { registry_->update(delta_time); }
