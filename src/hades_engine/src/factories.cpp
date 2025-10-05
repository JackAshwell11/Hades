// Related header
#include "factories.hpp"

// Local headers
#include "ecs/registry.hpp"
#include "ecs/stats.hpp"
#include "ecs/systems/attacks.hpp"
#include "ecs/systems/effects.hpp"
#include "ecs/systems/inventory.hpp"
#include "ecs/systems/level.hpp"
#include "ecs/systems/movements.hpp"
#include "ecs/systems/physics.hpp"
#include "ecs/systems/shop.hpp"
#include "events.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
namespace {
/// Alias for a factory function that creates components for a game object.
using ComponentFactory = std::function<void(Registry*, GameObjectID, cpVect)>;

/// Alias for a factory function that creates components for a game object with a given level.
using LeveledComponentFactory = std::function<void(Registry*, GameObjectID, cpVect, int)>;

/// The velocity of the bullet.
constexpr int BULLET_VELOCITY{1000};

/// Get the hitboxes for the game objects.
///
/// @return The hitboxes for the game objects.
auto get_hitboxes() -> auto& {
  static std::unordered_map<GameObjectType, std::vector<cpVect>> hitboxes;
  return hitboxes;
}

/// Add the components for the bullet game object.
const auto bullet_factory{[](Registry* registry, const GameObjectID game_object_id, const cpVect& position) {
  registry->add_component<Bullet>(game_object_id);
  registry->add_component<KinematicComponent>(game_object_id, position);
}};

/// Add the components for the enemy game object.
const auto enemy_factory{[](Registry* registry, const GameObjectID game_object_id, const cpVect& position, const int) {
  std::vector<std::unique_ptr<RangedAttack>> attacks;
  attacks.push_back(std::make_unique<SingleBulletAttack>(1, 10, 3 * SPRITE_SIZE, BULLET_VELOCITY));
  registry->add_component<Armour>(game_object_id, 50);
  registry->add_component<Attack>(game_object_id, std::move(attacks));
  registry->add_component<Health>(game_object_id, 100);
  registry->add_component<KinematicComponent>(game_object_id, position, get_hitboxes().at(GameObjectType::Enemy));
  registry->add_component<SteeringMovement>(
      game_object_id,
      std::unordered_map<SteeringMovementState, std::vector<SteeringBehaviours>>{
          {SteeringMovementState::Default, {SteeringBehaviours::ObstacleAvoidance, SteeringBehaviours::Wander}},
          {SteeringMovementState::Footprint, {SteeringBehaviours::FollowPath}},
          {SteeringMovementState::Target, {SteeringBehaviours::Pursue}}});
}};

/// Add the components for the wall game object.
const auto wall_factory{[](Registry* registry, const GameObjectID game_object_id, const cpVect& position) {
  registry->add_component<KinematicComponent>(game_object_id, position, true);
}};

/// Add the components for the floor game object.
const auto floor_factory{[](Registry*, const GameObjectID, const cpVect&) {}};

/// Add the components for the goal game object.
const auto goal_factory{[](Registry* registry, const GameObjectID game_object_id, const cpVect& position) {
  registry->add_component<KinematicComponent>(game_object_id, position);
}};

/// Add the components for the player game object.
const auto player_factory{[](Registry* registry, const GameObjectID game_object_id, const cpVect& position) {
  std::vector<std::unique_ptr<RangedAttack>> attacks;
  attacks.push_back(std::make_unique<SingleBulletAttack>(1, 20, 3 * SPRITE_SIZE, BULLET_VELOCITY));
  attacks.push_back(std::make_unique<MultiBulletAttack>(1, 10, 3 * SPRITE_SIZE, BULLET_VELOCITY, 5));
  registry->add_component<Armour>(game_object_id, 100);
  registry->add_component<Attack>(game_object_id, std::move(attacks));
  registry->add_component<Footprints>(game_object_id);
  registry->add_component<Health>(game_object_id, 200);
  registry->add_component<Inventory>(game_object_id);
  registry->add_component<KeyboardMovement>(game_object_id);
  registry->add_component<KinematicComponent>(game_object_id, position, get_hitboxes().at(GameObjectType::Player));
  registry->add_component<Money>(game_object_id);
  registry->add_component<PlayerLevel>(game_object_id);
  registry->add_component<StatusEffects>(game_object_id);
}};

/// Add the components for the health potion game object.
const auto health_potion_factory{[](Registry* registry, const GameObjectID game_object_id, const cpVect& position) {
  registry->add_component<EffectApplier>(game_object_id);
  const auto effect_applier{registry->get_component<EffectApplier>(game_object_id)};
  effect_applier->add_status_effect(EffectType::Regeneration, 5, 10, 1);
  registry->add_component<KinematicComponent>(game_object_id, position);
}};

/// Add the components for the chest game object,
const auto chest_factory{[](Registry* registry, const GameObjectID game_object_id, const cpVect& position) {
  registry->add_component<KinematicComponent>(game_object_id, position);
}};
}  // namespace
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

auto load_hitbox(const GameObjectType game_object_type, const std::vector<std::pair<double, double>>& hitbox) -> bool {
  if (get_hitboxes().contains(game_object_type)) {
    return false;
  }
  std::vector<cpVect> hitbox_points;
  hitbox_points.reserve(hitbox.size());
  for (const auto& [x, y] : hitbox) {
    hitbox_points.push_back(cpv(x, y) * SPRITE_SCALE);
  }
  get_hitboxes()[game_object_type] = hitbox_points;
  return true;
}

void clear_hitboxes() { get_hitboxes().clear(); }

auto create_game_object(Registry* registry, const GameObjectType game_object_type, const cpVect& position,
                        const std::optional<int> level) -> GameObjectID {
  // Define the factories for each game object type
  static const std::unordered_map<GameObjectType, ComponentFactory> factories{
      {GameObjectType::Bullet, bullet_factory}, {GameObjectType::Floor, floor_factory},
      {GameObjectType::Player, player_factory}, {GameObjectType::Wall, wall_factory},
      {GameObjectType::Goal, goal_factory},     {GameObjectType::HealthPotion, health_potion_factory},
      {GameObjectType::Chest, chest_factory}};
  static const std::unordered_map<GameObjectType, LeveledComponentFactory> leveled_factories{
      {GameObjectType::Enemy, enemy_factory},
  };

  GameObjectID game_object_id{-1};
  const auto game_object_position{game_object_type == GameObjectType::Bullet ? position : grid_pos_to_pixel(position)};
  if (const auto player_ids{registry->get_game_object_ids(GameObjectType::Player)};
      game_object_type != GameObjectType::Player || player_ids.empty()) {
    // Make a new game object
    game_object_id = registry->create_game_object(game_object_type);
    if (factories.contains(game_object_type)) {
      factories.at(game_object_type)(registry, game_object_id, game_object_position);
    }
    if (leveled_factories.contains(game_object_type) && level.has_value()) {
      leveled_factories.at(game_object_type)(registry, game_object_id, game_object_position, level.value());
    }
  } else {
    // Just move the existing player
    game_object_id = player_ids.at(0);
    cpBodySetPosition(*registry->get_component<KinematicComponent>(player_ids.at(0))->body, game_object_position);
  }
  notify<EventType::PositionChanged>(game_object_id, std::make_pair(game_object_position.x, game_object_position.y));
  return game_object_id;
}
