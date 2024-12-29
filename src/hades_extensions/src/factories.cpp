// Related header
#include "factories.hpp"

// Std headers
#include <numbers>

// Local headers
#include "ecs/stats.hpp"
#include "ecs/systems/attacks.hpp"
#include "ecs/systems/effects.hpp"
#include "ecs/systems/inventory.hpp"
#include "ecs/systems/movements.hpp"
#include "ecs/systems/physics.hpp"
#include "ecs/systems/sprite.hpp"
#include "ecs/systems/upgrade.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
namespace {
/// Get the hitboxes for the game objects.
///
/// @return The hitboxes for the game objects.
auto get_hitboxes() -> auto & {
  static std::unordered_map<GameObjectType, std::vector<cpVect>> hitboxes;
  return hitboxes;
}

/// The bullet factory.
///
/// @return The components for the bullet.
const auto bullet_factory{[](const int /*level*/) {
  return std::vector<std::shared_ptr<ComponentBase>>{
      std::make_shared<Damage>(10, -1),
      std::make_shared<KinematicComponent>(),
      std::make_shared<PythonSprite>(),
  };
}};

/// The enemy factory.
///
/// @return The components for the enemy.
const auto enemy_factory{[](const int /*level*/) {
  return std::vector<std::shared_ptr<ComponentBase>>{
      std::make_shared<Armour>(50, 5),
      std::make_shared<Attack>(std::vector{AttackAlgorithm::Ranged}),
      std::make_shared<AttackCooldown>(1, 3),
      std::make_shared<AttackRange>(3 * SPRITE_SIZE, 3),
      std::make_shared<Damage>(10, 3),
      std::make_shared<Health>(100, 5),
      std::make_shared<KinematicComponent>(get_hitboxes().at(GameObjectType::Enemy)),
      std::make_shared<MovementForce>(1000, 5),
      std::make_shared<PythonSprite>(),
      std::make_shared<SteeringMovement>(std::unordered_map<SteeringMovementState, std::vector<SteeringBehaviours>>{
          {SteeringMovementState::Default, {SteeringBehaviours::ObstacleAvoidance, SteeringBehaviours::Wander}},
          {SteeringMovementState::Footprint, {SteeringBehaviours::FollowPath}},
          {SteeringMovementState::Target, {SteeringBehaviours::Pursue}},
      }),
      std::make_shared<ViewDistance>(3 * SPRITE_SIZE, 3),
  };
}};

/// The wall factory.
///
/// @return The components for the wall.
const auto wall_factory{[](const int /*level*/) {
  return std::vector<std::shared_ptr<ComponentBase>>{
      std::make_shared<KinematicComponent>(true),

  };
}};

/// The floor factory.
///
/// @return The components for the floor.
const auto floor_factory{[](const int /*level*/) {
  return std::vector<std::shared_ptr<ComponentBase>>{
      std::make_shared<KinematicComponent>(),
  };
}};

/// The goal factory.
///
/// @return The components for the goal.
const auto goal_factory{[](const int /*level*/) {
  return std::vector<std::shared_ptr<ComponentBase>>{
      std::make_shared<KinematicComponent>(),
      std::make_shared<PythonSprite>(),
  };
}};

/// The player factory.
///
/// @return The components for the player.
const auto player_factory{[](const int /*level*/) {
  return std::vector<std::shared_ptr<ComponentBase>>{
      std::make_shared<Armour>(100, 5),
      std::make_shared<Attack>(
          std::vector{AttackAlgorithm::Ranged, AttackAlgorithm::Melee, AttackAlgorithm::AreaOfEffect}),
      std::make_shared<AttackCooldown>(1, 3),
      std::make_shared<AttackRange>(3 * SPRITE_SIZE, 3),
      std::make_shared<Damage>(20, 3),
      std::make_shared<Footprints>(),
      std::make_shared<FootprintInterval>(0.5, 3),
      std::make_shared<FootprintLimit>(5, 3),
      std::make_shared<Health>(200, 5),
      std::make_shared<Inventory>(),
      std::make_shared<InventorySize>(30, 3),
      std::make_shared<KeyboardMovement>(),
      std::make_shared<MeleeAttackSize>(std::numbers::pi / 4, 3),
      std::make_shared<Money>(),
      std::make_shared<MovementForce>(5000, 5),
      std::make_shared<PythonSprite>(),
      std::make_shared<KinematicComponent>(get_hitboxes().at(GameObjectType::Player)),
      std::make_shared<StatusEffect>(),
      std::make_shared<Upgrades>(std::unordered_map<std::type_index, std::pair<ActionFunction, ActionFunction>>{
          {typeid(Health), std::make_pair([](const int level) { return std::pow(2, level) + 10; },
                                          [](const int level) { return level * 10 + 5; })}}),
  };
}};

/// The health potion factory.
///
/// @return The components for the health potion.
const auto health_potion_factory{[](const int /*level*/) {
  return std::vector<std::shared_ptr<ComponentBase>>{
      std::make_shared<EffectLevel>(1, 3),
      std::make_shared<PythonSprite>(),
      std::make_shared<KinematicComponent>(),
  };
}};

/// The chest factory.
///
/// @return The components for the chest.
const auto chest_factory{[](const int /*level*/) {
  return std::vector<std::shared_ptr<ComponentBase>>{
      std::make_shared<PythonSprite>(),
      std::make_shared<KinematicComponent>(),
  };
}};
}  // namespace
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

auto load_hitbox(const GameObjectType game_object_type, const std::vector<std::pair<double, double>> &hitbox) -> bool {
  if (get_hitboxes().contains(game_object_type)) {
    return false;
  }
  std::vector<cpVect> hitbox_points;
  hitbox_points.reserve(hitbox.size());
  for (const auto &[x, y] : hitbox) {
    hitbox_points.push_back(cpv(x, y) * SPRITE_SCALE);
  }
  get_hitboxes()[game_object_type] = hitbox_points;
  return true;
}

void clear_hitboxes() { get_hitboxes().clear(); }

auto get_factories() -> const std::unordered_map<GameObjectType, ComponentFactory> & {
  static const std::unordered_map<GameObjectType, ComponentFactory> factories{
      {GameObjectType::Bullet, bullet_factory},
      {GameObjectType::Enemy, enemy_factory},
      {GameObjectType::Floor, floor_factory},
      {GameObjectType::Player, player_factory},
      {GameObjectType::Wall, wall_factory},
      {GameObjectType::Goal, goal_factory},
      {GameObjectType::HealthPotion, health_potion_factory},
      {GameObjectType::Chest, chest_factory},
  };
  return factories;
}
