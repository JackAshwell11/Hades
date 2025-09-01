// Related header
#include "factories.hpp"

// Std headers
#include <functional>
#include <numbers>

// Local headers
#include "ecs/stats.hpp"
#include "ecs/systems/attacks.hpp"
#include "ecs/systems/effects.hpp"
#include "ecs/systems/inventory.hpp"
#include "ecs/systems/level.hpp"
#include "ecs/systems/movements.hpp"
#include "ecs/systems/physics.hpp"
#include "ecs/systems/shop.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
namespace {
/// Alias for a factory function that creates components for a game object.
using ComponentFactory = std::function<std::vector<std::shared_ptr<ComponentBase>>()>;

/// Alias for a factory function that creates components for a game object with a given level.
using LeveledComponentFactory = std::function<std::vector<std::shared_ptr<ComponentBase>>(int)>;

/// The velocity of the bullet.
constexpr int BULLET_VELOCITY{1000};

/// Get the hitboxes for the game objects.
///
/// @return The hitboxes for the game objects.
auto get_hitboxes() -> auto& {
  static std::unordered_map<GameObjectType, std::vector<cpVect>> hitboxes;
  return hitboxes;
}

/// The bullet factory.
///
/// @return The components for the bullet.
const auto bullet_factory{[] {
  return std::vector<std::shared_ptr<ComponentBase>>{std::make_shared<Bullet>(),
                                                     std::make_shared<KinematicComponent>()};
}};

/// The enemy factory.
///
/// @return The components for the enemy.
const auto enemy_factory{[](const int /*level*/) {
  const auto attack{std::make_shared<Attack>()};
  attack->add_ranged_attack(std::make_unique<SingleBulletAttack>(
      AttackStat{1, 3}, AttackStat{10, 3}, AttackStat{3 * SPRITE_SIZE, 3}, AttackStat{BULLET_VELOCITY, 1}));
  return std::vector<std::shared_ptr<ComponentBase>>{
      std::make_shared<Armour>(50, 5),
      attack,
      std::make_shared<Health>(100, 5),
      std::make_shared<KinematicComponent>(get_hitboxes().at(GameObjectType::Enemy)),
      std::make_shared<MovementForce>(1000, 5),
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
const auto wall_factory{
    [] { return std::vector<std::shared_ptr<ComponentBase>>{std::make_shared<KinematicComponent>(true)}; }};

/// The floor factory.
///
/// @return The components for the floor.
const auto floor_factory{[] { return std::vector<std::shared_ptr<ComponentBase>>{}; }};

/// The goal factory.
///
/// @return The components for the goal.
const auto goal_factory{
    [] { return std::vector<std::shared_ptr<ComponentBase>>{std::make_shared<KinematicComponent>()}; }};

/// The player factory.
///
/// @return The components for the player.
const auto player_factory{[] {
  const auto attack{std::make_shared<Attack>()};
  attack->add_ranged_attack(std::make_unique<SingleBulletAttack>(
      AttackStat{1, 3}, AttackStat{20, 3}, AttackStat{3 * SPRITE_SIZE, 3}, AttackStat{BULLET_VELOCITY, 1}));
  attack->add_ranged_attack(std::make_unique<MultiBulletAttack>(AttackStat{1, 3}, AttackStat{10, 3},
                                                                AttackStat{3 * SPRITE_SIZE, 3},
                                                                AttackStat{BULLET_VELOCITY, 1}, AttackStat{5, 3}));
  attack->set_melee_attack({{1, 3}, {20, 3}, {3 * SPRITE_SIZE, 3}, {std::numbers::pi / 4, 3}});
  attack->set_special_attack({{1, 3}, {20, 3}, {3 * SPRITE_SIZE, 3}});
  return std::vector<std::shared_ptr<ComponentBase>>{
      std::make_shared<Armour>(100, 5),
      attack,
      std::make_shared<Footprints>(),
      std::make_shared<FootprintInterval>(0.5, 3),
      std::make_shared<FootprintLimit>(5, 3),
      std::make_shared<Health>(200, 5),
      std::make_shared<Inventory>(),
      std::make_shared<InventorySize>(30, 3),
      std::make_shared<KeyboardMovement>(),
      std::make_shared<KinematicComponent>(get_hitboxes().at(GameObjectType::Player)),
      std::make_shared<Money>(),
      std::make_shared<MovementForce>(5000, 5),
      std::make_shared<PlayerLevel>(),
      std::make_shared<StatusEffects>(),
  };
}};

/// The health potion factory.
///
/// @return The components for the health potion.
const auto health_potion_factory{[] {
  auto effect_applier{std::make_shared<EffectApplier>()};
  effect_applier->add_status_effect(EffectType::Regeneration, 5, 10, 1);
  return std::vector<std::shared_ptr<ComponentBase>>{std::move(effect_applier), std::make_shared<KinematicComponent>()};
}};

/// The chest factory.
///
/// @return The components for the chest.
const auto chest_factory{
    [] { return std::vector<std::shared_ptr<ComponentBase>>{std::make_shared<KinematicComponent>()}; }};

/// The shop factory.
///
/// @return The components for the shop.
const auto shop_factory{
    [] { return std::vector<std::shared_ptr<ComponentBase>>{std::make_shared<KinematicComponent>()}; }};
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

auto get_game_object_components(const GameObjectType game_object_type, const int level)
    -> std::vector<std::shared_ptr<ComponentBase>> {
  static const std::unordered_map<GameObjectType, ComponentFactory> factories{
      {GameObjectType::Bullet, bullet_factory}, {GameObjectType::Floor, floor_factory},
      {GameObjectType::Player, player_factory}, {GameObjectType::Wall, wall_factory},
      {GameObjectType::Goal, goal_factory},     {GameObjectType::HealthPotion, health_potion_factory},
      {GameObjectType::Chest, chest_factory},   {GameObjectType::Shop, shop_factory},
  };
  static const std::unordered_map<GameObjectType, LeveledComponentFactory> leveled_factories{
      {GameObjectType::Enemy, enemy_factory},
  };
  if (factories.contains(game_object_type)) {
    return factories.at(game_object_type)();
  }
  if (leveled_factories.contains(game_object_type)) {
    return leveled_factories.at(game_object_type)(level);
  }
  return {};
}
