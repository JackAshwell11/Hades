// Related header
#include "factories.hpp"

// Std headers
#include <numbers>

// Local headers
#include "ecs/stats.hpp"
#include "ecs/systems/armour_regen.hpp"
#include "ecs/systems/attacks.hpp"
#include "ecs/systems/effects.hpp"
#include "ecs/systems/inventory.hpp"
#include "ecs/systems/movements.hpp"
#include "ecs/systems/physics.hpp"
#include "ecs/systems/sprite.hpp"
#include "ecs/systems/upgrade.hpp"

auto get_factories() -> const std::unordered_map<GameObjectType, ComponentFactory>& {
  static const std::unordered_map<GameObjectType, ComponentFactory> factories{
      {GameObjectType::Wall,
       [] {
         return std::vector<std::shared_ptr<ComponentBase>>{
             std::make_shared<KinematicComponent>(true),
         };
       }},
      {GameObjectType::Player,
       [] {
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
             std::make_shared<StatusEffect>(),
             std::make_shared<Upgrades>(std::unordered_map<std::type_index, std::pair<ActionFunction, ActionFunction>>{
                 {typeid(Health), std::make_pair([](const int level) { return std::pow(2, level) + 10; },
                                                 [](const int level) { return level * 10 + 5; })}}),
         };
       }},
      {GameObjectType::HealthPotion,
       [] {
         return std::vector<std::shared_ptr<ComponentBase>>{
             std::make_shared<EffectLevel>(1, 3),
             std::make_shared<PythonSprite>(),
         };
       }},
  };
  return factories;
}
