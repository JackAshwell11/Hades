// Related header
#include "ecs/systems/attacks.hpp"

// Std headers
#include <numbers>
#include <utility>

// Local headers
#include "ecs/systems/movements.hpp"
#include "ecs/systems/physics.hpp"

namespace {
/// The size of the cone angle for the multi-bullet attack (45 degrees).
constexpr auto MULTI_BULLET_CONE_ANGLE{std::numbers::pi / 4};

/// Get the target IDs for an attack based on the game object type.
///
/// @param registry - The registry that manages the game objects, components, and systems.
/// @param game_object_id - The game object ID of the attacking game object.
/// @throws RegistryError - If the game object ID is not registered with the registry.
/// @return The target IDs for the attack.
auto get_target_ids(const Registry *registry, const GameObjectID game_object_id) -> std::vector<GameObjectID> {
  return registry->get_game_object_ids(registry->get_game_object_type(game_object_id) == GameObjectType::Player
                                           ? GameObjectType::Enemy
                                           : GameObjectType::Player);
}

/// Create multiple bullets spread out in a cone shape around the direction of the attack.
///
/// @param registry - The registry that manages the game objects, components, and systems.
/// @param game_object_id - The game object ID of the attacking game object.
/// @param bullet_count - The number of bullets to create.
/// @param velocity - The velocity of the bullets.
/// @param damage - The damage of the bullets.
/// @throws RegistryError - If the game object ID is not registered with the registry or does not have a kinematic
/// component.
void create_bullet_cone(const Registry *registry, const GameObjectID game_object_id, const int bullet_count,
                        const double velocity, const double damage) {
  const auto kinematic_component{registry->get_component<KinematicComponent>(game_object_id)};
  const auto direction{cpvforangle(kinematic_component->rotation)};
  const auto angle_step{bullet_count > 1 ? 2 * MULTI_BULLET_CONE_ANGLE / (bullet_count - 1) : 0.0};
  for (int i{0}; i < bullet_count; i++) {
    const auto bullet_angle{bullet_count > 1 ? -MULTI_BULLET_CONE_ANGLE + (i * angle_step) : 0.0};
    const auto bullet_position{cpBodyGetPosition(*kinematic_component->body) + direction * SPRITE_SIZE};
    const auto bullet_velocity{cpvrotate(direction, cpvforangle(bullet_angle)) * velocity};
    registry->get_system<PhysicsSystem>()->add_bullet({bullet_position, bullet_velocity}, damage,
                                                      registry->get_game_object_type(game_object_id));
  }
}
}  // namespace

void SingleBulletAttack::perform_attack(const Registry *registry, const GameObjectID game_object_id) const {
  create_bullet_cone(registry, game_object_id, 1, velocity.get_value(), damage.get_value());
}

void MultiBulletAttack::perform_attack(const Registry *registry, const GameObjectID game_object_id) const {
  create_bullet_cone(registry, game_object_id, static_cast<int>(bullet_count.get_value()), velocity.get_value(),
                     damage.get_value());
}

void MeleeAttack::perform_attack(const Registry *registry, const GameObjectID game_object_id) const {
  const auto kinematic_component{registry->get_component<KinematicComponent>(game_object_id)};
  const auto current_position{cpBodyGetPosition(*kinematic_component->body)};
  const auto direction{cpvforangle(kinematic_component->rotation)};
  for (const auto target : get_target_ids(registry, game_object_id)) {
    const auto target_position{cpBodyGetPosition(*registry->get_component<KinematicComponent>(target)->body)};
    const auto target_direction{cpvsub(target_position, current_position)};

    // Check if the target is within the attack range and circle sector
    if (const auto distance{cpvdist(current_position, target_position)}; distance <= range.get_value()) {
      if (const auto theta{std::atan2(cpvcross(direction, target_direction), cpvdot(direction, target_direction))};
          theta >= -size.get_value() && theta <= size.get_value()) {
        registry->get_system<DamageSystem>()->deal_damage(target, damage.get_value());
      }
    }
  }
}

void AreaOfEffectAttack::perform_attack(const Registry *registry, const GameObjectID game_object_id) const {
  const auto kinematic_component{registry->get_component<KinematicComponent>(game_object_id)};
  for (const auto target : get_target_ids(registry, game_object_id)) {
    if (cpvdist(cpBodyGetPosition(*kinematic_component->body),
                cpBodyGetPosition(*registry->get_component<KinematicComponent>(target)->body)) <= range.get_value()) {
      registry->get_system<DamageSystem>()->deal_damage(target, damage.get_value());
    }
  }
}

void AttackSystem::update(const double delta_time) const {
  for (const auto &[game_object_id, component_tuple] : get_registry()->find_components<Attack>()) {
    const auto [attack]{component_tuple};
    for (const auto &ranged_attack : attack->ranged_attacks) {
      ranged_attack->update(delta_time);
    }
    if (attack->melee_attack) {
      attack->melee_attack->update(delta_time);
    }
    if (attack->special_attack) {
      attack->special_attack->update(delta_time);
    }
    get_registry()->notify<EventType::AttackCooldownUpdate>(
        game_object_id,
        std::cmp_less(attack->selected_ranged_attack, attack->ranged_attacks.size())
            ? attack->ranged_attacks[attack->selected_ranged_attack]->get_time_until_attack()
            : 0.0,
        attack->melee_attack ? attack->melee_attack->get_time_until_attack() : 0.0,
        attack->special_attack ? attack->special_attack->get_time_until_attack() : 0.0);

    // If the game object has a steering movement component, they are in the target state, and their cooldown is up,
    // then attack
    if (get_registry()->has_component(game_object_id, typeid(SteeringMovement)) && !attack->ranged_attacks.empty()) {
      if (const auto steering_movement{get_registry()->get_component<SteeringMovement>(game_object_id)};
          steering_movement->movement_state == SteeringMovementState::Target) {
        (void)do_attack(game_object_id, AttackType::Ranged);
      }
    }
  }
}

void AttackSystem::previous_ranged_attack(const GameObjectID game_object_id) const {
  if (const auto attack{get_registry()->get_component<Attack>(game_object_id)}; attack->selected_ranged_attack > 0) {
    attack->selected_ranged_attack--;
    get_registry()->notify<EventType::RangedAttackSwitch>(attack->selected_ranged_attack);
  }
}

void AttackSystem::next_ranged_attack(const GameObjectID game_object_id) const {
  if (const auto attack{get_registry()->get_component<Attack>(game_object_id)};
      !attack->ranged_attacks.empty() &&
      std::cmp_less(attack->selected_ranged_attack, attack->ranged_attacks.size() - 1)) {
    attack->selected_ranged_attack++;
    get_registry()->notify<EventType::RangedAttackSwitch>(attack->selected_ranged_attack);
  }
}

auto AttackSystem::do_attack(const GameObjectID game_object_id, const AttackType attack_type) const -> bool {
  // Get the attack object based on the attack type
  const auto attack{get_registry()->get_component<Attack>(game_object_id)};
  BaseAttack *attack_obj{[&]() -> BaseAttack * {
    switch (attack_type) {
      case AttackType::Ranged:
        return std::cmp_greater_equal(attack->selected_ranged_attack, attack->ranged_attacks.size())
                   ? nullptr
                   : attack->get_selected_ranged_attack();
      case AttackType::Melee:
        return attack->melee_attack ? &attack->melee_attack.value() : nullptr;
      case AttackType::Special:
        return attack->special_attack ? &attack->special_attack.value() : nullptr;
      default:
        return nullptr;
    }
  }()};

  // Check if the game object can attack or not
  if (attack_obj == nullptr || !attack_obj->is_ready()) {
    return false;
  }

  // Perform the selected attack on the targets
  attack_obj->time_since_last_use = 0;
  attack_obj->perform_attack(get_registry(), game_object_id);
  return true;
}

void DamageSystem::deal_damage(const GameObjectID game_object_id, const double damage) const {
  // Damage the armour and carry over the extra damage to the health
  const auto health{get_registry()->get_component<Health>(game_object_id)};
  const auto armour{get_registry()->get_component<Armour>(game_object_id)};
  health->set_value(health->get_value() - std::max(damage - armour->get_value(), 0.0));
  armour->set_value(armour->get_value() - damage);

  // If the health is now 0, delete the game object
  if (health->get_value() <= 0) {
    get_registry()->mark_for_deletion(game_object_id);
  }
}
