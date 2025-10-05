// Related header
#include "ecs/systems/attacks.hpp"

// Std headers
#include <numbers>
#include <utility>

// Local headers
#include "ecs/registry.hpp"
#include "ecs/stats.hpp"
#include "ecs/systems/movements.hpp"
#include "events.hpp"

namespace {
/// The size of the cone angle for the multi-bullet attack (45 degrees).
constexpr auto MULTI_BULLET_CONE_ANGLE{std::numbers::pi / 4};

/// Create multiple bullets spread out in a cone shape around the direction of the attack.
///
/// @param registry - The registry that manages the game objects, components, and systems.
/// @param game_object_id - The game object ID of the attacking game object.
/// @param bullet_count - The number of bullets to create.
/// @param velocity - The velocity of the bullets.
/// @param damage - The damage of the bullets.
/// @throws RegistryError - If the game object ID is not registered with the registry or does not have a kinematic
/// component.
void create_bullet_cone(const Registry* registry, const GameObjectID game_object_id, const int bullet_count,
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

void SingleBulletAttack::perform_attack(const Registry* registry, const GameObjectID game_object_id) const {
  create_bullet_cone(registry, game_object_id, 1, velocity, damage);
}

void MultiBulletAttack::perform_attack(const Registry* registry, const GameObjectID game_object_id) const {
  create_bullet_cone(registry, game_object_id, bullet_count, velocity, damage);
}

void AttackSystem::update(const double delta_time) const {
  for (const auto& [game_object_id, component_tuple] : get_registry()->get_game_object_components<Attack>()) {
    const auto [attack]{component_tuple};
    for (const auto& ranged_attack : attack->ranged_attacks) {
      ranged_attack->update(delta_time);
    }
    notify<EventType::AttackCooldownUpdate>(
        game_object_id, std::cmp_less(attack->selected_ranged_attack, attack->ranged_attacks.size())
                            ? attack->ranged_attacks[attack->selected_ranged_attack]->get_time_until_attack()
                            : 0.0);

    // If the game object has a steering movement component, they are in the target state, and their cooldown is up,
    // then attack
    if (get_registry()->has_component<SteeringMovement>(game_object_id) && !attack->ranged_attacks.empty()) {
      if (const auto steering_movement{get_registry()->get_component<SteeringMovement>(game_object_id)};
          steering_movement->movement_state == SteeringMovementState::Target) {
        (void)do_attack(game_object_id);
      }
    }
  }
}

void AttackSystem::previous_ranged_attack(const GameObjectID game_object_id) const {
  if (const auto attack{get_registry()->get_component<Attack>(game_object_id)}; attack->selected_ranged_attack > 0) {
    attack->selected_ranged_attack--;
    notify<EventType::RangedAttackSwitch>(attack->selected_ranged_attack);
  }
}

void AttackSystem::next_ranged_attack(const GameObjectID game_object_id) const {
  if (const auto attack{get_registry()->get_component<Attack>(game_object_id)};
      !attack->ranged_attacks.empty() &&
      std::cmp_less(attack->selected_ranged_attack, attack->ranged_attacks.size() - 1)) {
    attack->selected_ranged_attack++;
    notify<EventType::RangedAttackSwitch>(attack->selected_ranged_attack);
  }
}

auto AttackSystem::do_attack(const GameObjectID game_object_id) const -> bool {
  // Check if the game object can attack or not
  const auto attack{get_registry()->get_component<Attack>(game_object_id)};
  if (std::cmp_greater_equal(attack->selected_ranged_attack, attack->ranged_attacks.size())) {
    return false;
  }
  BaseAttack* attack_obj{attack->get_selected_ranged_attack()};
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
  const auto old_health{health->get_value()};
  const auto old_armour{armour->get_value()};
  health->set_value(health->get_value() - std::max(damage - armour->get_value(), 0.0));
  armour->set_value(armour->get_value() - damage);
  if (health->get_value() != old_health) {
    notify<EventType::HealthChanged>(game_object_id, health->get_value() / health->get_max_value());
  }
  if (armour->get_value() != old_armour) {
    notify<EventType::ArmourChanged>(game_object_id, armour->get_value() / armour->get_max_value());
  }

  // If the health is now 0, delete the game object
  if (health->get_value() <= 0) {
    get_registry()->mark_for_deletion(game_object_id);
  }
}
