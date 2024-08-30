// Related header
#include "ecs/systems/attacks.hpp"

// Std headers
#include <numbers>

// External headers
#include <chipmunk/chipmunk_structs.h>

// Local headers
#include "ecs/stats.hpp"
#include "ecs/systems/movements.hpp"
#include "ecs/systems/physics.hpp"

// ----- CONSTANTS -------------------------------
constexpr int BULLET_VELOCITY{500};

// ----- FUNCTIONS ------------------------------
/// Performs an area of effect attack around the game object.
///
/// @param registry - The registry that manages the game objects, components, and systems.
/// @param attacker_id - The game object ID of the attacker.
/// @param current_position - The current position of the game object.
/// @param targets - The targets to attack.
void area_of_effect_attack(const Registry *registry, const GameObjectID attacker_id, const cpVect &current_position,
                           const std::vector<int> &targets) {
  // Find all targets that are within range and attack them
  for (const auto target : targets) {
    if (cpvdist(current_position, registry->get_component<KinematicComponent>(target)->body->p) <=
        registry->get_component<AttackRange>(attacker_id)->get_value()) {
      registry->get_system<DamageSystem>()->deal_damage(target, attacker_id);
    }
  }
}

/// Performs a melee attack in the direction the game object is facing.
///
/// @param registry - The registry that manages the game objects, components, and systems.
/// @param attacker_id - The game object ID of the attacker.
/// @param current_position - The current position of the game object.
/// @param current_rotation - The current rotation of the game object in radians.
/// @param targets - The targets to attack.
void melee_attack(const Registry *registry, const GameObjectID attacker_id, const cpVect &current_position,
                  const double current_rotation, const std::vector<int> &targets) {
  // Convert the rotation into a direction vector
  const auto direction{cpvforangle(current_rotation)};

  // Find all targets that can be attacked
  for (const auto target : targets) {
    // Calculate the target direction and distance
    const auto target_position{registry->get_component<KinematicComponent>(target)->body->p};
    const auto target_direction{cpvsub(target_position, current_position)};

    // Check if the target is within the attack range and circle sector
    if (const auto distance{cpvdist(current_position, target_position)};
        distance <= registry->get_component<AttackRange>(attacker_id)->get_value()) {
      const auto melee_attack_size{registry->get_component<MeleeAttackSize>(attacker_id)->get_value()};
      if (const auto theta{std::atan2(cpvcross(direction, target_direction), cpvdot(direction, target_direction))};
          theta >= -melee_attack_size && theta <= melee_attack_size) {
        registry->get_system<DamageSystem>()->deal_damage(target, attacker_id);
      }
    }
  }
}

/// Performs a ranged attack in the direction the game object is facing.
///
/// @param current_position - The current position of the game object.
/// @param current_rotation - The current rotation of the game object in radians.
/// @return The bullet's position and velocity.
auto ranged_attack(const cpVect &current_position, const double current_rotation) -> std::pair<cpVect, cpVect> {
  const auto direction{cpvforangle(current_rotation)};
  return {current_position + direction * SPRITE_SIZE, direction * BULLET_VELOCITY};
}

void AttackSystem::update(const double delta_time) const {
  for (const auto &[game_object_id, component_tuple] : get_registry()->find_components<Attack>()) {
    // Update the time since the last attack
    const auto [attack] = component_tuple;
    attack->time_since_last_attack += delta_time;

    // If the game object has a steering movement component, they are in the target state, and their cooldown is up,
    // then attack
    if (get_registry()->has_component(game_object_id, typeid(SteeringMovement))) {
      if (const auto steering_movement{get_registry()->get_component<SteeringMovement>(game_object_id)};
          steering_movement->movement_state == SteeringMovementState::Target) {
        do_attack(game_object_id, {steering_movement->target_id});
      }
    }
  }
}

void AttackSystem::do_attack(const GameObjectID game_object_id, const std::vector<int> &targets) const {
  // Check if the game object can attack or not
  const auto attack{get_registry()->get_component<Attack>(game_object_id)};
  if (attack->time_since_last_attack < get_registry()->get_component<AttackCooldown>(game_object_id)->get_value()) {
    return;
  }

  // Check if the game object has any attacks to perform
  if (attack->attack_state >= static_cast<int>(attack->attack_algorithms.size())) {
    return;
  }

  // Perform the selected attack on the targets
  attack->time_since_last_attack = 0;
  const auto kinematic_component{get_registry()->get_component<KinematicComponent>(game_object_id)};
  switch (attack->attack_algorithms[attack->attack_state]) {
    case AttackAlgorithm::AreaOfEffect:
      area_of_effect_attack(get_registry(), game_object_id, kinematic_component->body->p, targets);
      break;
    case AttackAlgorithm::Melee:
      melee_attack(get_registry(), game_object_id, kinematic_component->body->p, kinematic_component->rotation,
                   targets);
      break;
    case AttackAlgorithm::Ranged:
      get_registry()->notify_callbacks(EventType::BulletCreation,
                                       get_registry()->get_system<PhysicsSystem>()->add_bullet(
                                           ranged_attack(kinematic_component->body->p, kinematic_component->rotation),
                                           get_registry()->get_component<Damage>(game_object_id)->get_value()));
  }
}

void DamageSystem::deal_damage(const GameObjectID game_object_id, const GameObjectID attacker_id) const {
  // Damage the armour and carry over the extra damage to the health
  const auto health{get_registry()->get_component<Health>(game_object_id)};
  const auto armour{get_registry()->get_component<Armour>(game_object_id)};
  const auto damage{get_registry()->get_component<Damage>(attacker_id)->get_value()};
  health->set_value(health->get_value() - std::max(damage - armour->get_value(), 0.0));
  armour->set_value(armour->get_value() - damage);

  // If the health is now 0, delete the game object
  if (health->get_value() <= 0) {
    get_registry()->delete_game_object(game_object_id);
  }
}
