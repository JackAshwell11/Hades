// Custom includes
#include "game_objects/systems/attacks.hpp"
#include "game_objects/systems/attributes.hpp"

// ----- CONSTANTS -------------------------------
const double ATTACK_RANGE = 3 * SPRITE_SIZE;
const int BULLET_VELOCITY = 300;
const int DAMAGE = 10;
const double MELEE_ATTACK_OFFSET_LOWER = 45 * PI_RADIANS;
const double MELEE_ATTACK_OFFSET_UPPER = (2 * (180 * PI_RADIANS)) - MELEE_ATTACK_OFFSET_LOWER;

// ----- STRUCTURES ------------------------------
AttackResult AttackSystem::do_attack(Registry &registry, int game_object_id, std::vector<int> &targets) {
  // Perform the attack on the targets
  auto *attacks = registry.get_component<Attacks>(game_object_id);
  const KinematicObject *kinematic_object = registry.get_kinematic_object(game_object_id);
  switch (attacks->attack_algorithms[attacks->attack_state]) {
    case AttackAlgorithms::AreaOfEffect:area_of_effect_attack(registry, kinematic_object->position, targets);
      break;
    case AttackAlgorithms::Melee:
      melee_attack(registry,
                   kinematic_object->position,
                   kinematic_object->rotation * PI_RADIANS,
                   targets);
      break;
    case AttackAlgorithms::Ranged:
      return ranged_attack(kinematic_object->position,
                           kinematic_object->rotation * PI_RADIANS);
  }

  // Return an empty result as no ranged attack was performed
  return {};
}

void AttackSystem::area_of_effect_attack(Registry &registry,
                                         const Vec2d &current_position,
                                         const std::vector<int> &targets) {
  // Find all targets that are within range and attack them
  for (auto target : targets) {
    if (current_position.distance_to(registry.get_kinematic_object(target)->position) <= ATTACK_RANGE) {
      GameObjectAttributeSystem::deal_damage(registry, target, DAMAGE);
    }
  }
}

void AttackSystem::melee_attack(Registry &registry,
                                const Vec2d &current_position,
                                double current_rotation,
                                const std::vector<int> &targets) {
  // Calculate a vector that is perpendicular to the current rotation of the
  // game object
  Vec2d rotation = Vec2d(std::sin(current_rotation), std::cos(current_rotation));

  // Find all targets that can be attacked
  for (auto target : targets) {
    // Calculate the angle between the current rotation of the game object and
    // the direction the target is in
    Vec2d target_position = registry.get_kinematic_object(target)->position;
    double theta = (target_position - current_position).angle_between(rotation);

    // Test if the target is within range and within the circle's sector
    if (current_position.distance_to(target_position) <= ATTACK_RANGE
        && (theta <= MELEE_ATTACK_OFFSET_LOWER || theta >= MELEE_ATTACK_OFFSET_UPPER)) {
      GameObjectAttributeSystem::deal_damage(registry, target, DAMAGE);
    }
  }
}

AttackResult AttackSystem::ranged_attack(const Vec2d &current_position, double current_rotation) {
  return AttackResult{current_position, BULLET_VELOCITY * std::cos(current_rotation),
                      BULLET_VELOCITY * std::sin(current_rotation)};
}
