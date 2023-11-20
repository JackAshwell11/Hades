// Related header
#include "game_objects/systems/attacks.hpp"

// Local headers
#include "game_objects/stats.hpp"

// ----- CONSTANTS -------------------------------
constexpr double ATTACK_RANGE{3 * SPRITE_SIZE};
constexpr int BULLET_VELOCITY{300};
constexpr int DAMAGE{10};
constexpr double MELEE_ATTACK_OFFSET_LOWER{45 * PI_RADIANS};
constexpr double MELEE_ATTACK_OFFSET_UPPER{(2 * (180 * PI_RADIANS)) - MELEE_ATTACK_OFFSET_LOWER};

// ----- FUNCTIONS ------------------------------
/// Performs an area of effect attack around the game object.
///
/// @param registry - The registry that manages the game objects, components, and systems.
/// @param current_position - The current position of the game object.
/// @param targets - The targets to attack.
void area_of_effect_attack(const Registry *registry, const Vec2d &current_position, const std::vector<int> &targets) {
  // Find all targets that are within range and attack them
  for (auto target : targets) {
    if (current_position.distance_to(registry->get_kinematic_object(target)->position) <= ATTACK_RANGE) {
      registry->get_system<DamageSystem>()->deal_damage(target, DAMAGE);
    }
  }
}

/// Performs a melee attack in the direction the game object is facing.
///
/// @param registry - The registry that manages the game objects, components, and systems.
/// @param current_position - The current position of the game object.
/// @param current_rotation - The current rotation of the game object in radians.
/// @param targets - The targets to attack.
void melee_attack(const Registry *registry, const Vec2d &current_position, const double current_rotation,
                  const std::vector<int> &targets) {
  // Calculate a vector that is perpendicular to the current rotation of the game object
  const Vec2d rotation{std::sin(current_rotation), std::cos(current_rotation)};

  // Find all targets that can be attacked
  for (const auto target : targets) {
    // Calculate the angle between the current rotation of the game object and the direction the target is in
    const Vec2d target_position{registry->get_kinematic_object(target)->position};
    const double theta{(target_position - current_position).angle_between(rotation)};

    // Test if the target is within range and within the circle's sector
    if (current_position.distance_to(target_position) <= ATTACK_RANGE &&
        (theta <= MELEE_ATTACK_OFFSET_LOWER || theta >= MELEE_ATTACK_OFFSET_UPPER)) {
      registry->get_system<DamageSystem>()->deal_damage(target, DAMAGE);
    }
  }
}

/// Performs a ranged attack in the direction the game object is facing.
///
/// @param current_position - The current position of the game object.
/// @param current_rotation - The current rotation of the game object in radians.
/// @return The result of the attack.
auto ranged_attack(const Vec2d &current_position, const double current_rotation) -> std::tuple<Vec2d, double, double> {
  return {current_position, BULLET_VELOCITY * std::cos(current_rotation), BULLET_VELOCITY * std::sin(current_rotation)};
}

auto AttackSystem::do_attack(const GameObjectID game_object_id, const std::vector<int> &targets) const
    -> std::optional<std::tuple<Vec2d, double, double>> {
  // Perform the attack on the targets
  auto attacks{get_registry()->get_component<Attacks>(game_object_id)};
  const auto kinematic_object{get_registry()->get_kinematic_object(game_object_id)};
  switch (attacks->attack_algorithms[attacks->attack_state]) {
    case AttackAlgorithm::AreaOfEffect:
      area_of_effect_attack(get_registry(), kinematic_object->position, targets);
      break;
    case AttackAlgorithm::Melee:
      melee_attack(get_registry(), kinematic_object->position, kinematic_object->rotation * PI_RADIANS, targets);
      break;
    case AttackAlgorithm::Ranged:
      return ranged_attack(kinematic_object->position, kinematic_object->rotation * PI_RADIANS);
  }

  // Return an empty result as no ranged attack was performed
  return std::nullopt;
}

void DamageSystem::deal_damage(const GameObjectID game_object_id, const int damage) const {
  // Damage the armour and carry over the extra damage to the health
  auto health{get_registry()->get_component<Health>(game_object_id)};
  auto armour{get_registry()->get_component<Armour>(game_object_id)};
  health->set_value(health->get_value() - std::max(damage - armour->get_value(), 0.0));
  armour->set_value(armour->get_value() - damage);
}
