// Related header
#include "game_objects/systems/attacks.hpp"

// Std headers
#include <numbers>

// External headers
#include <chipmunk/chipmunk_structs.h>

// Local headers
#include "game_objects/stats.hpp"
#include "game_objects/systems/physics.hpp"

// ----- CONSTANTS -------------------------------
#define PI_RADIANS (std::numbers::pi / 180)
constexpr double ATTACK_RANGE{3 * SPRITE_SIZE};
constexpr int BULLET_VELOCITY{300};
constexpr int DAMAGE{10};
constexpr double MELEE_ATTACK_OFFSET_LOWER{-45 * PI_RADIANS};
constexpr double MELEE_ATTACK_OFFSET_UPPER{45 * PI_RADIANS};

// ----- FUNCTIONS ------------------------------
/// Performs an area of effect attack around the game object.
///
/// @param registry - The registry that manages the game objects, components, and systems.
/// @param current_position - The current position of the game object.
/// @param targets - The targets to attack.
void area_of_effect_attack(const Registry *registry, const cpVect &current_position, const std::vector<int> &targets) {
  // Find all targets that are within range and attack them
  for (const auto target : targets) {
    if (cpvdist(current_position, registry->get_component<KinematicComponent>(target)->body->p) <= ATTACK_RANGE) {
      registry->get_system<DamageSystem>()->deal_damage(target, DAMAGE);
    }
  }
}

/// Get the angle between two vectors.
///
/// @details This will always be between 0 and 2π.
/// @param lhs - The first vector to get the angle between.
/// @param rhs - The other vector to get the angle between.
/// @return The angle between the two vectors.
[[nodiscard]] inline auto angle_between(const cpVect &lhs, const cpVect &rhs) -> double {
  return std::atan2(cpvcross(lhs, rhs), cpvdot(lhs, rhs));
}

/// Performs a melee attack in the direction the game object is facing.
///
/// @param registry - The registry that manages the game objects, components, and systems.
/// @param current_position - The current position of the game object.
/// @param current_rotation - The current rotation of the game object in radians.
/// @param targets - The targets to attack.
void melee_attack(const Registry *registry, const cpVect &current_position, const double current_rotation,
                  const std::vector<int> &targets) {
  // Calculate a vector that is perpendicular to the current rotation of the game object
  const cpVect rotation{std::sin(current_rotation), std::cos(current_rotation)};

  // Find all targets that can be attacked
  for (const auto target : targets) {
    // Calculate the angle between the current rotation of the game object and the direction the target is in then test
    // if the target is within range and within the circle's sector
    const cpVect target_position{registry->get_component<KinematicComponent>(target)->body->p};
    if (const double theta{angle_between(target_position - current_position, rotation)};
        cpvdist(current_position, target_position) <= ATTACK_RANGE && theta >= MELEE_ATTACK_OFFSET_LOWER &&
        theta <= MELEE_ATTACK_OFFSET_UPPER) {
      registry->get_system<DamageSystem>()->deal_damage(target, DAMAGE);
    }
  }
}

/// Performs a ranged attack in the direction the game object is facing.
///
/// @param current_position - The current position of the game object.
/// @param current_rotation - The current rotation of the game object in radians.
/// @return The result of the attack.
auto ranged_attack(const cpVect &current_position, const double current_rotation)
    -> std::tuple<cpVect, double, double> {
  return {current_position, BULLET_VELOCITY * std::cos(current_rotation), BULLET_VELOCITY * std::sin(current_rotation)};
}

auto AttackSystem::do_attack(const GameObjectID game_object_id, const std::vector<int> &targets) const
    -> std::optional<std::tuple<cpVect, double, double>> {
  // Perform the attack on the targets
  const auto attacks{get_registry()->get_component<Attacks>(game_object_id)};
  const auto *const body{*get_registry()->get_component<KinematicComponent>(game_object_id)->body};
  switch (attacks->attack_algorithms[attacks->attack_state]) {
    case AttackAlgorithm::AreaOfEffect:
      area_of_effect_attack(get_registry(), body->p, targets);
      break;
    case AttackAlgorithm::Melee:
      melee_attack(get_registry(), body->p, body->a * PI_RADIANS, targets);
      break;
    case AttackAlgorithm::Ranged:
      return ranged_attack(body->p, body->a * PI_RADIANS);
  }

  // Return an empty result as no ranged attack was performed
  return std::nullopt;
}

void DamageSystem::deal_damage(const GameObjectID game_object_id, const int damage) const {
  // Damage the armour and carry over the extra damage to the health
  const auto health{get_registry()->get_component<Health>(game_object_id)};
  const auto armour{get_registry()->get_component<Armour>(game_object_id)};
  health->set_value(health->get_value() - std::max(damage - armour->get_value(), 0.0));
  armour->set_value(armour->get_value() - damage);
}
