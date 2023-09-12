// Std includes
#include <array>

// Custom includes
#include "game_objects/systems/attributes.hpp"

// ----- CONSTANTS -------------------------------
const int ARMOUR_REGEN_AMOUNT = 1;

// ----- STRUCTURES ------------------------------
/// Update a single game object attribute component.
///
/// @param game_object_attribute - The game object attribute component to update.
/// @param delta_time - The time interval since the last time the function was called.
void update_attribute(GameObjectAttributeBase *game_object_attribute, double delta_time) {
  // Update the status effect if one is applied
  if (auto status_effect = game_object_attribute->applied_status_effect) {
    status_effect->time_counter += delta_time;
    if (status_effect->time_counter >= status_effect->duration) {
      game_object_attribute->value(std::min(game_object_attribute->value(), status_effect->original_value));
      game_object_attribute->max_value(status_effect->original_max_value);
      game_object_attribute->applied_status_effect.reset();
    }
  }
}

void ArmourRegenSystem::update(Registry &registry, double delta_time) {
  for (auto &[_, component_tuple] : registry.find_components<Armour, ArmourRegen, ArmourRegenCooldown>()) {
    auto &[armour, armour_regen, armour_regen_cooldown] = component_tuple;
    armour_regen->time_since_armour_regen += delta_time;
    if (armour_regen->time_since_armour_regen >= armour_regen_cooldown->value()) {
      armour->value(armour->value() + ARMOUR_REGEN_AMOUNT);
      armour_regen->time_since_armour_regen = 0;
    }
  }
}

void GameObjectAttributeSystem::update(Registry &registry, double delta_time) {
  for (auto &[_, component_tuple] : registry.find_components<Armour>()) {
    update_attribute(std::get<0>(component_tuple), delta_time);
  }

  for (auto &[_, component_tuple] : registry.find_components<ArmourRegenCooldown>()) {
    update_attribute(std::get<0>(component_tuple), delta_time);
  }

  for (auto &[_, component_tuple] : registry.find_components<FireRatePenalty>()) {
    update_attribute(std::get<0>(component_tuple), delta_time);
  }

  for (auto &[_, component_tuple] : registry.find_components<Health>()) {
    update_attribute(std::get<0>(component_tuple), delta_time);
  }

  for (auto &[_, component_tuple] : registry.find_components<Money>()) {
    update_attribute(std::get<0>(component_tuple), delta_time);
  }

  for (auto &[_, component_tuple] : registry.find_components<MovementForce>()) {
    update_attribute(std::get<0>(component_tuple), delta_time);
  }

  for (auto &[_, component_tuple] : registry.find_components<ViewDistance>()) {
    update_attribute(std::get<0>(component_tuple), delta_time);
  }
}

void GameObjectAttributeSystem::deal_damage(Registry &registry, GameObjectID game_object_id, int damage) {
  // Damage the armour and carry over the extra damage to the health
  auto *health = registry.get_component<Health>(game_object_id);
  auto *armour = registry.get_component<Armour>(game_object_id);
  health->value(health->value() - std::max(damage - armour->value(), 0.0));
  armour->value(armour->value() - damage);
}
