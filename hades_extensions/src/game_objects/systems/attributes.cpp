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
void update_attribute(const std::shared_ptr<GameObjectAttributeBase> &game_object_attribute, double delta_time) {
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

void GameObjectAttributeSystem::update(double delta_time) {
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
