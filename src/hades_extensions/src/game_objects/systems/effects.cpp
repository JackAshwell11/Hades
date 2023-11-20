// Related header
#include "game_objects/systems/effects.hpp"

// Local headers
#include "game_objects/stats.hpp"

// ----- FUNCTIONS ------------------------------
void EffectSystem::update(const double delta_time) const {
  for (auto &[game_object_id, component_tuple] : get_registry()->find_components<StatusEffects>()) {
    // Create a vector to store the expired status effects
    auto &applied_effects{std::get<0>(component_tuple)->applied_effects};
    std::vector<StatusEffectType> expired_status_effects;

    // Update the status effects and keep track of the expired ones.
    // Note that in reality, delta_time will be ~0.016 (60 FPS), so big jumps in time where multiple intervals are
    // covered within a single update should never happen.
    // But if they do, this will accumulate the leftover time and the status effect is applied in subsequent updates
    for (auto &[status_effect_type, status_effect] : std::get<0>(component_tuple)->applied_effects) {
      status_effect.time_counter += delta_time;
      status_effect.leftover_time += delta_time;
      if (status_effect.time_counter >= status_effect.duration) {
        expired_status_effects.push_back(status_effect_type);
      } else if (status_effect.leftover_time >= status_effect.interval) {
        auto component = std::static_pointer_cast<Stat>(
            get_registry()->get_component(game_object_id, status_effect.target_component));
        component->set_value(component->get_value() + status_effect.value);
        status_effect.leftover_time -= status_effect.interval;
      }
    }

    // Remove the expired status effects
    for (auto &status_effect_type : expired_status_effects) {
      applied_effects.erase(status_effect_type);
    }
  }
}

auto EffectSystem::apply_instant_effect(const GameObjectID game_object_id, const std::type_index &target_component,
                                        const ActionFunction &increase_function, const int level) -> bool {
  // Check if the component is already at the maximum
  auto component{std::static_pointer_cast<Stat>(get_registry()->get_component(game_object_id, target_component))};
  if (component->get_value() == component->get_max_value()) {
    return false;
  }

  // Apply the instant effect
  component->set_value(component->get_value() + increase_function(level));
  return true;
}

auto EffectSystem::apply_status_effect(const GameObjectID game_object_id, const std::type_index &target_component,
                                       const StatusEffectData &status_effect_data, const int level) -> bool {
  // Check if the status effect has already been applied
  auto status_effects{get_registry()->get_component<StatusEffects>(game_object_id)};
  if (status_effects->applied_effects.contains(status_effect_data.status_effect_type)) {
    return false;
  }

  // Apply the status effect
  const StatusEffect status_effect{status_effect_data.increase(level), status_effect_data.duration(level),
                                   status_effect_data.interval(level), target_component};
  status_effects->applied_effects.emplace(status_effect_data.status_effect_type, status_effect);
  auto component{std::static_pointer_cast<Stat>(get_registry()->get_component(game_object_id, target_component))};
  component->set_value(component->get_value() + status_effect.value);
  return true;
}
