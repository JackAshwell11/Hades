// Related header
#include "game_objects/systems/effects.hpp"

// Local headers
#include "game_objects/stats.hpp"

// ----- FUNCTIONS ------------------------------
void EffectSystem::update(const double delta_time) const {
  for (const auto &[game_object_id, component_tuple] : get_registry()->find_components<StatusEffects>()) {
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
        const auto component = std::static_pointer_cast<Stat>(
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

auto EffectSystem::apply_effects(const GameObjectID game_object_id, const GameObjectID target_game_object_id) const
    -> bool {
  // Get the required components
  const auto effect_applier{get_registry()->get_component<EffectApplier>(game_object_id)};
  const auto target_status_effects{get_registry()->get_component<StatusEffects>(target_game_object_id)};

  // Check if the instant effects can be applied
  for (const auto &instant_types : std::views::keys(effect_applier->instant_effects)) {
    if (const auto target_component{
            std::static_pointer_cast<Stat>(get_registry()->get_component(target_game_object_id, instant_types))};
        target_component->get_value() == target_component->get_max_value()) {
      return false;
    }
  }

  // Check if the status effects can be applied
  for (const auto &status_types : std::views::values(effect_applier->status_effects)) {
    if (target_status_effects->applied_effects.contains(status_types.status_effect_type)) {
      return false;
    }
  }

  // Apply the instant effects
  for (const auto &[component_type, increase_function] : effect_applier->instant_effects) {
    const auto target_component{
        std::static_pointer_cast<Stat>(get_registry()->get_component(target_game_object_id, component_type))};
    target_component->set_value(target_component->get_value() + increase_function(1));
  }

  // Apply the status effects
  for (const auto &[component_type, status_effect_data] : effect_applier->status_effects) {
    const auto target_component{
        std::static_pointer_cast<Stat>(get_registry()->get_component(target_game_object_id, component_type))};
    const StatusEffect status_effect{status_effect_data.increase(1), status_effect_data.duration(1),
                                     status_effect_data.interval(1), component_type};
    target_status_effects->applied_effects.emplace(status_effect_data.status_effect_type, status_effect);
    target_component->set_value(target_component->get_value() + status_effect.value);
  }
  return true;
}
