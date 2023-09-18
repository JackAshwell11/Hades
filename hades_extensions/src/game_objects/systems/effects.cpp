// Custom includes
#include "game_objects/systems/effects.hpp"

// ----- STRUCTURES ------------------------------
void EffectSystem::update(double delta_time) {
  // Update the status effect time counters and check if they should be removed
  // TODO: Redo how status effects are stored. Maybe vector of tuples or
  //  unordered_map of type_index to vector of status effects or unordered_set
  //  of tuples, not sure
  for (auto &[_, component_tuple] : registry.find_components<StatusEffects>()) {
    for (auto &status_effect : std::get<0>(component_tuple)->status_effects) {}
  }
}

void EffectSystem::apply_instant_effect(GameObjectID game_object_id,
                                        const std::type_index &target_component,
                                        const EffectFunction &effect_function,
                                        int level) {
  // Apply the instant effect
  auto component = std::static_pointer_cast<Stat>(registry.get_component(game_object_id, target_component));
  component->set_value(component->get_value() + effect_function.increase(level));
}

void EffectSystem::apply_status_effect(GameObjectID game_object_id,
                                       const std::type_index &target_component,
                                       const EffectFunction &effect_function,
                                       int level) {
  // Get the component to apply the status effect to as well as the status
  // effect function
  auto component = std::static_pointer_cast<Stat>(registry.get_component(game_object_id, target_component));
  auto status_effects = registry.get_component<StatusEffects>(game_object_id);

  // Apply the status effect
  double value = effect_function.increase(level);
  status_effects->status_effects.emplace(target_component,
                                         StatusEffect{value, effect_function.duration(level), component->get_value(),
                                                      component->max_value});
  component->max_value += value;
  component->set_value(component->get_value() + value);
}
