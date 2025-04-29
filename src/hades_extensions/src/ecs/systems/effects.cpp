// Related header
#include "ecs/systems/effects.hpp"

// Local headers
#include "ecs/registry.hpp"
#include "ecs/stats.hpp"

auto BaseEffect::apply(Registry *registry, const GameObjectID game_object_id) const -> bool {
  const auto component{std::static_pointer_cast<Stat>(registry->get_component(game_object_id, target_component))};
  if (!component || component->get_value() == component->get_max_value()) {
    return false;
  }
  component->set_value(component->get_value() + value);
  return true;
}

auto StatusEffect::update(Registry *registry, const GameObjectID target, const double delta_time) -> bool {
  interval_accumulator += std::clamp(duration - time_elapsed, 0.0, delta_time);
  time_elapsed += delta_time;
  while (interval_accumulator >= interval) {
    apply(registry, target);
    interval_accumulator -= interval;
  }
  return time_elapsed >= duration;
}

void EffectSystem::update(const double delta_time) const {
  for (const auto &[game_object_id, component_tuple] : get_registry()->find_components<StatusEffects>()) {
    // Update and remove expired effects
    auto &active_effects{std::get<0>(component_tuple)->active_effects};
    std::erase_if(active_effects, [this, game_object_id, delta_time](auto &effect) {
      return effect->update(get_registry(), game_object_id, delta_time);
    });
  }
}

auto EffectSystem::apply_effects(const GameObjectID source, const GameObjectID target) const -> bool {
  // Get the required components
  const auto effect_applier{get_registry()->get_component<EffectApplier>(source)};
  const auto target_status_effects{get_registry()->get_component<StatusEffects>(target)};

  // Apply instant effects
  for (const auto &effect : effect_applier->instant_effects) {
    if (!effect->apply(get_registry(), target)) {
      return false;
    }
  }

  // Apply status effects
  for (const auto &effect : effect_applier->status_effects) {
    const auto new_effect{std::make_shared<StatusEffect>(*effect)};
    if (!new_effect->apply(get_registry(), target)) {
      return false;
    }
    target_status_effects->active_effects.push_back(new_effect);
  }
  return true;
}
