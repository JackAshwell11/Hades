// Related header
#include "ecs/systems/effects.hpp"

// Local headers
#include "ecs/registry.hpp"
#include "ecs/stats.hpp"

namespace {
/// Notifies the registry of a status effect update.
///
/// @param registry - The registry that manages the game objects, components, and systems.
/// @param effects - The status effects that have been applied to the game object.
void notify_status_effect_update(Registry *registry,
                                 const std::unordered_map<StatusEffectType, StatusEffect> &effects) {
  std::unordered_map<StatusEffectType, double> effect_data;
  for (const auto &[effect_type, effect] : effects) {
    effect_data.emplace(effect_type, effect.duration - effect.time_elapsed);
  }
  registry->notify<EventType::StatusEffectUpdate>(effect_data);
}
}  // namespace

auto BaseEffect::apply(const Registry *registry, const GameObjectID game_object_id) const -> bool {
  const auto component{std::static_pointer_cast<Stat>(registry->get_component(game_object_id, target_component))};
  if (!component) {
    return false;
  }
  if (value > 0 && component->get_value() == component->get_max_value()) {
    return false;
  }
  if (value < 0 && component->get_value() == 0) {
    return false;
  }
  component->set_value(component->get_value() + value);
  return true;
}

auto StatusEffect::update(const Registry *registry, const GameObjectID target, const double delta_time) -> bool {
  interval_accumulator += std::clamp(duration - time_elapsed, 0.0, delta_time);
  time_elapsed += delta_time;
  while (interval_accumulator >= interval) {
    (void)apply(registry, target);
    interval_accumulator -= interval;
  }
  return time_elapsed >= duration;
}

void EffectSystem::update(const double delta_time) const {
  for (const auto &[game_object_id, component_tuple] : get_registry()->find_components<StatusEffects>()) {
    // Update and remove expired effects
    auto &active_effects{std::get<0>(component_tuple)->active_effects};
    if (active_effects.empty()) {
      continue;
    }
    for (auto it{active_effects.begin()}; it != active_effects.end();) {
      if (it->second.update(get_registry(), game_object_id, delta_time)) {
        it = active_effects.erase(it);
      } else {
        ++it;
      }
    }
    notify_status_effect_update(get_registry(), active_effects);
  }
}

auto EffectSystem::apply_effects(const GameObjectID source, const GameObjectID target) const -> bool {
  // Get the required components
  const auto effect_applier{get_registry()->get_component<EffectApplier>(source)};
  const auto target_status_effects{get_registry()->get_component<StatusEffects>(target)};

  // Apply instant effects
  bool any_effect_applied{false};
  for (const auto &effect : effect_applier->instant_effects) {
    if (effect.apply(get_registry(), target)) {
      any_effect_applied = true;
    }
  }

  // Apply status effects
  bool status_effects_applied{false};
  for (const auto &[effect_type, effect] : effect_applier->status_effects) {
    // Extend the duration if it is already applied
    if (target_status_effects->active_effects.contains(effect_type)) {
      any_effect_applied = true;
      status_effects_applied = true;
      target_status_effects->active_effects.at(effect_type).duration += effect.duration;
      continue;
    }

    // Otherwise, create a new effect and apply it
    if (auto new_effect{effect}; new_effect.apply(get_registry(), target)) {
      any_effect_applied = true;
      status_effects_applied = true;
      target_status_effects->active_effects.emplace(effect_type, std::move(new_effect));
    }
  }

  // Only notify if any status effects were applied
  if (status_effects_applied) {
    notify_status_effect_update(get_registry(), target_status_effects->active_effects);
  }
  return any_effect_applied;
}
