// Ensure this file is only included once
#pragma once

// Custom includes
#include "game_objects/components.hpp"

// ----- STRUCTURES ------------------------------
/// Provides facilities to manipulate armour regen components.
struct ArmourRegenSystem : public SystemBase {
  /// Process update logic for an armour regeneration component.
  ///
  /// @param delta_time - The time interval since the last time the function was called.
  void update(Registry &registry, double delta_time) final;
};

/// Provides facilities to manipulate game object attributes.
struct GameObjectAttributeSystem : public SystemBase {
  /// Process update logic for a game object attribute.
  ///
  /// @param delta_time - The time interval since the last time the function was called.
  void update(Registry &registry, double delta_time) final;

  /// Upgrade the game object attribute to the next level if possible.
  ///
  /// @tparam T - The type of game object attribute to upgrade.
  /// @param registry - The registry to upgrade the game object attribute in.
  /// @param game_object_id - The ID of the game object to upgrade the game object attribute for.
  /// @param increase - The lambda function which calculates the next level's value based on the current level.
  template<typename T>
  static bool upgrade(Registry &registry, GameObjectID game_object_id, const std::function<double(int)> &increase) {
    // Check if the attribute can be upgraded or if it is already at max level
    auto *game_object_attribute = registry.get_component<T>(game_object_id);
    if (!game_object_attribute->is_upgradable()
        || game_object_attribute->current_level() >= game_object_attribute->level_limit()) {
      return false;
    }

    // Upgrade the attribute to the next level
    auto diff = increase(game_object_attribute->current_level() + 1) - increase(game_object_attribute->current_level());
    game_object_attribute->max_value(game_object_attribute->max_value() + diff);
    game_object_attribute->current_level(game_object_attribute->current_level() + 1);
    game_object_attribute->value(game_object_attribute->value() + diff);
    return true;
  }

  /// Apply an instant effect to the game object attribute if possible.
  ///
  /// @param game_object_id - The game object ID to upgrade the attribute of.
  /// @param increase - The lambda function which calculate the value of the instant effect.
  /// @param level - The level to initialise the instant effect at.
  template<typename T>
  static bool apply_instant_effect(Registry &registry,
                                   GameObjectID game_object_id,
                                   const std::function<double(int)> &increase,
                                   int level) {
    // Check if the attribute can have an instant effect or if it is already at
    // the max
    auto *game_object_attribute = registry.get_component<T>(game_object_id);
    if (!game_object_attribute->has_instant_effect()
        || game_object_attribute->value() == game_object_attribute->max_value()) {
      return false;
    }

    // Add the instant effect to the attribute
    game_object_attribute->value(game_object_attribute->value() + increase(level));
    return true;
  }

  /// Apply a status effect to the game object attribute if possible.
  ///
  /// @param game_object_id - The game object ID to upgrade the attribute of.
  /// @param status_effect - The lambda functions which calculate the value and duration values for the status effect.
  /// @param level - The level to initialise the status effect at.
  template<typename T>
  static bool apply_status_effect(Registry &registry,
                                  GameObjectID game_object_id,
                                  const std::tuple<std::function<double(int)>,
                                                   std::function<double(int)>> &status_effect,
                                  int level) {
    // Check if the attribute can have a status effect or if it already has one
    // TODO: See if all methods can use &game_object_attribute to avoid pointers

    // TODO: We may have a ruff problem for the binding for this method. Use
    //  https://chat.openai.com/c/4395342c-ea28-47eb-ab8b-7c922ed1cf17 to maybe
    //  sort it
    auto *game_object_attribute = registry.get_component<T>(game_object_id);
    if (!game_object_attribute->has_status_effect() || game_object_attribute->applied_status_effect.has_value()) {
      return false;
    }

    // Apply the status effect to this attribute
    double value = std::get<0>(status_effect)(level);
    game_object_attribute->applied_status_effect = StatusEffect{
        value,
        std::get<1>(status_effect)(level),
        game_object_attribute->value(),
        game_object_attribute->max_value(),
    };
    game_object_attribute->max_value(game_object_attribute->max_value() + value);
    game_object_attribute->value(game_object_attribute->value() + value);
    return true;
  }

  /// Deal damage to a game object.
  ///
  /// @param game_object_id - The game object ID to deal damage to.
  /// @param damage - The amount of damage to deal to the game object.
  static void deal_damage(Registry &registry, GameObjectID game_object_id, int damage);
};
