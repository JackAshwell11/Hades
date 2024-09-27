// Ensure this file is only included once
#pragma once

// Local headers
#include "ecs/registry.hpp"

// ----- CLASSES ------------------------------
/// Manages the registry and the event handling.
class GameEngine {
 public:
  /// Initialise the object.
  GameEngine() = default;

  /// Get the registry.
  ///
  /// @return The registry.
  auto get_registry() -> Registry& { return registry_; }

  /// Set the player game object ID.
  ///
  /// @param player_id - The player game object ID.
  void set_player_id(const GameObjectID player_id) { player_id_ = player_id; }

  /// Handle a key press event.
  ///
  /// @param symbol - The key symbol.
  /// @param modifiers - The key modifiers.
  void on_key_press(int symbol, int modifiers) const;

  /// Handle a key release event.
  ///
  /// @param symbol - The key symbol.
  /// @param modifiers - The key modifiers.
  void on_key_release(int symbol, int modifiers) const;

 private:
  /// The registry that manages game objects, components, and systems.
  Registry registry_;

  /// The player game object ID.
  GameObjectID player_id_{};
};
