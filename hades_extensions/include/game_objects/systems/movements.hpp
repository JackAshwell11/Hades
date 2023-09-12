// Ensure this file is only included once
#pragma once

// Custom includes
#include "game_objects/components.hpp"

// ----- STRUCTURES ------------------------------
/// Provides facilities to manipulate keyboard movement components.
struct KeyboardMovementSystem : public SystemBase {
  /// Calculate the new keyboard force to apply to the game object.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  /// @param game_object_id - The ID of the game object to calculate force for.
  /// @return The new force to apply to the game object.
  static Vec2d calculate_keyboard_force(Registry &registry, GameObjectID game_object_id);
};

/// Provides facilities to manipulate steering movement components.
struct SteeringMovementSystem : public SystemBase {
  /// Calculate the new steering force to apply to the game object.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  /// @param game_object_id - The ID of the game object to calculate force for.
  /// @return The new force to apply to the game object.
  static Vec2d calculate_steering_force(Registry &registry, GameObjectID game_object_id);

  /// Update the path lists for the game objects to follow.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  /// @param target_game_object_id - The ID of the game object to follow.
  /// @param footprints - The list of footprints to follow.
  static void update_path_list(Registry &registry, GameObjectID target_game_object_id, std::deque<Vec2d> &footprints);
};

/// Provides facilities to manipulate footprint components.
struct FootprintSystem : public SystemBase {
  /// Process update logic for a footprint component.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  /// @param delta_time - The time interval since the last time the function was called.
  static void update(Registry &registry, double delta_time);
};
