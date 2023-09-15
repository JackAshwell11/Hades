// Ensure this file is only included once
#pragma once

// Custom includes
#include "game_objects/components.hpp"

// ----- STRUCTURES ------------------------------
/// Provides facilities to manipulate footprint components.
struct FootprintSystem : public SystemBase {
  /// Initialise the system.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  explicit FootprintSystem(Registry &registry) : SystemBase(registry) {}

  /// Process update logic for a footprint component.
  ///
  /// @param delta_time - The time interval since the last time the function was called.
  void update(double delta_time) final;
};

/// Provides facilities to manipulate keyboard movement components.
struct KeyboardMovementSystem : public SystemBase {
  /// Initialise the system.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  explicit KeyboardMovementSystem(Registry &registry) : SystemBase(registry) {}

  /// Calculate the new keyboard force to apply to the game object.
  ///
  /// @param game_object_id - The ID of the game object to calculate force for.
  /// @return The new force to apply to the game object.
  Vec2d calculate_keyboard_force(GameObjectID game_object_id);
};

/// Provides facilities to manipulate steering movement components.
struct SteeringMovementSystem : public SystemBase {
  /// Initialise the system.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  explicit SteeringMovementSystem(Registry &registry) : SystemBase(registry) {}

  /// Calculate the new steering force to apply to the game object.
  ///
  /// @param game_object_id - The ID of the game object to calculate force for.
  /// @return The new force to apply to the game object.
  Vec2d calculate_steering_force(GameObjectID game_object_id);

  /// Update the path lists for the game objects to follow.
  ///
  /// @param target_game_object_id - The ID of the game object to follow.
  /// @param footprints - The list of footprints to follow.
  void update_path_list(GameObjectID target_game_object_id, const std::deque<Vec2d> &footprints);
};
