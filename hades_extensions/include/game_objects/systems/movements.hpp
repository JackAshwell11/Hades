// Ensure this file is only included once
#pragma once

// Std includes
#include <deque>

// Custom includes
#include "game_objects/registry.hpp"

// ----- ENUMS ------------------------------
/// Stores the different types of steering behaviours available.
enum class SteeringBehaviours {
  Arrive,
  Evade,
  Flee,
  FollowPath,
  ObstacleAvoidance,
  Pursuit,
  Seek,
  Wander,
};

/// Stores the different states the steering movement component can be in.
enum class SteeringMovementState {
  Default,
  Footprint,
  Target,
};

// ----- COMPONENTS ------------------------------
/// Allows a game object to periodically leave footprints around the game map.
struct Footprints : public ComponentBase {
  /// The footprints the game object has left.
  std::deque<Vec2d> footprints = {};

  /// The time since the game object last left a footprint.
  double time_since_last_footprint = 0;
};

/// Allows a game object's movement to be controlled by the keyboard.
struct KeyboardMovement : public ComponentBase {
  /// Whether the game object is moving north or not.
  bool moving_north = false;

  /// Whether the game object is moving east or not.
  bool moving_east = false;

  /// Whether the game object is moving south or not.
  bool moving_south = false;

  /// Whether the game object is moving west or not.
  bool moving_west = false;
};

/// Allows a game object's movement to be controlled by steering algorithms.
struct SteeringMovement : public ComponentBase {
  /// The steering behaviours used by the game object.
  std::unordered_map<SteeringMovementState, std::vector<SteeringBehaviours>> behaviours;

  /// The current movement state of the game object.
  SteeringMovementState movement_state = SteeringMovementState::Default;

  /// The game object ID of the target.
  int target_id = -1;

  /// The list of positions the game object should follow.
  std::vector<Vec2d> path_list;

  /// Initialise the component.
  ///
  /// @param behaviours - The steering behaviours used by the game object.
  explicit SteeringMovement(std::unordered_map<SteeringMovementState, std::vector<SteeringBehaviours>> behaviours)
      : behaviours(std::move(behaviours)) {}
};

// ----- SYSTEMS ------------------------------
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
  /// @throws RegistryException - If the game object does not exist or does not have a keyboard movement component.
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
  /// @throws RegistryException - If the game object does not exist or does not have a steering movement component.
  /// @return The new force to apply to the game object.
  Vec2d calculate_steering_force(GameObjectID game_object_id);

  /// Update the path lists for the game objects to follow.
  ///
  /// @param target_game_object_id - The ID of the game object to follow.
  /// @param footprints - The list of footprints to follow.
  void update_path_list(GameObjectID target_game_object_id, const std::deque<Vec2d> &footprints);
};
