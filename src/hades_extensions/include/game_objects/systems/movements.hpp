// Ensure this file is only included once
#pragma once

// Std headers
#include <deque>

// External headers
#include <chipmunk/chipmunk.h>

// Local headers
#include "game_objects/stats.hpp"

// ----- ENUMS ------------------------------
/// Stores the different types of steering behaviours available.
enum class SteeringBehaviours : std::uint8_t {
  Arrive,
  Evade,
  Flee,
  FollowPath,
  ObstacleAvoidance,
  Pursue,
  Seek,
  Wander,
};

/// Stores the different states the steering movement component can be in.
enum class SteeringMovementState : std::uint8_t {
  Default,
  Footprint,
  Target,
};

// ----- COMPONENTS ------------------------------
/// Allows a game object to determine the time interval between footprints.
struct FootprintInterval final : Stat {
  /// Initialise the object.
  ///
  /// @param value - The initial and maximum value of the footprint interval stat.
  /// @param maximum_level - The maximum level of the footprint interval stat.
  FootprintInterval(const double value, const int maximum_level) : Stat(value, maximum_level) {}
};

/// Allows a game object to determine the maximum number of footprints it can leave.
struct FootprintLimit final : Stat {
  /// Initialise the object.
  ///
  /// @param value - The initial and maximum value of the footprint limit stat.
  /// @param maximum_level - The maximum level of the footprint limit stat.
  FootprintLimit(const double value, const int maximum_level) : Stat(value, maximum_level) {}
};

/// Allows a game object to periodically leave footprints around the game map.
struct Footprints final : ComponentBase {
  /// The footprints the game object has left.
  std::deque<cpVect> footprints;

  /// The time since the game object last left a footprint.
  double time_since_last_footprint{0};
};

/// Allows a game object's movement to be controlled by the keyboard.
struct KeyboardMovement final : ComponentBase {
  /// Whether the game object is moving north or not.
  bool moving_north{false};

  /// Whether the game object is moving east or not.
  bool moving_east{false};

  /// Whether the game object is moving south or not.
  bool moving_south{false};

  /// Whether the game object is moving west or not.
  bool moving_west{false};
};

/// Allows a game object to determine how fast it can move.
struct MovementForce final : Stat {
  /// Initialise the object.
  ///
  /// @param value - The initial and maximum value of the movement force stat.
  /// @param maximum_level - The maximum level of the movement force stat.
  MovementForce(const double value, const int maximum_level) : Stat(value, maximum_level) {}
};

/// Allows a game object to determine how far it can see.
struct ViewDistance final : Stat {
  /// Initialise the object.
  ///
  /// @param value - The initial and maximum value of the view distance stat.
  /// @param maximum_level - The maximum level of the view distance stat.
  ViewDistance(const double value, const int maximum_level) : Stat(value, maximum_level) {}
};

/// Allows a game object's movement to be controlled by steering behaviours.
struct SteeringMovement final : ComponentBase {
  /// The steering behaviours used by the game object.
  std::unordered_map<SteeringMovementState, std::vector<SteeringBehaviours>> behaviours;

  /// The current movement state of the game object.
  SteeringMovementState movement_state{SteeringMovementState::Default};

  /// The game object ID of the target.
  int target_id{-1};

  /// The list of positions the game object should follow.
  std::vector<cpVect> path_list;

  /// Initialise the object.
  ///
  /// @param behaviours - The steering behaviours used by the game object.
  explicit SteeringMovement(
      const std::unordered_map<SteeringMovementState, std::vector<SteeringBehaviours>> &behaviours)
      : behaviours(behaviours) {}
};

// ----- SYSTEMS ------------------------------
/// Provides facilities to manipulate footprint components.
struct FootprintSystem final : SystemBase {
  /// Initialise the object.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  explicit FootprintSystem(Registry *registry) : SystemBase(registry) {}

  /// Process update logic for a footprint component.
  ///
  /// @param delta_time - The time interval since the last time the function was called.
  void update(double delta_time) const override;
};

/// Provides facilities to manipulate keyboard movement components.
struct KeyboardMovementSystem final : SystemBase {
  /// Initialise the object.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  explicit KeyboardMovementSystem(Registry *registry) : SystemBase(registry) {}

  /// Process update logic for a keyboard movement component.
  ///
  /// @param delta_time - The time interval since the last time the function was called.
  void update(double delta_time) const override;
};

/// Provides facilities to manipulate steering movement components.
struct SteeringMovementSystem final : SystemBase {
  /// Initialise the object.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  explicit SteeringMovementSystem(Registry *registry) : SystemBase(registry) {}

  /// Process update logic for a steering movement component.
  ///
  /// @param delta_time - The time interval since the last time the function was called.
  void update(double delta_time) const override;

  /// Update the path lists for the game objects to follow.
  ///
  /// @param target_game_object_id - The ID of the game object to follow.
  /// @param footprints - The list of footprints to follow.
  void update_path_list(GameObjectID target_game_object_id, const std::deque<cpVect> &footprints) const;
};
