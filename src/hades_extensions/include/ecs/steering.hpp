// Ensure this file is only included once
#pragma once

// Std headers
#include <vector>

// External headers
#include <chipmunk/chipmunk.h>

// The scale factor for the sprite.
constexpr double SPRITE_SCALE{0.5};

// The pixel size of the sprite.
constexpr double SPRITE_SIZE{128 * SPRITE_SCALE};

/// The != operator.
inline auto operator!=(const cpVect &lhs, const cpVect &rhs) -> bool { return lhs.x != rhs.x || lhs.y != rhs.y; }

/// The + operator.
inline auto operator+(const cpVect &lhs, const float val) -> cpVect { return {lhs.x + val, lhs.y + val}; }

/// The += operator.
inline auto operator+=(cpVect &lhs, const cpVect &rhs) -> cpVect {
  lhs.x += rhs.x;
  lhs.y += rhs.y;
  return lhs;
}

/// Allow a game object to move towards another game object and stand still.
///
/// @param current_position - The position of the game object.
/// @param target_position - The position of the target game object.
/// @return The new steering force from this behaviour.
auto arrive(const cpVect &current_position, const cpVect &target_position) -> cpVect;

/// Allow a game object to flee from another game object's predicted position.
///
/// @param current_position - The position of the game object.
/// @param target_position - The position of the target game object.
/// @param target_velocity - The velocity of the target game object.
/// @return The new steering force from this behaviour.
auto evade(const cpVect &current_position, const cpVect &target_position, const cpVect &target_velocity) -> cpVect;

/// Allow a game object to run away from another game object.
///
/// @param current_position - The position of the game object.
/// @param target_position - The position of the target game object.
/// @return The new steering force from this behaviour.
auto flee(const cpVect &current_position, const cpVect &target_position) -> cpVect;

/// Allow a game object to follow a pre-determined path.
///
/// @param current_position - The position of the game object.
/// @param path_list - The list of positions the game object should follow.
/// @throws std::length_error - If the path list is empty.
/// @return The new steering force from this behaviour.
auto follow_path(const cpVect &current_position, std::vector<cpVect> &path_list) -> cpVect;

/// Allow a game object to avoid obstacles in its path.
///
/// @param space - The Chipmunk2D space.
/// @param current_position - The position of the game object.
/// @param current_velocity - The velocity of the game object.
/// @return The new steering force from this behaviour.
auto obstacle_avoidance(cpSpace *space, const cpVect &current_position, const cpVect &current_velocity) -> cpVect;

/// Allow a game object to seek towards another game object's predicted position.
///
/// @param current_position - The position of the game object.
/// @param target_position - The position of the target game object.
/// @param target_velocity - The velocity of the target game object.
/// @return The new steering force from this behaviour.
auto pursue(const cpVect &current_position, const cpVect &target_position, const cpVect &target_velocity) -> cpVect;

/// Allow a game object to move towards another game object.
///
/// @param current_position - The position of the game object.
/// @param target_position - The position of the target game object.
/// @return The new steering force from this behaviour.
auto seek(const cpVect &current_position, const cpVect &target_position) -> cpVect;

/// Allow a game object to move in a random direction for a short period of time.
///
/// @param current_velocity - The velocity of the game object.
/// @param displacement_angle - The angle of the displacement force in radians.
/// @return The new steering force from this behaviour.
auto wander(const cpVect &current_velocity, double displacement_angle) -> cpVect;
