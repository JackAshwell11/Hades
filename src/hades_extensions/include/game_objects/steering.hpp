// Ensure this file is only included once
#pragma once

// Std headers
#include <cmath>
#include <numbers>
#include <unordered_set>

// Local headers
#include "hash_combine.hpp"

// ----- CONSTANTS ------------------------------
#define PI_RADIANS (std::numbers::pi / 180)
#define TWO_PI (2 * std::numbers::pi)
constexpr double SPRITE_SCALE{0.5};
constexpr double SPRITE_SIZE{128 * SPRITE_SCALE};

// ----- STRUCTURES ------------------------------
/// Represents a 2D vector.
struct Vec2d {
  auto operator==(const Vec2d &vec) const -> bool { return x == vec.x && y == vec.y; }

  auto operator!=(const Vec2d &vec) const -> bool { return x != vec.x || y != vec.y; }

  auto operator+(const Vec2d &vec) const -> Vec2d { return {x + vec.x, y + vec.y}; }

  auto operator+=(const Vec2d &vec) -> Vec2d {
    x += vec.x;
    y += vec.y;
    return *this;
  }

  auto operator-(const Vec2d &vec) const -> Vec2d { return {x - vec.x, y - vec.y}; }

  auto operator*(const double val) const -> Vec2d { return {x * val, y * val}; }

  auto operator/(const double val) const -> Vec2d { return {std::floor(x / val), std::floor(y / val)}; }

  /// The x value of the vector.
  double x;

  /// The y value of the vector.
  double y;

  /// Get the magnitude of the vector.
  ///
  /// @return The magnitude of the vector.
  [[nodiscard]] auto magnitude() const -> double { return std::hypot(x, y); }

  /// Normalise the vector
  ///
  /// @return The normalised vector.
  [[nodiscard]] auto normalised() const -> Vec2d {
    const double magnitude_val{this->magnitude()};
    return (magnitude_val == 0) ? Vec2d{0, 0} : Vec2d{x / magnitude_val, y / magnitude_val};
  }

  /// Rotate the vector by an angle.
  ///
  /// @param angle - The angle to rotate the vector by in radians.
  ///
  /// @return The rotated vector.
  [[nodiscard]] auto rotated(const double angle) const -> Vec2d {
    const double cos_angle{std::cos(angle)};
    const double sin_angle{std::sin(angle)};
    return {x * cos_angle - y * sin_angle, x * sin_angle + y * cos_angle};
  }

  /// Get the angle between this vector and another vector.
  ///
  /// @details This will always be between 0 and 2π.
  /// @param other - The other vector to get the angle between.
  /// @return The angle between the two vectors.
  [[nodiscard]] auto angle_between(const Vec2d &other) const -> double {
    const double cross_product{x * other.y - y * other.x};
    const double dot_product{x * other.x + y * other.y};
    return std::fmod(std::atan2(cross_product, dot_product) + TWO_PI, TWO_PI);
  }

  /// Get the distance to another vector.
  ///
  /// @param other - The vector to get the distance to.
  /// @return The distance to the other vector.
  [[nodiscard]] auto distance_to(const Vec2d &other) const -> double { return std::hypot(x - other.x, y - other.y); }
};

/// Stores various data about a game object for use in physics-related operations.
struct KinematicObject {
  /// The position of the game object.
  Vec2d position{0, 0};

  /// The velocity of the game object.
  Vec2d velocity{0, 0};

  /// The rotation of the game object.
  double rotation{0};
};

// ----- HASHES ------------------------------
template <>
struct std::hash<Vec2d> {
  auto operator()(const Vec2d &vec) const noexcept -> size_t {
    size_t res{0};
    hash_combine(res, vec.x);
    hash_combine(res, vec.y);
    return res;
  }
};

// ----- FUNCTIONS ------------------------------
/// Allow a game object to move towards another game object and stand still.
///
/// @param current_position - The position of the game object.
/// @param target_position - The position of the target game object.
/// @return The new steering force from this behaviour.
auto arrive(const Vec2d &current_position, const Vec2d &target_position) -> Vec2d;

/// Allow a game object to flee from another game object's predicted position.
///
/// @param current_position - The position of the game object.
/// @param target_position - The position of the target game object.
/// @param target_velocity - The velocity of the target game object.
/// @return The new steering force from this behaviour.
auto evade(const Vec2d &current_position, const Vec2d &target_position, const Vec2d &target_velocity) -> Vec2d;

/// Allow a game object to run away from another game object.
///
/// @param current_position - The position of the game object.
/// @param target_position - The position of the target game object.
/// @return The new steering force from this behaviour.
auto flee(const Vec2d &current_position, const Vec2d &target_position) -> Vec2d;

/// Allow a game object to follow a pre-determined path.
///
/// @param current_position - The position of the game object.
/// @param path_list - The list of positions the game object should follow.
/// @throws std::length_error - If the path list is empty.
/// @return The new steering force from this behaviour.
auto follow_path(const Vec2d &current_position, std::vector<Vec2d> &path_list) -> Vec2d;

/// Allow a game object to avoid obstacles in its path.
///
/// @param current_position - The position of the game object.
/// @param current_velocity - The velocity of the game object.
/// @param walls - The set of walls in the game.
/// @return The new steering force from this behaviour.
auto obstacle_avoidance(const Vec2d &current_position, const Vec2d &current_velocity,
                        const std::unordered_set<Vec2d> &walls) -> Vec2d;

/// Allow a game object to seek towards another game object's predicted position.
///
/// @param current_position - The position of the game object.
/// @param target_position - The position of the target game object.
/// @param target_velocity - The velocity of the target game object.
/// @return The new steering force from this behaviour.
auto pursue(const Vec2d &current_position, const Vec2d &target_position, const Vec2d &target_velocity) -> Vec2d;

/// Allow a game object to move towards another game object.
///
/// @param current_position - The position of the game object.
/// @param target_position - The position of the target game object.
/// @return The new steering force from this behaviour.
auto seek(const Vec2d &current_position, const Vec2d &target_position) -> Vec2d;

/// Allow a game object to move in a random direction for a short period of time.
///
/// @param current_velocity - The velocity of the game object.
/// @param displacement_angle - The angle of the displacement force in degrees.
/// @return The new steering force from this behaviour.
auto wander(const Vec2d &current_velocity, int displacement_angle) -> Vec2d;
