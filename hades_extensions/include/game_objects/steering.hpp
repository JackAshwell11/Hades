// Ensure this file is only included once
#pragma once

// Std includes
#include <numbers>
#include <unordered_set>
#include <vector>

// Custom includes
#include "hash_combine.hpp"

// ----- CONSTANTS ------------------------------
#define PI_RADIANS (std::numbers::pi / 180)
#define TWO_PI (2 * std::numbers::pi)
const double SPRITE_SCALE = 0.5;
const double SPRITE_SIZE = 128 * SPRITE_SCALE;

// ----- STRUCTURES ------------------------------
/// Represents a 2D vector.
struct Vec2d {
  inline bool operator==(const Vec2d &vec) const {
    return x == vec.x && y == vec.y;
  }

  inline bool operator!=(const Vec2d &vec) const {
    return x != vec.x || y != vec.y;
  }

  inline Vec2d operator+(const Vec2d &vec) const {
    return {x + vec.x, y + vec.y};
  }

  inline Vec2d operator+=(const Vec2d &vec) {
    x += vec.x;
    y += vec.y;
    return *this;
  }

  inline Vec2d operator-(const Vec2d &vec) const {
    return {x - vec.x, y - vec.y};
  }

  inline Vec2d operator*(const double val) const {
    return {x * val, y * val};
  }

  inline Vec2d operator/(const double val) const {
    return {std::floor(x / val), std::floor(y / val)};
  }

  /// The x value of the vector.
  double x;

  /// The y value of the vector.
  double y;

  /// Initialise the object.
  ///
  /// @param x_val - The x value of the vector.
  /// @param y_val - The y value of the vector.
  Vec2d(double x_val, double y_val) : x(x_val), y(y_val) {}

  /// Get the magnitude of the vector.
  ///
  /// @return The magnitude of the vector.
  [[nodiscard]] inline double magnitude() const {
    return std::hypot(x, y);
  }

  /// Normalise the vector
  ///
  /// @return The normalised vector.
  [[nodiscard]] inline Vec2d normalised() const {
    double magnitude = this->magnitude();
    return (magnitude == 0) ? Vec2d{0, 0} : Vec2d{x / magnitude, y / magnitude};
  }

  /// Rotate the vector by an angle.
  ///
  /// @param angle - The angle to rotate the vector by in radians.
  ///
  /// @return The rotated vector.
  [[nodiscard]] inline Vec2d rotated(double angle) const {
    double cos_angle = std::cos(angle);
    double sin_angle = std::sin(angle);
    return {x * cos_angle - y * sin_angle, x * sin_angle + y * cos_angle};
  }

  /// Get the angle between this vector and another vector.
  ///
  /// @details This will always be between 0 and 2π.
  /// @param other - The other vector to get the angle between.
  /// @return The angle between the two vectors.
  [[nodiscard]] double angle_between(const Vec2d &other) const {
    double cross_product = x * other.y - y * other.x;
    double dot_product = x * other.x + y * other.y;
    return std::fmod(std::atan2(cross_product, dot_product) + TWO_PI, TWO_PI);
  }

  /// Get the distance to another vector.
  ///
  /// @param other - The vector to get the distance to.
  /// @return The distance to the other vector.
  [[nodiscard]] inline double distance_to(const Vec2d &other) const {
    return std::hypot(x - other.x, y - other.y);
  }
};

/// Stores various data about a game object for use in physics-related operations.
struct KinematicObject {
  /// The position of the game object.
  Vec2d position;

  /// The velocity of the game object.
  Vec2d velocity;

  /// The rotation of the game object.
  double rotation;

  /// Initialise the object.
  ///
/// @param position_val - The position of the game object.
/// @param velocity_val - The velocity of the game object.
/// @param rotation_val - The rotation of the game object.
  KinematicObject(const Vec2d &position_val, const Vec2d &velocity_val, double rotation_val = 0)
      : position(position_val), velocity(velocity_val), rotation(rotation_val) {}
};

// ----- HASHES ------------------------------
template<>
struct std::hash<Vec2d> {
  size_t operator()(const Vec2d &vec) const {
    size_t res = 0;
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
Vec2d arrive(const Vec2d &current_position, const Vec2d &target_position);

/// Allow a game object to flee from another game object's predicted position.
///
/// @param current_position - The position of the game object.
/// @param target_position - The position of the target game object.
/// @param target_velocity - The velocity of the target game object.
/// @return The new steering force from this behaviour.
Vec2d evade(const Vec2d &current_position, const Vec2d &target_position, const Vec2d &target_velocity);

/// Allow a game object to run away from another game object.
///
/// @param current_position - The position of the game object.
/// @param current_velocity - The velocity of the game object.
/// @return The new steering force from this behaviour.
Vec2d flee(const Vec2d &current_position, const Vec2d &target_position);

/// Allow a game object to follow a pre-determined path.
///
/// @param current_position - The position of the game object.
/// @param path_list - The list of positions the game object should follow.
/// @throws std::length_error - The path list is empty.
/// @return The new steering force from this behaviour.
Vec2d follow_path(const Vec2d &current_position, std::vector<Vec2d> &path_list);

/// Allow a game object to avoid obstacles in its path.
///
/// @param current_position - The position of the game object.
/// @param current_velocity - The velocity of the game object.
/// @param walls - The set of walls in the game.
/// @return The new steering force from this behaviour.
Vec2d obstacle_avoidance(const Vec2d &current_position,
                         const Vec2d &current_velocity,
                         const std::unordered_set<Vec2d> &walls);

/// Allow a game object to seek towards another game object's predicted position.
///
/// @param current_position - The position of the game object.
/// @param target_position - The position of the target game object.
/// @param target_velocity - The velocity of the target game object.
/// @return The new steering force from this behaviour.
Vec2d pursuit(const Vec2d &current_position, const Vec2d &target_position, const Vec2d &target_velocity);

/// Allow a game object to move towards another game object.
///
/// @param current_position - The position of the game object.
/// @param target_position - The position of the target game object.
/// @return The new steering force from this behaviour.
Vec2d seek(const Vec2d &current_position, const Vec2d &target_position);

/// Allow a game object to move in a random direction for a short period of time.
///
/// @param current_velocity - The velocity of the game object.
/// @param displacement_angle - The angle of the displacement force in degrees.
/// @return The new steering force from this behaviour.
Vec2d wander(const Vec2d &current_velocity, int displacement_angle);
