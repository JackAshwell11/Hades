// Std includes
#include <stdexcept>
#include <numbers>
#include <unordered_set>

// Custom includes
#include "game_objects/steering.hpp"

// ----- CONSTANTS ------------------------------
#define TWO_PI (2 * std::numbers::pi)
const double MAX_SEE_AHEAD = 2 * SPRITE_SIZE;
const int MAX_VELOCITY = 200;
const double OBSTACLE_AVOIDANCE_ANGLE = 60 * PI_RADIANS;
const double PATH_POSITION_RADIUS = 1 * SPRITE_SIZE;
const double SLOWING_RADIUS = 3 * SPRITE_SIZE;
const int WANDER_CIRCLE_DISTANCE = 50;
const int WANDER_CIRCLE_RADIUS = 25;

// ----- STRUCTURES ------------------------------
Vec2d Vec2d::rotated(double angle) const {
  double cos_angle = std::cos(angle);
  double sin_angle = std::sin(angle);
  return {x * cos_angle - y * sin_angle, x * sin_angle + y * cos_angle};
}

double Vec2d::angle_between(const Vec2d &other) const {
  double cross_product = x * other.y - y * other.x;
  double dot_product = x * other.x + y * other.y;
  return std::fmod(std::atan2(cross_product, dot_product) + TWO_PI, TWO_PI);
}

// ----- FUNCTIONS ------------------------------
Vec2d arrive(const Vec2d &current_position, const Vec2d &target_position) {
  // Calculate a vector to the target and its length
  Vec2d direction = target_position - current_position;

  // Check if the game object is inside the slowing area
  if (direction.magnitude() < SLOWING_RADIUS) {
    return (direction * (direction.magnitude() / SLOWING_RADIUS)).normalised();
  }
  return direction.normalised();
}

Vec2d evade(const Vec2d &current_position, const Vec2d &target_position, const Vec2d &target_velocity) {
  // Calculate the future position of the target based on their distance and
  // steer away from it. Higher distances will require more time to reach, so
  // the future position will be further away
  return flee(current_position,
              target_position + target_velocity * (target_position.distance_to(current_position) / MAX_VELOCITY));
}

Vec2d flee(const Vec2d &current_position, const Vec2d &target_position) {
  return (current_position - target_position).normalised();
}

Vec2d follow_path(const Vec2d &current_position, std::vector<Vec2d> &path_list) {
  // Check if the path list is empty
  if (path_list.empty()) {
    throw std::length_error("The path list is empty");
  }

  // Check if the game object has reached the current path position. If so, move it to the end of the vector
  if (current_position.distance_to(path_list[0]) <= PATH_POSITION_RADIUS) {
    path_list.push_back(path_list[0]);
    path_list.erase(path_list.begin());
  }
  return seek(current_position, path_list[0]);
}

Vec2d obstacle_avoidance(const Vec2d &current_position,
                         const Vec2d &current_velocity,
                         const std::unordered_set<Vec2d> &walls) {
  // Create the lambda function to cast a ray from the game object's position
  // in the direction of its velocity at a given angle
  auto raycast = [&current_position, &current_velocity, &walls](double angle = 0) -> Vec2d {
    for (int step = SPRITE_SIZE; step <= MAX_SEE_AHEAD; step += SPRITE_SIZE) {
      Vec2d position = current_position + current_velocity.rotated(angle) * (step / 100.0);
      if (walls.contains(position / SPRITE_SIZE)) {
        return position;
      }
    }
    return {-1, -1};
  };

  // Check if the game object is going to collide with an obstacle
  Vec2d forward_ray = raycast();
  Vec2d left_ray = raycast(OBSTACLE_AVOIDANCE_ANGLE);
  Vec2d right_ray = raycast(-OBSTACLE_AVOIDANCE_ANGLE);

  // Check if there are any obstacles ahead
  if (forward_ray != Vec2d{-1, -1} && left_ray != Vec2d{-1, -1} && right_ray != Vec2d{-1, -1}) {
    // Turn around, there's a wall ahead
    return flee(current_position, forward_ray);
  }
  if (left_ray != Vec2d{-1, -1}) {
    // Turn right, there's a wall left
    return flee(current_position, left_ray);
  }
  if (right_ray != Vec2d{-1, -1}) {
    // Turn left, there's a wall right
    return flee(current_position, right_ray);
  }

  // No obstacles ahead, move forward
  return Vec2d{0, 0};
}

Vec2d pursuit(const Vec2d &current_position, const Vec2d &target_position, const Vec2d &target_velocity) {
  // Calculate the future position of the target based on their distance and
  // steer towards it. Higher distances will require more time to reach, so the
  // future position will be further away
  return seek(current_position,
              target_position + target_velocity * (target_position.distance_to(current_position) / MAX_VELOCITY));
}

Vec2d seek(const Vec2d &current_position, const Vec2d &target_position) {
  return (target_position - current_position).normalised();
}

Vec2d wander(const Vec2d &current_velocity, int displacement_angle) {
  // Calculate the position of an invisible circle in front of the game object
  Vec2d circle_center = current_velocity.normalised() * WANDER_CIRCLE_DISTANCE;

  // Add a displacement force to the centre of the circle to randomise the movement
  Vec2d displacement = (Vec2d(0, -1) * WANDER_CIRCLE_RADIUS).rotated(displacement_angle * PI_RADIANS);
  return (circle_center + displacement).normalised();
}
