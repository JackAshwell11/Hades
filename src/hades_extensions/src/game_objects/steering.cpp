// Related header
#include "game_objects/steering.hpp"

// Std headers
#include <stdexcept>

// ----- CONSTANTS ------------------------------
constexpr double MAX_SEE_AHEAD{2 * SPRITE_SIZE};
constexpr int MAX_VELOCITY{200};
constexpr double OBSTACLE_AVOIDANCE_ANGLE{60 * PI_RADIANS};
constexpr double PATH_POSITION_RADIUS{1 * SPRITE_SIZE};
constexpr double SLOWING_RADIUS{3 * SPRITE_SIZE};
constexpr int WANDER_CIRCLE_DISTANCE{50};
constexpr int WANDER_CIRCLE_RADIUS{25};

// ----- FUNCTIONS ------------------------------
auto arrive(const Vec2d &current_position, const Vec2d &target_position) -> Vec2d {
  // Calculate a vector to the target and its length
  const Vec2d direction{target_position - current_position};

  // Check if the game object is inside the slowing area
  if (direction.magnitude() < SLOWING_RADIUS) {
    return (direction * (direction.magnitude() / SLOWING_RADIUS)).normalised();
  }
  return direction.normalised();
}

auto evade(const Vec2d &current_position, const Vec2d &target_position, const Vec2d &target_velocity) -> Vec2d {
  // Calculate the future position of the target based on their distance, and steer away from it.
  // Higher distances will require more time to reach, so the future position will be further away
  return flee(current_position,
              target_position + target_velocity * (target_position.distance_to(current_position) / MAX_VELOCITY));
}

auto flee(const Vec2d &current_position, const Vec2d &target_position) -> Vec2d {
  return (current_position - target_position).normalised();
}

auto follow_path(const Vec2d &current_position, std::vector<Vec2d> &path_list) -> Vec2d {
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

auto obstacle_avoidance(const Vec2d &current_position, const Vec2d &current_velocity,
                        const std::unordered_set<Vec2d> &walls) -> Vec2d {
  // Create the lambda function to cast a ray from the game object's position in the direction of its velocity at a
  // given angle
  auto raycast{[&current_position, &current_velocity, &walls](const double angle = 0) -> Vec2d {
    // Pre-calculate some values used during the raycast
    const auto rotated_velocity{current_velocity.rotated(angle)};
    constexpr int step_count{static_cast<int>(MAX_SEE_AHEAD / SPRITE_SIZE)};

    // Perform the raycast
    for (int step = 1; step <= step_count; step++) {
      if (Vec2d position{current_position + rotated_velocity * (step * SPRITE_SIZE / 100.0)};
          walls.contains(position / SPRITE_SIZE)) {
        return position;
      }
    }
    return {-1, -1};
  }};

  // Check if the game object is going to collide with an obstacle
  const Vec2d forward_ray{raycast()};
  const Vec2d left_ray{raycast(OBSTACLE_AVOIDANCE_ANGLE)};
  const Vec2d right_ray{raycast(-OBSTACLE_AVOIDANCE_ANGLE)};

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

auto pursue(const Vec2d &current_position, const Vec2d &target_position, const Vec2d &target_velocity) -> Vec2d {
  // Calculate the future position of the target based on their distance, and steer away from it.
  // Higher distances will require more time to reach, so the future position will be further away
  return seek(current_position,
              target_position + target_velocity * (target_position.distance_to(current_position) / MAX_VELOCITY));
}

auto seek(const Vec2d &current_position, const Vec2d &target_position) -> Vec2d {
  return (target_position - current_position).normalised();
}

auto wander(const Vec2d &current_velocity, const int displacement_angle) -> Vec2d {
  // Calculate the position of an invisible circle in front of the game object
  const Vec2d circle_center{current_velocity.normalised() * WANDER_CIRCLE_DISTANCE};

  // Add a displacement force to the centre of the circle to randomise the movement
  const Vec2d displacement{(Vec2d{0, -1} * WANDER_CIRCLE_RADIUS).rotated(displacement_angle * PI_RADIANS)};
  return (circle_center + displacement).normalised();
}
