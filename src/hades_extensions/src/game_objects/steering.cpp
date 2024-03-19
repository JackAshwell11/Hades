// Related header
#include "game_objects/steering.hpp"

// Std headers
#include <numbers>
#include <stdexcept>

// ----- CONSTANTS ------------------------------
#define PI_RADIANS (std::numbers::pi / 180)
#define TWO_PI (2 * std::numbers::pi)
constexpr double MAX_SEE_AHEAD{2 * SPRITE_SIZE};
constexpr int MAX_VELOCITY{200};
constexpr double OBSTACLE_AVOIDANCE_ANGLE{60 * PI_RADIANS};
constexpr double PATH_POSITION_RADIUS{1 * SPRITE_SIZE};
constexpr double SLOWING_RADIUS{3 * SPRITE_SIZE};
constexpr int WANDER_CIRCLE_DISTANCE{50};
constexpr int WANDER_CIRCLE_RADIUS{25};

// ----- OPERATORS ------------------------------
inline auto operator/(const cpVect &lhs, const double val) -> cpVect {
  return {std::floor(lhs.x / val), std::floor(lhs.y / val)};
}

// ----- FUNCTIONS ------------------------------
auto arrive(const cpVect &current_position, const cpVect &target_position) -> cpVect {
  // Calculate a vector to the target and its length
  const cpVect direction{target_position - current_position};

  // Check if the game object is inside the slowing area
  if (cpvlength(direction) < SLOWING_RADIUS) {
    return cpvnormalize(direction * (cpvlength(direction) / SLOWING_RADIUS));
  }
  return cpvnormalize(direction);
}

auto evade(const cpVect &current_position, const cpVect &target_position, const cpVect &target_velocity) -> cpVect {
  // Calculate the future position of the target based on their distance, and steer away from it.
  // Higher distances will require more time to reach, so the future position will be further away
  return flee(current_position,
              target_position + target_velocity * (cpvdist(target_position, current_position) / MAX_VELOCITY));
}

auto flee(const cpVect &current_position, const cpVect &target_position) -> cpVect {
  return cpvnormalize(current_position - target_position);
}

auto follow_path(const cpVect &current_position, std::vector<cpVect> &path_list) -> cpVect {
  // Check if the path list is empty
  if (path_list.empty()) {
    throw std::length_error("The path list is empty.");
  }

  // Check if the game object has reached the current path position. If so, move it to the end of the vector
  if (cpvdist(current_position, path_list[0]) <= PATH_POSITION_RADIUS) {
    path_list.push_back(path_list[0]);
    path_list.erase(path_list.begin());
  }
  return seek(current_position, path_list[0]);
}

auto obstacle_avoidance(const cpVect &current_position, const cpVect &current_velocity, const std::unordered_set<cpVect> &walls) -> cpVect {
  // Create the lambda function to cast a ray from the game object's position in the direction of its velocity at a
  // given angle
  auto raycast{[&current_position, &current_velocity, &walls](const double angle = 0) -> cpVect {
    // Pre-calculate some values used during the raycast
    const auto rotated_velocity{cpvrotate(current_velocity, cpvforangle(angle))};
    constexpr int step_count{static_cast<int>(MAX_SEE_AHEAD / SPRITE_SIZE)};

    // Perform the raycast
    for (int step{1}; step <= step_count; step++) {
      if (cpVect position{current_position + rotated_velocity * (step * SPRITE_SIZE / 100.0)};
          walls.contains(position / SPRITE_SIZE)) {
        return position;
      }
    }
    return {-1, -1};
  }};

  // Check if the game object is going to collide with an obstacle
  const cpVect forward_ray{raycast()};
  const cpVect left_ray{raycast(OBSTACLE_AVOIDANCE_ANGLE)};
  const cpVect right_ray{raycast(-OBSTACLE_AVOIDANCE_ANGLE)};

  // Check if there are any obstacles ahead
  if (forward_ray != cpVect{-1, -1} && left_ray != cpVect{-1, -1} && right_ray != cpVect{-1, -1}) {
    // Turn around, there's a wall ahead
    return flee(current_position, forward_ray);
  }
  if (left_ray != cpVect{-1, -1}) {
    // Turn right, there's a wall left
    return flee(current_position, left_ray);
  }
  if (right_ray != cpVect{-1, -1}) {
    // Turn left, there's a wall right
    return flee(current_position, right_ray);
  }

  // No obstacles ahead, move forward
  return {0, 0};
}

auto pursue(const cpVect &current_position, const cpVect &target_position, const cpVect &target_velocity) -> cpVect {
  // Calculate the future position of the target based on their distance, and steer away from it.
  // Higher distances will require more time to reach, so the future position will be further away
  return seek(current_position,
              target_position + target_velocity * (cpvdist(target_position, current_position) / MAX_VELOCITY));
}

auto seek(const cpVect &current_position, const cpVect &target_position) -> cpVect {
  return cpvnormalize(target_position - current_position);
}

auto wander(const cpVect &current_velocity, const int displacement_angle) -> cpVect {
  // Calculate the position of an invisible circle in front of the game object
  const cpVect circle_center{cpvnormalize(current_velocity) * WANDER_CIRCLE_DISTANCE};

  // Add a displacement force to the centre of the circle to randomise the movement
  const cpVect displacement{cpvrotate(cpVect{0, -1} * WANDER_CIRCLE_RADIUS, cpvforangle(displacement_angle * PI_RADIANS))};
  return cpvnormalize(circle_center + displacement);
}
