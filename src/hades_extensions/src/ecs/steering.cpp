// Related header
#include "ecs/steering.hpp"

// Std headers
#include <cmath>
#include <numbers>
#include <stdexcept>

// Local headers
#include "ecs/types.hpp"

// The value of PI in radians.
#define PI_RADIANS (std::numbers::pi / 180)

namespace {
// The maximum distance the game object can see ahead.
constexpr double MAX_SEE_AHEAD{2 * SPRITE_SIZE};

// The maximum velocity of the game object.
constexpr int MAX_VELOCITY{200};

// The angle in which the game object can avoid obstacles.
constexpr double OBSTACLE_AVOIDANCE_ANGLE{60 * PI_RADIANS};

// The radius of the path position.
constexpr double PATH_POSITION_RADIUS{1 * SPRITE_SIZE};

// The distance of when the game object should start slowing down.
constexpr double SLOWING_RADIUS{3 * SPRITE_SIZE};

// The distance of the wander circle from the game object.
constexpr int WANDER_CIRCLE_DISTANCE{50};

// The radius of the wander circle.
constexpr int WANDER_CIRCLE_RADIUS{25};
}  // namespace

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

auto obstacle_avoidance(cpSpace *space, const cpVect &current_position, const cpVect &current_velocity) -> cpVect {
  // Create the lambda function to cast a ray from the game object's position in the direction of its velocity at a
  // given angle
  auto raycast{[&space, &current_position, &current_velocity](const double angle = 0) -> cpVect {
    // Calculate the end position of the ray
    const auto rotated_velocity{cpvrotate(cpvnormalize(current_velocity), cpvforangle(angle))};
    const auto end_position{current_position + rotated_velocity * MAX_SEE_AHEAD};

    // Perform the raycast
    cpSegmentQueryInfo info;
    if (cpSpaceSegmentQueryFirst(space, current_position, end_position, SPRITE_SIZE / 4,
                                 {CP_NO_GROUP, CP_ALL_CATEGORIES, static_cast<cpBitmask>(GameObjectType::Wall)},
                                 &info) != nullptr) {
      return info.point;
    }
    return cpvzero;
  }};

  // Check if there are any obstacles ahead. If so, accumulate the avoidance force
  cpVect avoidance_force{};
  for (const auto &ray : {raycast(), raycast(OBSTACLE_AVOIDANCE_ANGLE), raycast(-OBSTACLE_AVOIDANCE_ANGLE)}) {
    if (ray != cpvzero) {
      avoidance_force += flee(current_position, ray);
    }
  }
  return avoidance_force;
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

auto wander(const cpVect &current_velocity, const double displacement_angle) -> cpVect {
  // Calculate the position of an invisible circle in front of the game object
  const cpVect circle_center{cpvnormalize(current_velocity) * WANDER_CIRCLE_DISTANCE};

  // Add a displacement force to the centre of the circle to randomise the movement
  const cpVect displacement{cpvrotate(cpv(0, -1) * WANDER_CIRCLE_RADIUS, cpvforangle(displacement_angle))};
  return cpvnormalize(circle_center + displacement);
}
