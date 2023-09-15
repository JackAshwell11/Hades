// Std includes
#include <random>

// Custom includes
#include "game_objects/systems/movements.hpp"

// ----- CONSTANTS ------------------------------
const double FOOTPRINT_INTERVAL = 0.5;
const int FOOTPRINT_LIMIT = 10;
const double TARGET_DISTANCE = 3 * SPRITE_SIZE;

// ----- STRUCTURES ------------------------------
void FootprintSystem::update(double delta_time) {
  // Update the time since the last footprint then check if a new footprint
  // should be created
  for (auto &[game_object_id, component_tuple] : registry.find_components<Footprints>()) {
    auto footprints = std::get<0>(component_tuple);
    footprints->time_since_last_footprint += delta_time;
    if (footprints->time_since_last_footprint < FOOTPRINT_INTERVAL) {
      return;
    }

    // Reset the counter and create a new footprint making sure to only keep
    // FOOTPRINT_LIMIT footprints
    const Vec2d current_position = registry.get_kinematic_object(game_object_id)->position;
    footprints->time_since_last_footprint = 0;
    if (footprints->footprints.size() >= FOOTPRINT_LIMIT) {
      footprints->footprints.pop_front();
    }
    footprints->footprints.push_back(current_position);

    // Update the path list for all SteeringMovement components
    registry.find_system<SteeringMovementSystem>()->update_path_list(game_object_id, footprints->footprints);
  }
}

Vec2d KeyboardMovementSystem::calculate_keyboard_force(GameObjectID game_object_id) {
  auto keyboard_movement = registry.get_component<KeyboardMovement>(game_object_id);
  return Vec2d{static_cast<double>(keyboard_movement->moving_east - keyboard_movement->moving_west),
               static_cast<double>(keyboard_movement->moving_north - keyboard_movement->moving_south)}
      * registry.get_component<MovementForce>(game_object_id)->value();
}

Vec2d SteeringMovementSystem::calculate_steering_force(GameObjectID game_object_id) {
  // Determine if the movement state should change or not
  auto steering_movement = registry.get_component<SteeringMovement>(game_object_id);
  const auto kinematic_owner = registry.get_kinematic_object(game_object_id);
  const auto kinematic_target = registry.get_kinematic_object(steering_movement->target_id);
  if (kinematic_owner->position.distance_to(kinematic_target->position) <= TARGET_DISTANCE) {
    steering_movement->movement_state = SteeringMovementState::Target;
  } else if (!steering_movement->path_list.empty()) {
    steering_movement->movement_state = SteeringMovementState::Footprint;
  } else {
    steering_movement->movement_state = SteeringMovementState::Default;
  }

  // Calculate the new force to apply to the game object
  Vec2d steering_force{0, 0};
  std::random_device random_device;
  std::mt19937_64 number_generator{random_device()};
  std::uniform_int_distribution<> degree_distribution{0, 360};
  for (const auto &behaviour : steering_movement->behaviours[steering_movement->movement_state]) {
    switch (behaviour) {
      case SteeringBehaviours::Arrive:steering_force += arrive(kinematic_owner->position, kinematic_target->position);
        break;
      case SteeringBehaviours::Evade:
        steering_force += evade(kinematic_owner->position,
                                kinematic_target->position,
                                kinematic_target->velocity);
        break;
      case SteeringBehaviours::Flee:steering_force += flee(kinematic_owner->position, kinematic_target->position);
        break;
      case SteeringBehaviours::FollowPath:
        steering_force += follow_path(kinematic_owner->position, steering_movement->path_list);
        break;
      case SteeringBehaviours::ObstacleAvoidance:
        steering_force += obstacle_avoidance(kinematic_owner->position,
                                             kinematic_owner->velocity,
                                             registry.get_walls());
        break;
      case SteeringBehaviours::Pursuit:
        steering_force += pursuit(kinematic_owner->position,
                                  kinematic_target->position,
                                  kinematic_target->velocity);
        break;
      case SteeringBehaviours::Seek:steering_force += seek(kinematic_owner->position, kinematic_target->position);
        break;
      case SteeringBehaviours::Wander:
        steering_force += wander(kinematic_owner->velocity, degree_distribution(number_generator));
        break;
    }
  }

  // Return the normalised steering force
  return steering_force.normalised() * registry.get_component<MovementForce>(game_object_id)->value();
}

void SteeringMovementSystem::update_path_list(GameObjectID target_game_object_id, const std::deque<Vec2d> &footprints) {
  // Update the path list for all SteeringMovement components that have the correct target ID
  for (auto &[game_object_id, component_tuple] : registry.find_components<SteeringMovement>()) {
    auto steering_movement = std::get<0>(component_tuple);
    if (steering_movement->target_id != target_game_object_id) {
      continue;
    }

    // Get the current position of the target and clear the path list
    const Vec2d current_position = registry.get_kinematic_object(game_object_id)->position;
    steering_movement->path_list.clear();

    // Get the closest footprint to the target that is still within range of
    // the game object
    auto closest_footprint = footprints.end();
    double closest_distance = TARGET_DISTANCE;
    for (auto it = footprints.begin(); it != footprints.end(); ++it) {
      const double distance = current_position.distance_to(*it);
      if (distance < closest_distance) {
        closest_footprint = it;
        closest_distance = distance;
      }
    }

    // If a footprint was found, set the path list a slice of the footprints
    // list from closest_footprint
    if (closest_footprint != footprints.end()) {
      steering_movement->path_list.assign(closest_footprint, footprints.end());
    }
  }
}
