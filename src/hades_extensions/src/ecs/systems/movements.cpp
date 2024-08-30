// Related header
#include "ecs/systems/movements.hpp"

// Std headers
#include <numbers>
#include <random>

// External headers
#include <chipmunk/chipmunk_structs.h>

// Local headers
#include "ecs/registry.hpp"
#include "ecs/systems/physics.hpp"

// ----- FUNCTIONS ------------------------------
/// Calculate the new steering force to apply to the game object.
///
/// @param registry - The registry that manages the game objects, components, and systems.
/// @param steering_movement - The steering movement component of the game object.
/// @param kinematic_owner - The kinematic component of the game object.
/// @param kinematic_target - The kinematic component of the target game object.
auto calculate_steering_force(const Registry *registry, const std::shared_ptr<SteeringMovement> &steering_movement,
                              const cpBody *kinematic_owner, const cpBody *kinematic_target) -> cpVect {
  cpVect steering_force{cpvzero};
  std::random_device random_device;
  std::mt19937_64 number_generator{random_device()};
  for (const auto &behaviour : steering_movement->behaviours[steering_movement->movement_state]) {
    switch (behaviour) {
      case SteeringBehaviours::Arrive:
        steering_force += arrive(kinematic_owner->p, kinematic_target->p);
        break;
      case SteeringBehaviours::Evade:
        steering_force += evade(kinematic_owner->p, kinematic_target->p, kinematic_target->v);
        break;
      case SteeringBehaviours::Flee:
        steering_force += flee(kinematic_owner->p, kinematic_target->p);
        break;
      case SteeringBehaviours::FollowPath:
        steering_force += follow_path(kinematic_owner->p, steering_movement->path_list);
        break;
      case SteeringBehaviours::ObstacleAvoidance:
        steering_force += obstacle_avoidance(registry->get_space(), kinematic_owner->p, kinematic_owner->v);
        break;
      case SteeringBehaviours::Pursue:
        steering_force += pursue(kinematic_owner->p, kinematic_target->p, kinematic_target->v);
        break;
      case SteeringBehaviours::Seek:
        steering_force += seek(kinematic_owner->p, kinematic_target->p);
        break;
      case SteeringBehaviours::Wander:
        steering_force +=
            wander(kinematic_owner->v, std::uniform_real_distribution{0.0, std::numbers::pi * 2}(number_generator));
        break;
    }
  }
  return steering_force;
}

void FootprintSystem::update(const double delta_time) const {
  // Update the time since the last footprint then check if a new footprint should be created
  for (const auto &[game_object_id, component_tuple] : get_registry()->find_components<Footprints>()) {
    const auto footprints{std::get<0>(component_tuple)};
    footprints->time_since_last_footprint += delta_time;
    if (footprints->time_since_last_footprint <
        get_registry()->get_component<FootprintInterval>(game_object_id)->get_value()) {
      return;
    }

    // Reset the counter and create a new footprint making sure to only keep FOOTPRINT_LIMIT footprints
    const cpVect current_position{get_registry()->get_component<KinematicComponent>(game_object_id)->body->p};
    footprints->time_since_last_footprint = 0;
    if (static_cast<int>(footprints->footprints.size()) >=
        get_registry()->get_component<FootprintLimit>(game_object_id)->get_value()) {
      footprints->footprints.pop_front();
    }
    footprints->footprints.push_back(current_position);

    // Update the path list for all SteeringMovement components
    get_registry()->get_system<SteeringMovementSystem>()->update_path_list(game_object_id, footprints->footprints);
  }
}

void KeyboardMovementSystem::update(const double /*delta_time*/) const {
  for (const auto &[game_object_id, component_tuple] : get_registry()->find_components<KeyboardMovement>()) {
    const auto keyboard_movement{std::get<0>(component_tuple)};
    get_registry()->get_system<PhysicsSystem>()->add_force(
        game_object_id, {static_cast<double>(static_cast<int>(keyboard_movement->moving_east) -
                                             static_cast<int>(keyboard_movement->moving_west)),
                         static_cast<double>(static_cast<int>(keyboard_movement->moving_north) -
                                             static_cast<int>(keyboard_movement->moving_south))});
  }
}

void SteeringMovementSystem::update(const double /*delta_time*/) const {
  for (const auto &[game_object_id, component_tuple] :
       get_registry()->find_components<SteeringMovement, KinematicComponent>()) {
    // Unpack the components and check if the target game object exists
    const auto [steering_movement, kinematic_owner] = component_tuple;
    if (!get_registry()->has_component(steering_movement->target_id, typeid(KinematicComponent))) {
      continue;
    }
    const auto kinematic_target{get_registry()->get_component<KinematicComponent>(steering_movement->target_id)};

    // Determine if the movement state should change or not
    if (cpvdist(kinematic_owner->body->p, kinematic_target->body->p) <=
        get_registry()->get_component<ViewDistance>(game_object_id)->get_value()) {
      steering_movement->movement_state = SteeringMovementState::Target;
    } else if (!steering_movement->path_list.empty()) {
      steering_movement->movement_state = SteeringMovementState::Footprint;
    } else {
      steering_movement->movement_state = SteeringMovementState::Default;
    }

    // Calculate and apply the new steering force to the game object
    const auto steering_force{
        calculate_steering_force(get_registry(), steering_movement, *kinematic_owner->body, *kinematic_target->body)};
    get_registry()->get_system<PhysicsSystem>()->add_force(game_object_id, steering_force);
    kinematic_owner->rotation = cpvtoangle(steering_force);
  }
}

void SteeringMovementSystem::update_path_list(const GameObjectID target_game_object_id,
                                              const std::deque<cpVect> &footprints) const {
  // Update the path list for all SteeringMovement components that have the correct target ID
  for (const auto &[game_object_id, component_tuple] : get_registry()->find_components<SteeringMovement>()) {
    const auto steering_movement{std::get<0>(component_tuple)};
    if (steering_movement->target_id != target_game_object_id) {
      continue;
    }

    // Get the current position of the target and clear the path list
    const cpVect current_position{get_registry()->get_component<KinematicComponent>(game_object_id)->body->p};
    steering_movement->path_list.clear();

    // Get the closest footprint to the target that is still within range of the game object
    auto closest_footprint{footprints.end()};
    double closest_distance{get_registry()->get_component<ViewDistance>(game_object_id)->get_value()};
    for (auto it{footprints.begin()}; it != footprints.end(); ++it) {
      if (const double distance{cpvdist(current_position, *it)}; distance < closest_distance) {
        closest_footprint = it;
        closest_distance = distance;
      }
    }

    // If a footprint was found, set the path list a slice of the footprints list from closest_footprint
    if (closest_footprint != footprints.end()) {
      steering_movement->path_list.assign(closest_footprint, footprints.end());
    }
  }
}
