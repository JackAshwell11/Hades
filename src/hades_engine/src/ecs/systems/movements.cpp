// Related header
#include "ecs/systems/movements.hpp"

// Std headers
#include <numbers>
#include <random>

// Local headers
#include "ecs/registry.hpp"
#include "ecs/steering.hpp"
#include "ecs/systems/physics.hpp"

namespace {
/// The footprint interval.
constexpr auto FOOTPRINT_INTERVAL{0.5};

/// The footprint limit.
constexpr auto FOOTPRINT_LIMIT{5};

/// The view distance.
constexpr auto VIEW_DISTANCE{3 * SPRITE_SIZE};

/// Calculate the new steering force to apply to the game object.
///
/// @param registry - The registry that manages the game objects, components, and systems.
/// @param steering_movement - The steering movement component of the game object.
/// @param kinematic_owner - The kinematic component of the game object.
/// @param kinematic_target - The kinematic component of the target game object.
auto calculate_steering_force(const Registry* registry, const std::shared_ptr<SteeringMovement>& steering_movement,
                              const cpBody* kinematic_owner, const cpBody* kinematic_target) -> cpVect {
  cpVect steering_force{cpvzero};
  const auto kinematic_owner_position{cpBodyGetPosition(kinematic_owner)};
  const auto kinematic_owner_velocity{cpBodyGetVelocity(kinematic_owner)};
  const auto kinematic_target_position{cpBodyGetPosition(kinematic_target)};
  const auto kinematic_target_velocity{cpBodyGetVelocity(kinematic_target)};
  for (const auto& behaviour : steering_movement->behaviours[steering_movement->movement_state]) {
    switch (behaviour) {
      case SteeringBehaviours::Arrive:
        steering_force += arrive(kinematic_owner_position, kinematic_target_position);
        break;
      case SteeringBehaviours::Evade:
        steering_force += evade(kinematic_owner_position, kinematic_target_position, kinematic_target_velocity);
        break;
      case SteeringBehaviours::Flee:
        steering_force += flee(kinematic_owner_position, kinematic_target_position);
        break;
      case SteeringBehaviours::FollowPath:
        steering_force += follow_path(kinematic_owner_position, steering_movement->path_list);
        break;
      case SteeringBehaviours::ObstacleAvoidance:
        steering_force += obstacle_avoidance(registry->get_space(), kinematic_owner_position, kinematic_owner_velocity);
        break;
      case SteeringBehaviours::Pursue:
        steering_force += pursue(kinematic_owner_position, kinematic_target_position, kinematic_target_velocity);
        break;
      case SteeringBehaviours::Seek:
        steering_force += seek(kinematic_owner_position, kinematic_target_position);
        break;
      case SteeringBehaviours::Wander:
        std::random_device device;
        const auto displacement_angle{std::uniform_real_distribution{0.0, std::numbers::pi * 2}(device)};
        steering_force += wander(kinematic_owner_velocity, displacement_angle);
        break;
    }
  }
  return steering_force;
}
}  // namespace

void FootprintSystem::update(const double delta_time) const {
  // Update the time since the last footprint then check if a new footprint should be created
  for (const auto& [game_object_id, component_tuple] : get_registry()->get_game_object_components<Footprints>()) {
    const auto footprints{std::get<0>(component_tuple)};
    footprints->time_since_last_footprint += delta_time;
    if (footprints->time_since_last_footprint < FOOTPRINT_INTERVAL) {
      return;
    }

    // Reset the counter and create a new footprint making sure to only keep FOOTPRINT_LIMIT footprints
    const cpVect current_position{
        cpBodyGetPosition(*get_registry()->get_component<KinematicComponent>(game_object_id)->body)};
    footprints->time_since_last_footprint = 0;
    if (std::cmp_greater_equal(footprints->footprints.size(), FOOTPRINT_LIMIT)) {
      footprints->footprints.pop_front();
    }
    footprints->footprints.push_back(current_position);

    // Update the path list for all SteeringMovement components
    get_registry()->get_system<SteeringMovementSystem>()->update_path_list(game_object_id, footprints->footprints);
  }
}

void KeyboardMovementSystem::update(const double /*delta_time*/) const {
  for (const auto& [game_object_id, component_tuple] : get_registry()->get_game_object_components<KeyboardMovement>()) {
    const auto keyboard_movement{std::get<0>(component_tuple)};
    get_registry()->get_system<PhysicsSystem>()->add_force(
        game_object_id, {static_cast<double>(static_cast<int>(keyboard_movement->moving_east) -
                                             static_cast<int>(keyboard_movement->moving_west)),
                         static_cast<double>(static_cast<int>(keyboard_movement->moving_north) -
                                             static_cast<int>(keyboard_movement->moving_south))});
  }
}

void SteeringMovementSystem::update(const double /*delta_time*/) const {
  for (const auto& [game_object_id, component_tuple] :
       get_registry()->get_game_object_components<SteeringMovement, KinematicComponent>()) {
    // Unpack the components and check if the target game object exists
    const auto [steering_movement, kinematic_owner] = component_tuple;
    if (!get_registry()->has_component<KinematicComponent>(steering_movement->target_id)) {
      continue;
    }
    const auto kinematic_target{get_registry()->get_component<KinematicComponent>(steering_movement->target_id)};

    // Determine if the movement state should change or not
    if (cpvdist(cpBodyGetPosition(*kinematic_owner->body), cpBodyGetPosition(*kinematic_target->body)) <=
        VIEW_DISTANCE) {
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
                                              const std::deque<cpVect>& footprints) const {
  // Update the path list for all SteeringMovement components that have the correct target ID
  for (const auto& [game_object_id, component_tuple] : get_registry()->get_game_object_components<SteeringMovement>()) {
    const auto steering_movement{std::get<0>(component_tuple)};
    if (steering_movement->target_id != target_game_object_id) {
      continue;
    }

    // Get the current position of the target and clear the path list
    const cpVect current_position{
        cpBodyGetPosition(*get_registry()->get_component<KinematicComponent>(game_object_id)->body)};
    steering_movement->path_list.clear();

    // Get the closest footprint to the target that is still within range of the game object
    auto closest_footprint{footprints.end()};
    double closest_distance{VIEW_DISTANCE};
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
