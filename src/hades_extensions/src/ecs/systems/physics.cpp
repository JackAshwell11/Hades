// Related header
#include "ecs/systems/physics.hpp"

// Local headers
#include "ecs/systems/attacks.hpp"
#include "ecs/systems/movements.hpp"
#include "ecs/systems/sprite.hpp"
#include "factories.hpp"

void PhysicsSystem::add_force(const GameObjectID game_object_id, const cpVect &force) const {
  cpBodyApplyForceAtLocalPoint(
      *get_registry()->get_component<KinematicComponent>(game_object_id)->body,
      cpvnormalize(force) * get_registry()->get_component<MovementForce>(game_object_id)->get_value(), cpvzero);
}

void PhysicsSystem::add_bullet(const std::pair<cpVect, cpVect> &bullet, const double damage,
                               const GameObjectType source_type) const {
  const auto bullet_id{get_registry()->create_game_object(GameObjectType::Bullet, get<0>(bullet),
                                                          get_factories().at(GameObjectType::Bullet)(0))};
  const auto bullet_component{get_registry()->get_component<Bullet>(bullet_id)};
  bullet_component->damage = damage;
  bullet_component->source_type = source_type;
  cpBodySetVelocity(*get_registry()->get_component<KinematicComponent>(bullet_id)->body, get<1>(bullet));
}

auto PhysicsSystem::get_nearest_item(const GameObjectID game_object_id) const -> GameObjectID {
  // Get the current position of the game object
  const cpVect &game_object_position{
      cpBodyGetPosition(*get_registry()->get_component<KinematicComponent>(game_object_id)->body)};

  // Collect all relevant game objects
  std::vector<std::pair<GameObjectID, cpFloat>> distances;
  for (const auto &[id, component_tuple] : get_registry()->find_components<KinematicComponent>()) {
    if (const auto kinematic_component{std::get<0>(component_tuple)};
        game_object_id != id && (cpShapeGetSensor(*kinematic_component->shape) != 0U) &&
        get_registry()->get_game_object_type(id) != GameObjectType::Floor && !kinematic_component->collected) {
      if (const cpFloat distance{
              cpvlength(cpvsub(game_object_position, cpBodyGetPosition(*kinematic_component->body)))};
          distance <= SPRITE_SIZE / 2) {
        distances.emplace_back(id, distance);
      }
    }
  }

  // Return the nearest game object, otherwise, return -1
  const auto nearest_element{std::ranges::min_element(
      distances.begin(), distances.end(), [](const auto &lhs, const auto &rhs) { return lhs.second < rhs.second; })};
  return nearest_element != distances.end() ? nearest_element->first : -1;
}

auto PhysicsSystem::get_wall_distances(const cpVect &current_position) const -> std::vector<cpVect> {
  const auto space{get_registry()->get_space()};
  auto raycast{[&space, &current_position](const cpVect direction) -> cpVect {
    // Calculate the end position of the ray based on the direction
    const auto end_position{current_position + cpvnormalize(direction) * MAX_WALL_DISTANCE};

    // Perform the raycast
    cpSegmentQueryInfo info;
    if (cpSpaceSegmentQueryFirst(space, current_position, end_position, SPRITE_SIZE / 4,
                                 {CP_NO_GROUP, CP_ALL_CATEGORIES, static_cast<cpBitmask>(GameObjectType::Wall)},
                                 &info) != nullptr) {
      return info.point;
    }
    return {MAX_WALL_DISTANCE, MAX_WALL_DISTANCE};
  }};

  // Perform raycasts in all directions
  return {raycast({0, 1}), raycast({1, 0}),  raycast({0, -1}), raycast({-1, 0}),
          raycast({1, 1}), raycast({1, -1}), raycast({-1, 1}), raycast({-1, -1})};
}
