// Related header
#include "ecs/systems/physics.hpp"

// Local headers
#include "ecs/registry.hpp"
#include "ecs/systems/attacks.hpp"
#include "ecs/systems/movements.hpp"
#include "factories.hpp"

void KinematicComponent::reset() {
  cpBodySetPosition(*body, cpvzero);
  cpBodySetVelocity(*body, cpvzero);
  rotation = 0;
}

void PhysicsSystem::update(const double delta_time) const { cpSpaceStep(get_registry()->get_space(), delta_time); }

void PhysicsSystem::add_force(const GameObjectID game_object_id, const cpVect& force) const {
  cpBodyApplyForceAtLocalPoint(
      *get_registry()->get_component<KinematicComponent>(game_object_id)->body,
      cpvnormalize(force) * get_registry()->get_component<MovementForce>(game_object_id)->get_value(), cpvzero);
}

void PhysicsSystem::add_bullet(const std::pair<cpVect, cpVect>& bullet, const double damage,
                               const GameObjectType source_type) const {
  const auto bullet_id{create_game_object(get_registry(), GameObjectType::Bullet, get<0>(bullet))};
  const auto bullet_component{get_registry()->get_component<Bullet>(bullet_id)};
  bullet_component->damage = damage;
  bullet_component->source_type = source_type;
  cpBodySetVelocity(*get_registry()->get_component<KinematicComponent>(bullet_id)->body, get<1>(bullet));
}

auto PhysicsSystem::get_nearest_item(const GameObjectID game_object_id) const -> GameObjectID {
  // Get the current position of the game object
  const cpVect& game_object_position{
      cpBodyGetPosition(*get_registry()->get_component<KinematicComponent>(game_object_id)->body)};

  // Determine what information is needed for the query
  struct QueryInfo {
    GameObjectID nearest_id{-1};
    cpFloat min_distance{std::numeric_limits<cpFloat>::infinity()};
    GameObjectID query_id{};
  } query_info{.query_id = game_object_id};

  // Find the nearest game object that matches the query
  auto callback{[](cpShape* shape, cpVect /*point*/, const cpFloat distance, cpVect /*gradient*/, void* query) -> void {
    auto* info{static_cast<QueryInfo*>(query)};
    const auto shape_id{cpShapeToGameObjectID(shape)};
    if (shape_id == info->query_id) {
      return;
    }
    if (distance < info->min_distance && distance <= SPRITE_SIZE / 2) {
      info->nearest_id = shape_id;
      info->min_distance = distance;
    }
  }};
  cpSpacePointQuery(get_registry()->get_space(), game_object_position, SPRITE_SIZE / 2,
                    {CP_NO_GROUP, CP_ALL_CATEGORIES,
                     ~(static_cast<cpBitmask>(GameObjectType::Wall) | static_cast<cpBitmask>(GameObjectType::Floor))},
                    callback, &query_info);

  // Return the nearest game object ID
  return query_info.nearest_id;
}

auto PhysicsSystem::get_wall_distances(const cpVect& current_position) const -> std::vector<cpVect> {
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
