// Related header
#include "ecs/registry.hpp"

// Std headers
#ifdef __GNUC__
#include <cxxabi.h>
#endif

// Custom headers
#include "ecs/systems/attacks.hpp"
#include "ecs/systems/physics.hpp"
#include "events.hpp"

namespace {
/// The percentage of velocity a game object will retain after a second.
constexpr double DAMPING{0.0001};

/// Create a Chipmunk2D collision handler to deal with bullet collisions.
///
/// @param registry - The registry to create the collision handler for.
/// @param game_object_type - The type of game object to create the collision handler for.
void create_bullet_collision_handler(Registry& registry, GameObjectType game_object_type) {
  auto* func{cpSpaceAddCollisionHandler(registry.get_space(), static_cast<cpCollisionType>(game_object_type),
                                        static_cast<cpCollisionType>(GameObjectType::Bullet))};
  func->userData = reinterpret_cast<cpDataPointer>(&registry);
  func->beginFunc = [](cpArbiter* arbiter, cpSpace* /*space*/, void* data) -> cpBool {
    // Get the registry and the shapes that are colliding
    auto* registry_val{static_cast<Registry*>(data)};
    cpShape* shape1{nullptr};
    cpShape* shape2{nullptr};
    cpArbiterGetShapes(arbiter, &shape1, &shape2);

    // Get the game object IDs of the shapes
    const auto game_object_one{cpShapeToGameObjectID(shape1)};
    const auto game_object_two{cpShapeToGameObjectID(shape2)};

    // Check if we should handle this collision or not
    const auto bullet{registry_val->get_component<Bullet>(game_object_two)};
    if (bullet->source_type == registry_val->get_game_object_type(game_object_one)) {
      return true;
    }

    // Deal damage to the first shape if it is an entity
    if (static_cast<GameObjectType>(cpShapeGetCollisionType(shape1)) != GameObjectType::Wall) {
      registry_val->get_system<DamageSystem>()->deal_damage(game_object_one, bullet->damage);
    }

    // Delete the bullet
    registry_val->mark_for_deletion(game_object_two);

    // Set the collision to be handled
    return false;
  };
}

/// The collision handler for checking if the player is inside a wall.
///
/// @param arbiter - The arbiter for the collision.
/// @param data - The registry.
/// @return Always true to allow the collision to continue.
auto player_wall_collision_handler(cpArbiter* arbiter, cpSpace* /*space*/, void* data) -> cpBool {
  // Get the registry and the shapes that are colliding
  auto* registry{static_cast<Registry*>(data)};
  cpShape* shape1{nullptr};
  cpShape* shape2{nullptr};
  cpArbiterGetShapes(arbiter, &shape1, &shape2);

  // Delete the player if it is inside the wall
  const auto player_id{cpShapeToGameObjectID(shape1)};
  const auto wall_id{cpShapeToGameObjectID(shape2)};
  const auto player_position{cpBodyGetPosition(*registry->get_component<KinematicComponent>(player_id)->body)};
  if (const auto wall_position{cpBodyGetPosition(*registry->get_component<KinematicComponent>(wall_id)->body)};
      cpvdist(player_position, wall_position) < (SPRITE_SIZE / 2)) {
    registry->mark_for_deletion(player_id);
  }
  return true;
}
}  // namespace

auto type_name_from_info(const std::type_info& info) -> std::string {
#ifdef __GNUC__
  int status{0};
  std::unique_ptr<char, void (*)(void*)> res{abi::__cxa_demangle(info.name(), nullptr, nullptr, &status), std::free};
  return (status == 0 && res) ? std::string(res.get()) : info.name();
#else
  return std::string(info.name()).substr(7);
#endif
}

Registry::Registry() {
  // Set the damping to ensure the game objects don't drift
  cpSpaceSetDamping(*space_, DAMPING);

  // Add the bullet collision handlers
  create_bullet_collision_handler(*this, GameObjectType::Player);
  create_bullet_collision_handler(*this, GameObjectType::Enemy);
  create_bullet_collision_handler(*this, GameObjectType::Wall);

  // Add a collision handler to prevent the player from going through walls
  auto* func{cpSpaceAddCollisionHandler(get_space(), static_cast<cpCollisionType>(GameObjectType::Player),
                                        static_cast<cpCollisionType>(GameObjectType::Wall))};
  func->userData = this;
  func->preSolveFunc = player_wall_collision_handler;
}

auto Registry::create_game_object(const GameObjectType game_object_type) -> GameObjectID {
  // Get the game object ID to use
  GameObjectID game_object_id;
  if (!recycled_ids_.empty()) {
    // Reuse a recycled ID if available
    game_object_id = recycled_ids_.front();
    recycled_ids_.pop();
  } else {
    // Use the next game object ID
    game_object_id = next_game_object_id_;
    next_game_object_id_++;
  }

  // Create the game object
  game_object_types_[game_object_id] = game_object_type;
  game_object_ids_[game_object_type].push_back(game_object_id);
  notify<EventType::GameObjectCreation>(game_object_id, game_object_type);
  return game_object_id;
}

void Registry::delete_game_object(const GameObjectID game_object_id) {
  // Check if the game object is registered or not
  if (!has_game_object(game_object_id)) {
    throw RegistryError(game_object_id);
  }

  // Remove the shape and body from the space if the game object has a kinematic component
  if (has_component<KinematicComponent>(game_object_id)) {
    const auto kinematic_component{get_component<KinematicComponent>(game_object_id)};
    const auto shape{*kinematic_component->shape};
    const auto body{*kinematic_component->body};
    if (cpSpaceContainsShape(*space_, shape) && cpSpaceContainsBody(*space_, body)) {
      cpSpaceRemoveShape(*space_, shape);
      cpSpaceRemoveBody(*space_, body);
    }
  }

  // Notify the callbacks then delete the game object
  notify<EventType::GameObjectDeath>(game_object_id);
  for (auto& component_map : components_ | std::views::values) {
    component_map.erase(game_object_id);
  }
  std::erase(game_object_ids_[get_game_object_type(game_object_id)], game_object_id);
  game_object_types_.erase(game_object_id);
  recycled_ids_.push(game_object_id);
}

void Registry::clear_game_objects(const std::unordered_set<GameObjectID>& game_object_ids_to_preserve) {
  std::unordered_set<GameObjectID> ids_to_delete;
  for (const auto& game_object_id : game_object_types_ | std::views::keys) {
    if (!game_object_ids_to_preserve.contains(game_object_id)) {
      ids_to_delete.insert(game_object_id);
    }
  }
  for (const auto game_object_id : ids_to_delete) {
    delete_game_object(game_object_id);
  }
}

auto Registry::has_game_object(const GameObjectID game_object_id) const -> bool {
  return game_object_types_.contains(game_object_id);
}

auto Registry::get_game_object_type(const GameObjectID game_object_id) const -> GameObjectType {
  if (!game_object_types_.contains(game_object_id)) {
    throw RegistryError(game_object_id);
  }
  return game_object_types_.at(game_object_id);
}

auto Registry::get_game_object_ids(const GameObjectType game_object_type) const -> std::vector<GameObjectID> {
  const auto ids{game_object_ids_.find(game_object_type)};
  return ids != game_object_ids_.end() ? ids->second : std::vector<GameObjectID>{};
}

void Registry::mark_for_deletion(const GameObjectID game_object_id) { objects_to_delete_.insert(game_object_id); }

void Registry::update(const double delta_time) {
  // Update all the systems in the registry
  for (const auto& system : systems_ | std::views::values) {
    system->update(delta_time);
  }

  // Delete all marked game objects
  for (const auto game_object_id : objects_to_delete_) {
    delete_game_object(game_object_id);
  }
  objects_to_delete_.clear();
}
