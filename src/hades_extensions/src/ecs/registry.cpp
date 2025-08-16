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

/// The collision handler for checking if the player is inside a wall.
///
/// @param arbiter - The arbiter for the collision.
/// @param data - The registry.
/// @return Always true to allow the collision to continue.
auto player_wall_collision_handler(cpArbiter *arbiter, cpSpace * /*space*/, void *data) -> cpBool {
  // Get the registry and the shapes that are colliding
  auto *registry{static_cast<Registry *>(data)};
  cpShape *shape1{nullptr};
  cpShape *shape2{nullptr};
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

#ifdef __GNUC__
/// Demangle the type name.
///
/// @param type - The type to demangle.
/// @return The demangled type name.
auto demangle(const std::type_index &type) -> std::string {
  int status;
  const std::unique_ptr<char, void (*)(void *)> res{abi::__cxa_demangle(type.name(), nullptr, nullptr, &status),
                                                    std::free};
  return (status == 0) ? res.get() : type.name();
}
#else
/// Demangle the type name.
///
/// @param type - The type to demangle.
/// @return The demangled type name.
auto demangle(const std::type_index &type) -> std::string { return std::string(type.name()).substr(7); }
#endif

auto RegistryError::to_string(const std::type_index &value) -> std::string { return demangle(value); }

auto RegistryError::to_string(const GameObjectID value) -> std::string { return std::to_string(value); }

Registry::Registry() {
  // Set the damping to ensure the game objects don't drift
  cpSpaceSetDamping(*space_, DAMPING);

  // Add the bullet collision handlers
  createBulletCollisionHandler(GameObjectType::Player);
  createBulletCollisionHandler(GameObjectType::Enemy);
  createBulletCollisionHandler(GameObjectType::Wall);

  // Add a collision handler to prevent the player from going through walls
  auto *func{cpSpaceAddCollisionHandler(get_space(), static_cast<cpCollisionType>(GameObjectType::Player),
                                        static_cast<cpCollisionType>(GameObjectType::Wall))};
  func->userData = this;
  func->preSolveFunc = player_wall_collision_handler;
}

auto Registry::create_game_object(const GameObjectType game_object_type, const cpVect &position,
                                  const std::vector<std::shared_ptr<ComponentBase>> &&components) -> GameObjectID {
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

  // Add the components to the game object
  game_objects_[game_object_id] = {};
  game_object_types_[game_object_id] = game_object_type;
  game_object_ids_[game_object_type].push_back(game_object_id);
  auto game_object_position{game_object_type == GameObjectType::Bullet ? position : grid_pos_to_pixel(position)};
  for (const auto &component : components) {
    // Check if the component already exists in the registry
    const auto &obj{*component};
    if (has_component(game_object_id, typeid(obj))) {
      continue;
    }

    // Check if the component is a kinematic component. If so, add the body and shape to the space
    if (typeid(obj) == typeid(KinematicComponent)) {
      const auto kinematic_component{std::static_pointer_cast<KinematicComponent>(component)};
      auto *const body{*kinematic_component->body};
      auto *const shape{*kinematic_component->shape};
      cpBodySetPosition(body, game_object_position);
      cpShapeSetCollisionType(shape, static_cast<cpCollisionType>(game_object_type));
      cpShapeSetFilter(shape, {CP_NO_GROUP, static_cast<cpBitmask>(game_object_type), CP_ALL_CATEGORIES});
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
      cpShapeSetUserData(shape, reinterpret_cast<void *>(static_cast<uintptr_t>(game_object_id)));
      cpShapeSetBody(shape, body);
      cpSpaceAddBody(*space_, body);
      cpSpaceAddShape(*space_, shape);
    }

    // Add the component to the registry
    game_objects_[game_object_id][typeid(obj)] = component;
  }

  // Increment the game object ID and return the current game object ID
  notify<EventType::GameObjectCreation>(game_object_id, game_object_type,
                                        std::pair{game_object_position.x, game_object_position.y});
  return game_object_id;
}

void Registry::delete_game_object(const GameObjectID game_object_id) {
  // Check if the game object is registered or not
  if (!game_objects_.contains(game_object_id)) {
    throw RegistryError("game object", game_object_id);
  }

  // Remove the shape and body from the space if the game object has a kinematic component
  if (has_component(game_object_id, typeid(KinematicComponent))) {
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
  std::erase(game_object_ids_[get_game_object_type(game_object_id)], game_object_id);
  game_objects_.erase(game_object_id);
  game_object_types_.erase(game_object_id);
  recycled_ids_.push(game_object_id);
}

void Registry::clear_game_objects(const std::unordered_set<GameObjectID> &game_object_ids_to_preserve) {
  std::unordered_set<GameObjectID> ids_to_delete;
  for (const auto game_object_id : game_objects_ | std::views::keys) {
    if (game_object_ids_to_preserve.contains(game_object_id)) {
      continue;
    }
    ids_to_delete.insert(game_object_id);
  }
  for (const auto game_object_id : ids_to_delete) {
    delete_game_object(game_object_id);
  }
}

auto Registry::has_game_object(const GameObjectID game_object_id) const -> bool {
  return game_objects_.contains(game_object_id);
}

auto Registry::has_component(const GameObjectID game_object_id, const std::type_index &component_type) const -> bool {
  // Check if the game object is registered or not
  if (!has_game_object(game_object_id)) {
    return false;
  }

  // Check if the game object has the component or not
  return game_objects_.at(game_object_id).contains(component_type);
}

auto Registry::get_component(const GameObjectID game_object_id, const std::type_index &component_type) const
    -> std::shared_ptr<ComponentBase> {
  // Check if the game object has the component or not
  if (!has_component(game_object_id, component_type)) {
    throw RegistryError(game_object_id, component_type);
  }

  // Return the specified component
  return game_objects_.at(game_object_id).at(component_type);
}

auto Registry::get_game_object_type(const GameObjectID game_object_id) const -> GameObjectType {
  // Check if the game object is registered or not
  if (!game_object_types_.contains(game_object_id)) {
    throw RegistryError("game object", game_object_id);
  }

  // Return the game object type
  return game_object_types_.at(game_object_id);
}

auto Registry::get_game_object_ids(const GameObjectType game_object_type) const -> std::vector<GameObjectID> {
  const auto ids{game_object_ids_.find(game_object_type)};
  return ids != game_object_ids_.end() ? ids->second : std::vector<GameObjectID>{};
}

auto Registry::get_game_object_components(const GameObjectID game_object_id) const
    -> std::vector<std::shared_ptr<ComponentBase>> {
  // Check if the game object is registered or not
  if (!has_game_object(game_object_id)) {
    throw RegistryError("game object", game_object_id);
  }

  // Return the components of the game object
  std::vector<std::shared_ptr<ComponentBase>> components;
  for (const auto &component : std::views::values(game_objects_.at(game_object_id))) {
    components.push_back(component);
  }
  return components;
}

void Registry::mark_for_deletion(const GameObjectID game_object_id) { objects_to_delete_.insert(game_object_id); }

void Registry::update(const double delta_time) {
  // Update all the systems in the registry
  for (const auto &[_, system] : systems_) {
    system->update(delta_time);
  }

  // Delete all marked game objects
  for (const auto game_object_id : objects_to_delete_) {
    delete_game_object(game_object_id);
  }
  objects_to_delete_.clear();
}

void Registry::createBulletCollisionHandler(GameObjectType game_object_type) {
  auto *func{cpSpaceAddCollisionHandler(get_space(), static_cast<cpCollisionType>(game_object_type),
                                        static_cast<cpCollisionType>(GameObjectType::Bullet))};
  func->userData = this;
  func->beginFunc = [](cpArbiter *arbiter, cpSpace * /*space*/, void *data) -> cpBool {
    // Get the registry and the shapes that are colliding
    auto *registry{static_cast<Registry *>(data)};
    cpShape *shape1{nullptr};
    cpShape *shape2{nullptr};
    cpArbiterGetShapes(arbiter, &shape1, &shape2);

    // Get the game object IDs of the shapes
    const auto game_object_one{cpShapeToGameObjectID(shape1)};
    const auto game_object_two{cpShapeToGameObjectID(shape2)};

    // Check if we should handle this collision or not
    const auto bullet{registry->get_component<Bullet>(game_object_two)};
    if (bullet->source_type == registry->get_game_object_type(game_object_one)) {
      return true;
    }

    // Deal damage to the first shape if it is an entity
    if (static_cast<GameObjectType>(cpShapeGetCollisionType(shape1)) != GameObjectType::Wall) {
      registry->get_system<DamageSystem>()->deal_damage(game_object_one, bullet->damage);
    }

    // Delete the bullet
    registry->mark_for_deletion(game_object_two);

    // Set the collision to be handled
    return false;
  };
}
