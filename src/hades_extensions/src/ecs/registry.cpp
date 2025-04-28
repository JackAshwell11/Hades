// Related header
#include "ecs/registry.hpp"

// Custom headers
#include "ecs/systems/attacks.hpp"
#include "ecs/systems/physics.hpp"

namespace {
// The percentage of velocity a game object will retain after a second.
constexpr double DAMPING = 0.0001;

/// Convert a Chipmunk2D data pointer to a game object ID.
///
/// @param data - The Chipmunk2D data pointer to convert.
/// @return The game object ID.
auto cpDataPointerToGameObjectID(void *data) -> GameObjectID {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  return static_cast<GameObjectID>(reinterpret_cast<uintptr_t>(data));
}

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
  const auto player_id{cpDataPointerToGameObjectID(cpShapeGetUserData(shape1))};
  const auto wall_id{cpDataPointerToGameObjectID(cpShapeGetUserData(shape2))};
  const auto player_position{cpBodyGetPosition(*registry->get_component<KinematicComponent>(player_id)->body)};
  if (const auto wall_position{cpBodyGetPosition(*registry->get_component<KinematicComponent>(wall_id)->body)};
      cpvdist(player_position, wall_position) < (SPRITE_SIZE / 2)) {
    registry->mark_for_deletion(player_id);
  }
  return cpTrue;
}
}  // namespace

Registry::Registry(const std::mt19937 &random_generator) : random_generator_{random_generator} {
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
  // Add the components to the game object
  game_objects_[next_game_object_id_] = {};
  game_object_types_[next_game_object_id_] = game_object_type;
  game_object_ids_[game_object_type].push_back(next_game_object_id_);
  for (const auto &component : components) {
    // Check if the component already exists in the registry
    const auto &obj{*component};
    if (has_component(next_game_object_id_, typeid(obj))) {
      continue;
    }

    // Check if the component is a kinematic component. If so, add the body and shape to the space
    if (typeid(obj) == typeid(KinematicComponent)) {
      const auto kinematic_component = std::static_pointer_cast<KinematicComponent>(component);
      auto *const body = *kinematic_component->body;
      auto *const shape = *kinematic_component->shape;
      cpBodySetPosition(body, game_object_type == GameObjectType::Bullet ? position : grid_pos_to_pixel(position));
      cpShapeSetCollisionType(shape, static_cast<cpCollisionType>(game_object_type));
      cpShapeSetFilter(shape, {CP_NO_GROUP, static_cast<cpBitmask>(game_object_type), CP_ALL_CATEGORIES});
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
      cpShapeSetUserData(shape, reinterpret_cast<void *>(static_cast<uintptr_t>(next_game_object_id_)));
      cpShapeSetBody(shape, body);
      cpSpaceAddBody(*space_, body);
      cpSpaceAddShape(*space_, shape);
    }

    // Add the component to the registry
    game_objects_[next_game_object_id_][typeid(obj)] = component;
  }

  // Increment the game object ID and return the current game object ID
  notify<EventType::GameObjectCreation>(next_game_object_id_);
  next_game_object_id_++;
  return next_game_object_id_ - 1;
}

void Registry::delete_game_object(const GameObjectID game_object_id) {
  // Check if the game object is registered or not
  if (!game_objects_.contains(game_object_id)) {
    throw RegistryError("game object", game_object_id);
  }

  // Remove the shape and body from the space if the game object has a kinematic component
  if (has_component(game_object_id, typeid(KinematicComponent))) {
    cpSpaceRemoveShape(*space_, *get_component<KinematicComponent>(game_object_id)->shape);
    cpSpaceRemoveBody(*space_, *get_component<KinematicComponent>(game_object_id)->body);
  }

  // Notify the callbacks then delete the game object
  notify<EventType::GameObjectDeath>(game_object_id);
  std::erase(game_object_ids_[get_game_object_type(game_object_id)], game_object_id);
  game_objects_.erase(game_object_id);
  game_object_types_.erase(game_object_id);
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

auto Registry::get_game_object_ids(const GameObjectType game_object_type) -> std::vector<GameObjectID> {
  const auto ids{game_object_ids_.find(game_object_type)};
  return ids != game_object_ids_.end() ? ids->second : std::vector<GameObjectID>{};
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
    const auto game_object_one{cpDataPointerToGameObjectID(cpShapeGetUserData(shape1))};
    const auto game_object_two{cpDataPointerToGameObjectID(cpShapeGetUserData(shape2))};

    // Deal damage to the first shape if it is an entity
    if (static_cast<GameObjectType>(cpShapeGetCollisionType(shape1)) != GameObjectType::Wall) {
      registry->get_system<DamageSystem>()->deal_damage(game_object_one, game_object_two);
    }

    // Delete the bullet
    registry->mark_for_deletion(game_object_two);

    // Set the collision to be handled
    return cpFalse;
  };
}
