// Related header
#include "game_objects/registry.hpp"

// Custom headers
#include "game_objects/systems/attacks.hpp"
#include "game_objects/systems/physics.hpp"

// ----- FUNCTIONS ------------------------------
/// Convert a Chipmunk2D data pointer to a game object ID.
///
/// @param data - The Chipmunk2D data pointer to convert.
/// @return The game object ID.
auto cpDataPointerToGameObjectID(void *data) -> GameObjectID {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  return static_cast<GameObjectID>(reinterpret_cast<uintptr_t>(data));
}

Registry::Registry() {
  // Set the damping to ensure the game objects don't drift
  cpSpaceSetDamping(*space_, DAMPING);

  // Add the collision handlers for the bullets
  createCollisionHandlerFunc(GameObjectType::Player, GameObjectType::Bullet);
  createCollisionHandlerFunc(GameObjectType::Enemy, GameObjectType::Bullet);
  createCollisionHandlerFunc(GameObjectType::Wall, GameObjectType::Bullet);
}

auto Registry::create_game_object(const GameObjectType game_object_type, const cpVect &position,
                                  const std::vector<std::shared_ptr<ComponentBase>> &&components) -> GameObjectID {
  // Add the components to the game object
  game_objects_[next_game_object_id_] = {};
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
  notify_callbacks(EventType::GameObjectDeath, game_object_id);
  game_objects_.erase(game_object_id);
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

void Registry::createCollisionHandlerFunc(GameObjectType game_object_one, GameObjectType game_object_two) {
  auto *func = cpSpaceAddCollisionHandler(get_space(), static_cast<cpCollisionType>(game_object_one),
                                          static_cast<cpCollisionType>(game_object_two));
  func->userData = this;
  func->beginFunc = [](cpArbiter *arbiter, cpSpace * /*space*/, void *data) -> cpBool {
    // Get the registry and the shapes that are colliding
    auto *registry{static_cast<Registry *>(data)};
    cpShape *shape1{nullptr};
    cpShape *shape2{nullptr};
    cpArbiterGetShapes(arbiter, &shape1, &shape2);

    // Get the game object IDs of the shapes
    auto collision_data{std::make_unique<std::pair<GameObjectID, GameObjectID>>(
        cpDataPointerToGameObjectID(cpShapeGetUserData(shape1)),
        cpDataPointerToGameObjectID(cpShapeGetUserData(shape2)))};

    // Deal damage to the first shape if it is an entity
    if (static_cast<GameObjectType>(cpShapeGetCollisionType(shape1)) != GameObjectType::Wall) {
      cpSpaceAddPostStepCallback(
          registry->get_space(),
          [](cpSpace * /*space*/, void *ids, void *registry_ptr) {
            const auto *collision_ids{static_cast<std::pair<GameObjectID, GameObjectID> *>(ids)};
            if (const auto *reg{static_cast<Registry *>(registry_ptr)};
                reg->has_component(collision_ids->second, typeid(KinematicComponent))) {
              reg->get_system<DamageSystem>()->deal_damage(collision_ids->first, collision_ids->second);
            }
          },
          collision_data.get(), registry);
    }

    // Register the post step callback to delete the bullet
    cpSpaceAddPostStepCallback(
        registry->get_space(),
        [](cpSpace * /*space*/, void *bullet_id, void *registry_ptr) {
          if (auto *reg{static_cast<Registry *>(registry_ptr)};
              reg->has_component(cpDataPointerToGameObjectID(bullet_id), typeid(KinematicComponent))) {
            reg->delete_game_object(cpDataPointerToGameObjectID(bullet_id));
          }
        },
        cpShapeGetUserData(shape2), registry);

    // Release the collision data so Chipmunk2D can take ownership of it
    [[maybe_unused]] auto *const ptr{collision_data.release()};
    return cpFalse;
  };
}
