// Related header
#include "game_objects/registry.hpp"

// Custom headers
#include "game_objects/systems/attacks.hpp"
#include "game_objects/systems/physics.hpp"

// ----- FUNCTIONS ------------------------------
Registry::Registry() {
  // Set the damping to ensure the game objects don't drift
  cpSpaceSetDamping(*_space, DAMPING);

  // Add the collision handlers for the bullets
  createCollisionHandlerFunc(GameObjectType::Player, GameObjectType::Bullet);
  createCollisionHandlerFunc(GameObjectType::Enemy, GameObjectType::Bullet);
  createCollisionHandlerFunc(GameObjectType::Wall, GameObjectType::Bullet);
}

auto Registry::create_game_object(const GameObjectType game_object_type, const cpVect &position,
                                  const std::vector<std::shared_ptr<ComponentBase>> &&components) -> GameObjectID {
  // Add the components to the game object
  _game_objects[_next_game_object_id] = {};
  for (const auto &component : components) {
    // Check if the component already exists in the registry
    const auto &obj{*component};
    if (has_component(_next_game_object_id, typeid(obj))) {
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
      cpShapeSetBody(shape, body);
      cpSpaceAddBody(*_space, body);
      cpSpaceAddShape(*_space, shape);
      _shapes[shape] = _next_game_object_id;
    }

    // Add the component to the registry
    _game_objects[_next_game_object_id][typeid(obj)] = component;
  }

  // Increment the game object ID and return the current game object ID
  _next_game_object_id++;
  return _next_game_object_id - 1;
}

void Registry::delete_game_object(const GameObjectID game_object_id) {
  // Check if the game object is registered or not
  if (!_game_objects.contains(game_object_id)) {
    throw RegistryError("game object", game_object_id);
  }

  // Remove the shape and body from the space if the game object has a kinematic component
  if (has_component(game_object_id, typeid(KinematicComponent))) {
    _shapes.erase(*get_component<KinematicComponent>(game_object_id)->shape);
    cpSpaceRemoveShape(*_space, *get_component<KinematicComponent>(game_object_id)->shape);
    cpSpaceRemoveBody(*_space, *get_component<KinematicComponent>(game_object_id)->body);
  }

  // Delete the game object from the system
  _game_objects.erase(game_object_id);
}

auto Registry::get_component(const GameObjectID game_object_id, const std::type_index &component_type) const
    -> std::shared_ptr<ComponentBase> {
  // Check if the game object has the component or not
  if (!has_component(game_object_id, component_type)) {
    throw RegistryError("game object", game_object_id, " or does not have the required component");
  }

  // Return the specified component
  return _game_objects.at(game_object_id).at(component_type);
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

    // Deal damage to the first shape if it is an entity
    if (static_cast<GameObjectType>(cpShapeGetCollisionType(shape1)) != GameObjectType::Wall) {
      registry->get_system<DamageSystem>()->deal_damage(registry->_shapes[shape1], DAMAGE);
    }

    // Register the post step callback to delete the bullet
    cpSpaceAddPostStepCallback(
        registry->get_space(),
        [](cpSpace * /*space*/, void *key, void *bullet) {
          auto *reg{static_cast<Registry *>(key)};
          reg->delete_game_object(reg->_shapes[static_cast<cpShape *>(bullet)]);
        },
        registry, shape2);
    return cpFalse;
  };
}
