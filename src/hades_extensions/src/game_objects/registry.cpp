// Related header
#include "game_objects/registry.hpp"

// Custom headers
#include "game_objects/systems/physics.hpp"

// ----- FUNCTIONS ------------------------------
auto Registry::create_game_object(const cpVect &position,
                                  const std::vector<std::shared_ptr<ComponentBase>> &&components) -> GameObjectID {
  // Add the components to the game object
  game_objects_[next_game_object_id_] = {};
  for (const auto &component : components) {
    // Check if the component already exists in the registry
    [[maybe_unused]] const auto &obj{*component};
    if (has_component(next_game_object_id_, typeid(obj))) {
      continue;
    }

    // Check if the component is a kinematic component. If so, add the body and shape to the space
    if (typeid(obj) == typeid(KinematicComponent)) {
      add_chipmunk_object(*std::static_pointer_cast<KinematicComponent>(component)->body,
                          *std::static_pointer_cast<KinematicComponent>(component)->shape, position);
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

  // Delete the game object from the system
  game_objects_.erase(game_object_id);
}

auto Registry::get_component(const GameObjectID game_object_id, const std::type_index &component_type) const
    -> std::shared_ptr<ComponentBase> {
  // Check if the game object has the component or not
  if (!has_component(game_object_id, component_type)) {
    throw RegistryError("game object", game_object_id, " or does not have the required component");
  }

  // Return the specified component
  return game_objects_.at(game_object_id).at(component_type);
}

void Registry::add_wall(const cpVect &wall) {
  // Add the wall to the set of walls
  walls_.emplace(wall);

  // Create a Chipmunk2D object for the wall
  auto *const body{cpBodyNewStatic()};
  auto *const shape{cpBoxShapeNew(body, SPRITE_SIZE, SPRITE_SIZE, 0.0)};
  add_chipmunk_object(body, shape, wall);
  cpShapeSetCollisionType(shape, static_cast<cpCollisionType>(PhysicsType::Wall));
}

void Registry::add_chipmunk_object(cpBody *body, cpShape *shape, const cpVect &position) const {
  cpSpaceAddBody(*space_, body);
  cpShapeSetBody(shape, body);
  cpSpaceAddShape(*space_, shape);
  cpBodySetPosition(body, grid_pos_to_pixel(position));
}

// TODO: Go over all includes to make sure they're sorted correctly and in correct sections with correct styling
// TODO: Add grid_pos_to_pixel to global bindings (or just game_objects)
// TODO: Try and reduce amount of comments
// TODO: Try and use floats everywhere instead of doubles where possible
