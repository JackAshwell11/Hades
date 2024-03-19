// Related header
#include "game_objects/registry.hpp"

// Custom headers
#include "game_objects/systems/physics.hpp"

// ----- FUNCTIONS ------------------------------
auto Registry::create_game_object(const std::vector<std::shared_ptr<ComponentBase>> &&components) -> GameObjectID {
  // Add the components to the game object
  game_objects_[next_game_object_id_] = {};
  for (const auto &component : components) {
    // Check if the component already exists in the registry
    [[maybe_unused]] const auto &obj{*component};
    if (has_component(next_game_object_id_, typeid(obj))) {
      continue;
    }

    // Add the component to the registry
    game_objects_[next_game_object_id_][typeid(obj)] = component;

    // Check if the component is a kinematic component. If so, add the body to the space
    if (typeid(obj) == typeid(KinematicComponent)) {
      // Get the body and shape from the kinematic component
      auto *const body{*std::static_pointer_cast<KinematicComponent>(component)->body};
      auto *const shape{*std::static_pointer_cast<KinematicComponent>(component)->shape};

      // Add the body and shape to the space
      cpSpaceAddBody(*space_, body);
      cpBodyAddShape(body, shape);
      cpSpaceAddShape(*space_, shape);
    }
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

void Registry::add_wall(const cpVect &wall) const {
  auto *const body{cpSpaceAddBody(*space_, cpBodyNewStatic())};
  auto *const shape{cpBoxShapeNew(body, SPRITE_SIZE, SPRITE_SIZE, 0.0)};
  cpBodyAddShape(body, shape);
  cpSpaceAddShape(*space_, shape);
  cpBodySetPosition(body, wall);
  // TODO: Fix this
}

// TODO: Go over all = initialisers and see if they can switch to {}
