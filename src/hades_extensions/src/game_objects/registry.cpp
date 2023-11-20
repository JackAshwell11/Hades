// Related header
#include "game_objects/registry.hpp"

// Std headers
#include <memory>

// ----- FUNCTIONS ------------------------------
auto Registry::create_game_object(const std::vector<std::shared_ptr<ComponentBase>> &&components, const bool kinematic)
    -> GameObjectID {
  // Add the game object to the system
  game_objects_[next_game_object_id_] = {};
  if (kinematic) {
    kinematic_objects_[next_game_object_id_] = std::make_unique<KinematicObject>();
  }

  // Add the components to the game object
  for (const auto &component : components) {
    // Check if the component already exists in the registry
    [[maybe_unused]] const auto &obj{*component};
    if (has_component(next_game_object_id_, typeid(obj))) {
      continue;
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

  // Delete the game object from the system
  game_objects_.erase(game_object_id);
  if (kinematic_objects_.contains(game_object_id)) {
    kinematic_objects_.erase(game_object_id);
  }
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

auto Registry::get_kinematic_object(const GameObjectID game_object_id) const -> std::shared_ptr<KinematicObject> {
  // Check if the game object is registered or not
  if (!kinematic_objects_.contains(game_object_id)) {
    throw RegistryError("game object", game_object_id, " or is not kinematic");
  }

  // Return the kinematic object
  return kinematic_objects_.at(game_object_id);
}
