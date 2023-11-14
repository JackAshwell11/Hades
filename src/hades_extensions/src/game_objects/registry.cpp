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
    // TODO: See if this can be simplified (maybe just decltype)
    const std::string component_type{ComponentIdentifier<std::remove_reference_t<decltype(*component)>>::identifier};
    if (has_component(next_game_object_id_, component_type)) {
      continue;
    }

    // Add the component to the registry
    game_objects_[next_game_object_id_][component_type] = component;
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

auto Registry::get_component(const GameObjectID game_object_id, const std::string &component_type) const
    -> std::shared_ptr<ComponentBase> {
  // Check if the game object has the component or not
  if (!has_component(game_object_id, component_type)) {
    throw RegistryError("game object", game_object_id);
  }

  // Return the specified component
  return game_objects_.at(game_object_id).at(component_type);
}

auto Registry::get_system(const std::string &system_type) const -> std::shared_ptr<SystemBase> {
  // Check if the system is registered
  auto system_result{systems_.find(system_type)};
  if (system_result == systems_.end()) {
    throw RegistryError("system type", system_type);
  }

  // Return the system
  return system_result->second;
}

auto Registry::get_kinematic_object(const GameObjectID game_object_id) const -> std::shared_ptr<KinematicObject> {
  // Check if the game object is registered or not
  if (!kinematic_objects_.contains(game_object_id)) {
    throw RegistryError("game object", game_object_id);
  }

  // Return the kinematic object
  return kinematic_objects_.at(game_object_id);
}
