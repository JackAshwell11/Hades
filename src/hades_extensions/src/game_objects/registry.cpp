// Related header
#include "game_objects/registry.hpp"

// Std headers
#include <memory>
#include <typeindex>
#include <unordered_map>

// ----- CLASSES ------------------------------
GameObjectID Registry::create_game_object(bool kinematic) {
  // Add the game object to the system
  game_objects_[next_game_object_id_] = std::unordered_map<ObjectType, std::shared_ptr<ComponentBase>>{};
  if (kinematic) {
    kinematic_objects_[next_game_object_id_] = std::make_unique<KinematicObject>();
  }

  // Increment the game object ID and return the current game object ID
  next_game_object_id_++;
  return next_game_object_id_ - 1;
}

void Registry::delete_game_object(GameObjectID game_object_id) {
  // Check if the game object is registered or not
  if (!game_objects_.contains(game_object_id)) {
    throw RegistryException("game object", game_object_id);
  }

  // Delete the game object from the system
  game_objects_.erase(game_object_id);
  for (auto &[_, ids] : components_) {
    ids.erase(game_object_id);
  }
  if (kinematic_objects_.contains(game_object_id)) {
    kinematic_objects_.erase(game_object_id);
  }
}

std::shared_ptr<ComponentBase> Registry::get_component(GameObjectID game_object_id, ObjectType component_type) const {
  // Check if the game object has the component or not
  if (!has_component(game_object_id, component_type)) {
    throw RegistryException("game object", game_object_id);
  }

  // Return the specified component
  return game_objects_.at(game_object_id).at(component_type);
}

std::shared_ptr<KinematicObject> Registry::get_kinematic_object(GameObjectID game_object_id) const {
  // Check if the game object is registered or not
  if (!kinematic_objects_.contains(game_object_id)) {
    throw RegistryException("game object", game_object_id);
  }

  // Return the kinematic object
  return kinematic_objects_.at(game_object_id);
}
