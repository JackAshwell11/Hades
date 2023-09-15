// Std includes
#include <memory>
#include <typeindex>
#include <unordered_map>

// Custom includes
#include "game_objects/registry.hpp"

// ----- CLASSES ------------------------------
GameObjectID Registry::create_game_object(bool kinematic, std::vector<std::unique_ptr<ComponentBase>> &&components) {
  // Add the game object to the system
  game_objects_[next_game_object_id_] = std::unordered_map<ObjectType, std::shared_ptr<ComponentBase>>{};
  if (kinematic) {
    kinematic_objects_[next_game_object_id_] = std::make_unique<KinematicObject>(Vec2d{0, 0}, Vec2d{0, 0});
  }

  // Add the game object to the components
  for (auto &component : components) {
    // Check if the component already exists in the registry
    [[maybe_unused]] auto &component_obj = *component;
    const std::type_info &component_type = typeid(component_obj);
    if (components_[component_type].contains(next_game_object_id_)) {
      continue;
    }

    // Add the component to the registry
    components_[component_type].insert(next_game_object_id_);
    game_objects_[next_game_object_id_][component_type] = std::move(component);
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

void Registry::add_system(std::shared_ptr<SystemBase> system) {
  // Check if the system is already registered
  [[maybe_unused]] auto &system_obj = *system;
  const std::type_info &system_type = typeid(system_obj);
  if (systems_.contains(typeid(system_obj))) {
    throw RegistryException("system", system_type.name(), "is already registered with the registry");
  }

  // Add the system to the registry
  systems_[system_type] = std::move(system);
}

std::shared_ptr<KinematicObject> Registry::get_kinematic_object(GameObjectID game_object_id) {
  // Check if the game object is registered or not
  if (!kinematic_objects_.contains(game_object_id)) {
    throw RegistryException("game object", game_object_id);
  }

  // Return the kinematic object
  return kinematic_objects_[game_object_id];
}
