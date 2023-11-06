// Related header
#include "game_objects/registry.hpp"

// Std headers
#include <memory>
#include <typeindex>
#include <unordered_map>

// ----- FUNCTIONS ------------------------------
auto Registry::create_game_object(const bool kinematic) -> GameObjectID {
  // Add the game object to the system
  game_objects_[next_game_object_id_] = std::unordered_map<ObjectType, std::shared_ptr<ComponentBase>>{};
  if (kinematic) {
    kinematic_objects_[next_game_object_id_] = std::make_unique<KinematicObject>();
  }

  // Increment the game object ID and return the current game object ID
  next_game_object_id_++;
  return next_game_object_id_ - 1;
}

void Registry::delete_game_object(const GameObjectID game_object_id) {
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

void Registry::add_components(const GameObjectID game_object_id,
                              const std::vector<std::shared_ptr<ComponentBase>> &&components) {
  // Check if the game object is registered or not
  if (!game_objects_.contains(game_object_id)) {
    throw RegistryException("game object", game_object_id);
  }

  // Add the components to the game object
  for (const auto &component : components) {
    // Check if the component already exists in the registry
    [[maybe_unused]] const auto &obj{*component};
    const std::type_index component_type{typeid(obj)};
    if (has_component(game_object_id, component_type)) {
      continue;
    }

    // Add the component to the registry
    components_[component_type].insert(game_object_id);
    game_objects_[game_object_id][component_type] = component;
  }
}

auto Registry::get_component(const GameObjectID game_object_id, const ObjectType component_type) const
    -> std::shared_ptr<ComponentBase> {
  // Check if the game object has the component or not
  if (!has_component(game_object_id, component_type)) {
    throw RegistryException("game object", game_object_id);
  }

  // Return the specified component
  return game_objects_.at(game_object_id).at(component_type);
}

auto Registry::get_kinematic_object(const GameObjectID game_object_id) const -> std::shared_ptr<KinematicObject> {
  // Check if the game object is registered or not
  if (!kinematic_objects_.contains(game_object_id)) {
    throw RegistryException("game object", game_object_id);
  }

  // Return the kinematic object
  return kinematic_objects_.at(game_object_id);
}
