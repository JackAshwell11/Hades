// Std includes
#include <initializer_list>
#include <memory>
#include <typeindex>
#include <unordered_map>

// Custom includes
#include "game_objects/registry.hpp"

// ----- CLASSES ------------------------------
void Registry::update(float delta_time) {
  for (auto &[_, system] : systems_) {
    system->update(*this, delta_time);
  }
}

int Registry::create_game_object(bool kinematic, std::initializer_list<ComponentBase> components) {
  // Add the game object to the system
  if (kinematic) {
    kinematic_objects_[next_game_object_id_] = std::make_unique<KinematicObject>(Vec2d{0, 0}, Vec2d{0, 0});
  }

  // Add the game object to the components
  for (const auto &component : components) {
    components_[typeid(component)].insert(next_game_object_id_);
    game_objects_[next_game_object_id_][typeid(component)] = std::make_unique<ComponentBase>(component);
  }

  // Increment the game object ID and return the current game object ID
  next_game_object_id_++;
  return next_game_object_id_ - 1;
}

void Registry::delete_game_object(int game_object_id) {
  // Check if the game object is registered or not
  if (game_objects_.find(game_object_id) == game_objects_.end()) {
    throw RegistryError("game object", game_object_id);
  }

  // Delete the game object from the system
  game_objects_.erase(game_object_id);
  for (auto [component_type, ids] : components_) {
    ids.erase(game_object_id);
  }
  if (kinematic_objects_.find(game_object_id) != kinematic_objects_.end()) {
    kinematic_objects_.erase(game_object_id);
  }
}

void Registry::add_system(SystemBase system) {}

template<typename T>
T &Registry::get_system(std::type_index system) {}

template<typename T>
T &Registry::get_component(int game_object_id) {}

template<typename ... T>
std::vector<std::tuple<int, std::tuple<T &...>>> Registry::get_components() {}

KinematicObject &Registry::get_kinematic_object(int game_object_id) {
  // Check if the game object is registered or not
  if (kinematic_objects_.find(game_object_id) == kinematic_objects_.end()) {
    throw RegistryError("game object", game_object_id);
  }

  // Return the kinematic object
  return *kinematic_objects_[game_object_id];
}
