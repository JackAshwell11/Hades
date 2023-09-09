// Ensure this file is only included once
#pragma once

// Std includes
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <unordered_set>

// Custom includes
#include "steering.hpp"

// ----- STRUCTURES ------------------------------
// Create some type aliases to simplify the code
using GameObjectID = int;

// Add a forward declaration for the registry class
class Registry;

/// The base class for all components.
struct ComponentBase {
  /// The virtual destructor.
  virtual ~ComponentBase() = default;
};

/// The base class for all systems.
struct SystemBase {
  /// Process update logic for a system.
  ///
  /// @param registry - The registry to process the update logic for.
  /// @param delta_time - The time interval since the last time the function was called.
  virtual void update(Registry &registry, float delta_time) {};
};

// ----- CLASSES ------------------------------
/// Manages game objects, components, and systems that are registered.
class Registry {
 public:
  /// Create a new game object.
  ///
  /// @param kinematic - Whether the game object should have a kinematic object or not.
  /// @param components - The components to add to the game object.
  /// @return The game object ID.
  GameObjectID create_game_object(bool kinematic, std::vector<std::unique_ptr<ComponentBase>> &&components);

  /// Delete a game object.
  ///
  /// @param game_object_id - The game object ID.
  /// @throw RegistryError - If the game object is not registered.
  void delete_game_object(GameObjectID game_object_id);

  /// Checks if a game object has a given component or not.
  ///
  /// @tparam T - The type of component to check for.
  /// @param game_object_id - The game object ID.
  /// @return Whether the game object has the component or not.
  template<typename T>
  inline bool has_component(GameObjectID game_object_id) {
    return components_.contains(typeid(T)) && components_[typeid(T)].contains(game_object_id);
  }

  /// Get a component from the registry given a game object ID.
  ///
  /// @tparam T - The type of component to get.
  /// @param game_object_id - The game object ID.
  /// @return The component from the registry.
  template<typename T>
  T *get_component(GameObjectID game_object_id) {
    // Check if the game object has the component or not
    if (!has_component<T>(game_object_id)) {
      return nullptr;
    }

    // Return the specified component casting it to T
    return dynamic_cast<T *>(game_objects_[game_object_id][typeid(T)].get());
  }

  /// Find all the game objects that have the required components.
  ///
  /// @tparam Ts - The types of components to find.
  /// @return A vector of tuples containing the game object ID and the required components.
  template<typename ... Ts>
  std::vector<std::tuple<GameObjectID, std::tuple<Ts *...>>> find_components() {
    // Create a vector of tuples to store the components
    std::vector<std::tuple<GameObjectID, std::tuple<Ts *...>>> components;

    // Iterate over all game objects
    // TODO: See if get_component and has_component can be used here
    for (auto &[game_object_id, game_object_components] : game_objects_) {
      // Check if the game object has all the components using a fold expression
      bool has_components = true;
      ((has_components &= game_object_components.contains(typeid(Ts))), ...);

      // If the game object has all the components, cast them to T and add them
      // to the vector
      if (has_components) {
        auto components_result = std::make_tuple(dynamic_cast<Ts *>(game_object_components[typeid(Ts)].get()) ...);
        components.emplace_back(game_object_id, components_result);
      }
    }

    // Return the components
    return components;
  }

  /// Add a system to the registry.
  ///
  /// @tparam T - The type of system to add.
  /// @param system - The system to add to the registry.
  template<typename T>
  inline void add_system(std::unique_ptr<T> system) {
    systems_[typeid(T)] = std::move(system);
  }

  /// Find a system in the registry.
  ///
  /// @tparam T - The type of system to find.
  /// @return The system.
  template<typename T>
  inline T *find_system() {
    auto system_result = systems_.find(typeid(T));
    return (system_result != systems_.end()) ? dynamic_cast<T *>(system_result->second.get()) : nullptr;
  }

  /// Update all the systems.
  ///
  /// @param delta_time - The time interval since the last time the function was called.
  inline void update(float delta_time) {
    for (auto &[_, system] : systems_) {
      system->update(*this, delta_time);
    }
  }

  /// Get a kinematic object given a game object ID.
  ///
  /// @param game_object_id - The game object ID.
  /// @throw RegistryError - If the game object is not registered.
  /// @return The kinematic object.
  std::unique_ptr<KinematicObject> get_kinematic_object(GameObjectID game_object_id);

  /// Add a wall to the system
  ///
  /// @param wall - The wall to add to the system.
  inline void add_wall(Vec2d wall) {
    walls_.emplace(wall);
  }

  /// Get the walls in the system.
  ///
  /// @return The walls in the system.
  const inline std::unordered_set<Vec2d> &get_walls() const {
    return walls_;
  }

 private:
  /// The next game object ID to use.
  GameObjectID next_game_object_id_ = 0;

  /// The components registered with the registry.
  std::unordered_map<std::type_index, std::unordered_set<GameObjectID>> components_;

  /// The game objects registered with the registry.
  std::unordered_map<GameObjectID, std::unordered_map<std::type_index, std::unique_ptr<ComponentBase>>> game_objects_;

  /// The systems registered with the registry.
  std::unordered_map<std::type_index, std::unique_ptr<SystemBase>> systems_;

  /// The kinematic objects registered with the registry.
  std::unordered_map<GameObjectID, std::unique_ptr<KinematicObject>> kinematic_objects_;

  /// The walls registered with the registry.
  std::unordered_set<Vec2d> walls_;
};

// TODO: Look over all includes (need to decide if each file includes
//  everything (even duplicates) or only what it needs (takes from other
//  includes))

// TODO: Try and add some more type aliases

// TODO: Try and see if all raw pointers can be replaced with smart pointers
//  (maybe switch to shared_ptr (e.g. get_kinematic_object)). Maybe some places
//  can use raw pointers (only really abstract stuff, idk need more research
//  and advice)

// TODO: Maybe change all nullptr to exceptions and add new macro in tests to
//  abstract try catch so message can be tested
