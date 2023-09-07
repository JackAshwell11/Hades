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

// TODO: Implement systems and look over all includes (need to decide if each
//  file includes everything (even duplicates) or only what it needs (takes
//  from other includes))
// ----- STRUCTURES ------------------------------
// Add a forward declaration for the registry class
class Registry;

/// The base class for all components.
struct ComponentBase {
  /// The default constructor.
  ComponentBase() = default;

  /// The virtual destructor.
  virtual ~ComponentBase() = default;
};

/// The base class for all systems.
struct SystemBase {
  /// The default constructor.
  SystemBase() = default;

  /// The virtual destructor.
  virtual ~SystemBase() = default;

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
  /// The default constructor.
  Registry() = default;

  /// Create a new game object.
  ///
  /// @param kinematic - Whether the game object should have a kinematic object or not.
  /// @param components - The components to add to the game object.
  /// @return The game object ID.
  int create_game_object(bool kinematic, std::vector<std::unique_ptr<ComponentBase>> &&components);

  /// Delete a game object.
  ///
  /// @param game_object_id - The game object ID.
  /// @throw RegistryError - If the game object is not registered.
  void delete_game_object(int game_object_id);

  /// Get a component from the registry given a game object ID.
  ///
  /// @tparam T - The type of component to get.
  /// @param game_object_id - The game object ID.
  /// @return The component from the registry.
  template<typename T>
  T *get_component(int game_object_id) {
    // Check if the game object is registered or not
    if (game_objects_.find(game_object_id) == game_objects_.end()) {
      return nullptr;
    }

    // Check if the component is registered or not
    if (game_objects_[game_object_id].find(typeid(T)) == game_objects_[game_object_id].end()) {
      return nullptr;
    }

    // Return the specified component casting it to T
    return dynamic_cast<T *>(game_objects_[game_object_id][typeid(T)].get());
  }

  /// Get the components for all game objects that have the given components.
  ///
  /// @tparam Ts - The types of components to get.
  /// @return A vector of tuples containing the game object ID and the components.
  template<typename ... Ts>
  std::vector<std::tuple<int, std::tuple<Ts *...>>> get_components() {
    // Create a vector of tuples to store the components
    std::vector<std::tuple<int, std::tuple<Ts *...>>> components;

    // Iterate over all game objects
    for (auto &[game_object_id, game_object_components] : game_objects_) {
      // Check if the game object has all the components using a fold expression
      bool has_components = true;
      ((has_components &= game_object_components.find(typeid(Ts)) != game_object_components.end()), ...);

      // If the game object has all the components, cast them to T and add them
      // to the vector
      if (has_components) {
        auto components_result = std::make_tuple(dynamic_cast<Ts *>(game_object_components[typeid(Ts)].get()) ...);
        components.emplace_back(game_object_id, components_result);
      }
    }

    // Return the component
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

  /// Get a system from the registry.
  ///
  /// @tparam T - The type of system to get.
  /// @return The system.
  template<typename T>
  T *get_system() {
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
  std::unique_ptr<KinematicObject> get_kinematic_object(int game_object_id);

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
  int next_game_object_id_ = 0;

  /// The components registered with the registry.
  std::unordered_map<std::type_index, std::unordered_set<int>> components_;

  /// The game objects registered with the registry.
  std::unordered_map<int, std::unordered_map<std::type_index, std::unique_ptr<ComponentBase>>> game_objects_;

  /// The systems registered with the registry.
  std::unordered_map<std::type_index, std::unique_ptr<SystemBase>> systems_;

  /// The kinematic objects registered with the registry.
  std::unordered_map<int, std::unique_ptr<KinematicObject>> kinematic_objects_;

  /// The walls registered with the registry.
  std::unordered_set<Vec2d> walls_;
};

// TODO: Maybe adding a bunch of helper methods to check for game objects and
//  components in the registry could be helpful in c++ conversion + simplify
//  the registry tests are

// TODO: See if nesting can be reduced to max 3 in python and c++ + reduce
//  abbreviations (https://youtu.be/-J3wNP6u5YU?si=thrvnkBpDg2ScXxF)
