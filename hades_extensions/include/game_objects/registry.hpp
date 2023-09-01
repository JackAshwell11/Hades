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

// TODO: Implement registry.cpp, systems and look over all includes (need to
//  decide if each file includes everything (even duplicates) or only what it
//  needs (takes from other includes))

// TODO: Need to find someway to have components and system only be subclasses
//  of ComponentBase and SystemBase
// ----- STRUCTURES ------------------------------
// Add a forward declaration for the registry class
class Registry;

/// The base class for all components.
struct ComponentBase {
  /// The default constructor.
  ComponentBase() = default;

  /// The copy constructor.
  ComponentBase(const ComponentBase &component) = default;
};

/// The base class for all systems.
struct SystemBase {
  /// The default constructor.
  SystemBase() = default;

  /// Process update logic for a system.
  ///
  /// @param registry - The registry to process the update logic for.
  /// @param delta_time - The time interval since the last time the function was called.
  virtual void update(Registry &registry, float delta_time) = 0;
};

// ----- CLASSES ------------------------------
/// Raised when an error occurs with the registry.
class RegistryError : public std::runtime_error {
 public:
  /// Initialise the error.
  ///
  /// @tparam T - The type of item that is not registered.
  /// @param not_registered_type - The type of item that is not registered.
  /// @param value - The value that is not registered.
  /// @param error - The error raised by the registry.
  template<typename T>
  RegistryError(const std::string &not_registered_type,
                const T &value,
                const std::string &error = "is not registered with the registry") : std::runtime_error(
      "The " + not_registered_type + " `" + std::to_string(value) + "` " + error + ".") {};
};

/// Manages game objects, components, and systems that are registered.
class Registry {
 public:
  /// The default constructor.
  Registry() = default;

  /// Update all the systems.
  ///
  /// @param delta_time - The time interval since the last time the function was called.
  void update(float delta_time);

  /// Create a new game object.
  ///
  /// @param kinematic - Whether the game object should have a kinematic object or not.
  /// @param components - The components to add to the game object.
  /// @return The game object ID.
  int create_game_object(bool kinematic, std::initializer_list<ComponentBase> components);

  /// Delete a game object.
  ///
  /// @param game_object_id - The game object ID.
  /// @throw RegistryError - If the game object is not registered.
  void delete_game_object(int game_object_id);

  /// Add a system to the registry.
  ///
  /// @param system - The system to add to the registry.
  /// @throw RegistryError - If the system is already registered.
  void add_system(SystemBase system);

  /// Get a system from the registry.
  ///
  /// @tparam T - The type of system to get.
  /// @throw RegistryError - If the system is not registered.
  /// @return The system.
  template<typename T>
  T &get_system(std::type_index system);

  /// Get a component from the registry given a game object ID.
  ///
  /// @param game_object_id - The game object ID.
  /// @tparam T - The type of component to get.
  /// @throw RegistryError - If the game object or component are not registered.
  /// @return The component from the registry.
  template<typename T>
  T &get_component(int game_object_id);

  /// Get the components for all game objects that have the given components.
  ///
  /// @tparam T - The types of components to get.
  /// @return A vector of tuples containing the game object ID and the components.
  template<typename ... T>
  std::vector<std::tuple<int, std::tuple<T &...>>> get_components();

  /// Get a kinematic object given a game object ID.
  ///
  /// @param game_object_id - The game object ID.
  /// @throw RegistryError - If the game object is not registered.
  /// @return The kinematic object.
  KinematicObject &get_kinematic_object(int game_object_id);

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
  std::unordered_set<std::unique_ptr<Vec2d>> walls_;
};
