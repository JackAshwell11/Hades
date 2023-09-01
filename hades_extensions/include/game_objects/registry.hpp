// Ensure this file is only included once
#pragma once

// Std includes
#include <stdexcept>
#include <string>
#include <memory>
#include <typeindex>
#include <unordered_map>
#include <unordered_set>

// Custom includes
#include "steering.hpp"

// ----- STRUCTURES ------------------------------
// Add a forward declaration for the registry class
class Registry;

/// The base class for all components.
struct ComponentBase {
  ComponentBase(const ComponentBase &component) = default;
  ComponentBase() = default;

  virtual ~ComponentBase() = default;
};

/// The base class for all systems.
struct SystemBase {
  virtual ~SystemBase() = default;

  /// Process update logic for a system.
  ///
  /// @param registry - The registry to process the update logic for.
  /// @param delta_time - The time interval since the last time the function was called.
  virtual void update(Registry &registry, float delta_time) = 0;
};

// ----- CLASSES ------------------------------
/// Raised when an error occurs with the registry.
///
/// @param not_registered_type - The type of item that is not registered.
/// @param value - The value that is not registered.
/// @param error - The error raised by the registry.
class RegistryError : public std::runtime_error {
 public:
  template<typename T>
  RegistryError(const std::string &not_registered_type,
                const T &value,
                const std::string &error = "is not registered with the registry") : std::runtime_error(
      "The " + not_registered_type + " `" + std::to_string(value) + "` " + error + ".") {};
};

/// Manages game objects, components, and systems that are registered.
class Registry {
 public:
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
  void add_system(std::unique_ptr<SystemBase> system);

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
  int next_game_object_id = 0;
  std::unordered_map<std::type_index, std::unordered_set<int>> components_;
  std::unordered_map<int, std::unordered_map<std::type_index, std::unique_ptr<ComponentBase>>> game_objects_;
  std::unordered_map<std::type_index, std::unique_ptr<SystemBase>> systems_;
  std::unordered_map<int, std::unique_ptr<KinematicObject>> kinematic_objects_;
  std::unordered_set<std::unique_ptr<Vec2d>> walls_;
};
