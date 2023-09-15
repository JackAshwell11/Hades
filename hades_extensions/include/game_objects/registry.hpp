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
using ObjectType = std::type_index;

// Add a forward declaration for the registry class
class Registry;

/// The base class for all components.
struct ComponentBase {
  /// The virtual destructor.
  virtual ~ComponentBase() = default;
};

/// The base class for all systems.
struct SystemBase {
  ///
  Registry &registry;

  /// Initialise the system.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  explicit SystemBase(Registry &registry) : registry(registry) {}

  /// Process update logic for a system.
  ///
  /// @param delta_time - The time interval since the last time the function was called.
  virtual void update(double delta_time) {};
};

// ----- EXCEPTIONS ------------------------------
/// Raised when an error occurs with the registry.
class RegistryException : public std::runtime_error {
 public:
  /// Initialise the exception.
  ///
  /// @tparam T - The type of item that is not registered.
  /// @param not_registered_type - The type of item that is not registered.
  /// @param value - The value that is not registered.
  /// @param error - The error raised by the registry.
  template<typename T>
  RegistryException(const std::string &not_registered_type,
                    const T &value,
                    const std::string &error = "is not registered with the registry") : std::runtime_error(
      "The " + not_registered_type + " `" + to_string(value) + "` " + error + ".") {};

 private:
  /// Convert a value to a string.
  ///
  /// @param value - The value to convert to a string.
  /// @return The value as a string.
  static std::string to_string(const char *value) {
    return {value};
  }

  /// Convert a value to a string.
  ///
  /// @tparam T - The type of value to convert to a string.
  /// @param value - The value to convert to a string.
  /// @return The value as a string.
  template<typename T>
  static std::string to_string(const T &value) {
    return std::to_string(value);
  }
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
  /// @throw RegistryException - If the game object is not registered.
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
  /// @throw RegistryException - If the game object is not registered or if the game object does not have the component.
  /// @return The component from the registry.
  template<typename T>
  std::shared_ptr<T> get_component(GameObjectID game_object_id) {
    // Check if the game object has the component or not
    if (!has_component<T>(game_object_id)) {
      throw RegistryException("game object", game_object_id);
    }

    // Return the specified component casting it to T
    return std::static_pointer_cast<T>(game_objects_[game_object_id][typeid(T)]);
  }

  // TODO: Could may experiment with intersection/union idea where current
  //  implementation is intersection of all components and adding a `complete`
  //  or `full` default true parameter may allow us to enable union which will
  //  return all matches that have at least one of the components

  /// Find all the game objects that have the required components.
  ///
  /// @tparam Ts - The types of components to find.
  /// @return A vector of tuples containing the game object ID and the required components.
  template<typename ... Ts>
  std::vector<std::tuple<GameObjectID, std::tuple<std::shared_ptr<Ts> ...>>> find_components() {
    // Create a vector of tuples to store the components
    std::vector<std::tuple<GameObjectID, std::tuple<std::shared_ptr<Ts> ...>>> components;

    // Iterate over all game objects
    for (auto &[game_object_id, game_object_components] : game_objects_) {
      // Check if the game object has all the components using a fold expression
      if (!(has_component<Ts>(game_object_id) && ...)) {
        continue;
      }

      // Game object has all the components so cast them to T and add them to
      // the vector
      auto components_result = std::make_tuple(std::static_pointer_cast<Ts>(game_object_components[typeid(Ts)]) ...);
      components.emplace_back(game_object_id, components_result);
    }

    // Return the components
    return components;
  }

  /// Add a system to the registry.
  ///
  /// @tparam T - The type of system to add.
  /// @param system - The system to add to the registry.
  /// @throw RegistryException - If the system is already registered.
  void add_system(std::shared_ptr<SystemBase> system);

  /// Find a system in the registry.
  ///
  /// @tparam T - The type of system to find.
  /// @throw RegistryException - If the system is not registered.
  /// @return The system.
  template<typename T>
  std::shared_ptr<T> find_system() {
    // Check if the system is registered
    auto system_result = systems_.find(typeid(T));
    if (system_result == systems_.end()) {
      throw RegistryException("system", typeid(T).name());
    }

    // Return the system casting it to T
    return std::static_pointer_cast<T>(system_result->second);
  }

  /// Update all the systems.
  ///
  /// @param delta_time - The time interval since the last time the function was called.
  inline void update(double delta_time) {
    for (auto &[_, system] : systems_) {
      system->update(delta_time);
    }
  }

  /// Get a kinematic object given a game object ID.
  ///
  /// @param game_object_id - The game object ID.
  /// @throw RegistryException - If the game object is not registered.
  /// @return The kinematic object.
  std::shared_ptr<KinematicObject> get_kinematic_object(GameObjectID game_object_id);

  /// Add a wall to the system
  ///
  /// @param wall - The wall to add to the system.
  inline void add_wall(const Vec2d &wall) {
    walls_.emplace(wall);
  }

  /// Get the walls in the system.
  ///
  /// @return The walls in the system.
  [[nodiscard]] const inline std::unordered_set<Vec2d> &get_walls() const {
    return walls_;
  }

 private:
  /// The next game object ID to use.
  GameObjectID next_game_object_id_ = 0;

  /// The components registered with the registry.
  std::unordered_map<ObjectType, std::unordered_set<GameObjectID>> components_;

  /// The game objects registered with the registry.
  std::unordered_map<GameObjectID, std::unordered_map<ObjectType, std::shared_ptr<ComponentBase>>> game_objects_;

  /// The systems registered with the registry.
  std::unordered_map<ObjectType, std::shared_ptr<SystemBase>> systems_;

  /// The kinematic objects registered with the registry.
  std::unordered_map<GameObjectID, std::shared_ptr<KinematicObject>> kinematic_objects_;

  /// The walls registered with the registry.
  std::unordered_set<Vec2d> walls_;
};
