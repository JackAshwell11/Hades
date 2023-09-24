// Ensure this file is only included once
#pragma once

// Std includes
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeindex>

// Custom includes
#include "steering.hpp"

// ----- TYPEDEFS ------------------------------
// Create some type aliases to simplify the code
using ActionFunction = std::function<double(int)>;
using GameObjectID = int;
using ObjectType = std::type_index;

// ----- STRUCTURES ------------------------------
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

  /// Initialise the object.
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
  /// Initialise the object.
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
  static inline std::string to_string(const char *value) {
    return {value};
  }

  /// Convert a value to a string.
  ///
  /// @tparam T - The type of value to convert to a string.
  /// @param value - The value to convert to a string.
  /// @return The value as a string.
  template<typename T>
  static inline std::string to_string(const T &value) {
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
  /// @return The game object ID.
  GameObjectID create_game_object(bool kinematic = false);

  /// Delete a game object.
  ///
  /// @param game_object_id - The game object ID.
  /// @throws RegistryException - If the game object is not registered.
  void delete_game_object(GameObjectID game_object_id);

  /// Checks if a game object has a given component or not.
  ///
  /// @param game_object_id - The game object ID.
  /// @param component_type - The type of component to check for.
  /// @return Whether the game object has the component or not.
  [[nodiscard]] inline bool has_component(GameObjectID game_object_id, ObjectType component_type) const {
    return components_.contains(component_type) && components_.at(component_type).contains(game_object_id);
  }

  /// Add a component to a game object in the registry.
  ///
  /// @tparam ComponentType - The type of component to add.
  /// @tparam Args - The types of arguments to pass to the component constructor.
  /// @param game_object_id - The game object ID.
  /// @param args - The arguments to pass to the component constructor.
  /// @throws RegistryException - If the game object is not registered or if the game object already has the component.
  template<typename ComponentType, typename ... Args>
  void add_component(GameObjectID game_object_id, Args &&... args) {
    // Check if the game object is registered or not
    if (!game_objects_.contains(game_object_id)) {
      throw RegistryException("game object", game_object_id);
    }

    // Check if the component already exists in the registry
    if (has_component(game_object_id, typeid(ComponentType))) {
      return;
    }

    // Add the component to the registry
    components_[typeid(ComponentType)].insert(game_object_id);
    game_objects_[game_object_id][typeid(ComponentType)] = std::make_shared<ComponentType>(std::forward<Args>(args) ...);
  }

  /// Get a component from the registry.
  ///
  /// @tparam T - The type of component to get.
  /// @param game_object_id - The game object ID.
  /// @throws RegistryException - If the game object is not registered or if the game object does not have the component.
  /// @return The component from the registry.
  template<typename T>
  inline std::shared_ptr<T> get_component(GameObjectID game_object_id) const {
    return std::static_pointer_cast<T>(get_component(game_object_id, typeid(T)));
  }

  /// Get a component from the registry.
  ///
  /// @param game_object_id - The game object ID.
  /// @param component_type - The type of component to get.
  /// @return The component from the registry.
  [[nodiscard]] std::shared_ptr<ComponentBase> get_component(GameObjectID game_object_id,
                                                             ObjectType component_type) const;

  /// Find all the game objects that have the required components.
  ///
  /// @tparam Ts - The types of components to find.
  /// @return A vector of tuples containing the game object ID and the required components.
  template<typename ... Ts>
  std::vector<std::tuple<GameObjectID, std::tuple<std::shared_ptr<Ts> ...>>> find_components() const {
    // Create a vector of tuples to store the components
    std::vector<std::tuple<GameObjectID, std::tuple<std::shared_ptr<Ts> ...>>> components;

    // Iterate over all game objects
    for (auto &[game_object_id, game_object_components] : game_objects_) {
      // Check if the game object has all the components using a fold expression
      if (!(has_component(game_object_id, typeid(Ts)) && ...)) {
        continue;
      }

      // Game object has all the components, so cast them to T and add them to
      // the vector
      auto components_result = std::make_tuple(std::static_pointer_cast<Ts>(game_object_components.at(typeid(Ts))) ...);
      components.emplace_back(game_object_id, components_result);
    }

    // Return the components
    return components;
  }

  /// Add a system to the registry.
  ///
  /// @tparam SystemType - The type of system to add.
  /// @throws RegistryException - If the system is already registered.
  template<typename SystemType>
  void add_system() {
    // Check if the system is already registered
    const std::type_info &system_type = typeid(SystemType);
    if (systems_.contains(system_type)) {
      throw RegistryException("system", system_type.name(), "is already registered with the registry");
    }

    // Add the system to the registry
    systems_[system_type] = std::make_shared<SystemType>(*this);
  }

  /// Find a system in the registry.
  ///
  /// @tparam T - The type of system to find.
  /// @throws RegistryException - If the system is not registered.
  /// @return The system.
  template<typename T>
  std::shared_ptr<T> find_system() const {
    // Check if the system is registered
    auto system_result = systems_.find(typeid(T));
    if (system_result == systems_.end()) {
      throw RegistryException("system", typeid(T).name());
    }

    // Return the system casting it to T
    return std::static_pointer_cast<T>(system_result->second);
  }

  /// Update all the systems in the registry.
  ///
  /// @param delta_time - The time interval since the last time the function was called.
  inline void update(double delta_time) const {
    for (auto &[_, system] : systems_) {
      system->update(delta_time);
    }
  }

  /// Get the kinematic object for a game object.
  ///
  /// @param game_object_id - The game object ID.
  /// @throws RegistryException - If the game object is not registered or if the game object does not have a kinematic
  /// object.
  /// @return The kinematic object.
  [[nodiscard]] std::shared_ptr<KinematicObject> get_kinematic_object(GameObjectID game_object_id) const;

  /// Add a wall to the registry.
  ///
  /// @param wall - The wall to add to the registry.
  inline void add_wall(const Vec2d &wall) {
    walls_.emplace(wall);
  }

  /// Get the walls in the registry.
  ///
  /// @return The walls in the registry.
  [[nodiscard]] inline const std::unordered_set<Vec2d> &get_walls() const {
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
