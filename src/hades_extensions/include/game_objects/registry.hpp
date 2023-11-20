// Ensure this file is only included once
#pragma once

// Std headers
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeindex>

// Local headers
#include "steering.hpp"

// ----- TYPEDEFS ------------------------------
// Represents unique identifiers for game objects
using GameObjectID = int;
using ActionFunction = std::function<double(int)>;

// ----- BASE TYPES ------------------------------
// Add a forward declaration for the registry class
class Registry;

/// The base class for all components.
struct ComponentBase {
  /// The copy assignment operator.
  ///
  /// @param other - The other component to copy.
  auto operator=(const ComponentBase &) -> ComponentBase & = default;

  /// The move assignment operator.
  ///
  /// @param other - The other component to move.
  auto operator=(ComponentBase &&) -> ComponentBase & = default;

  /// The default constructor.
  ComponentBase() = default;

  /// The virtual destructor.
  virtual ~ComponentBase() = default;

  /// The copy constructor.
  ///
  /// @param other - The other component to copy.
  ComponentBase(const ComponentBase &) = default;

  /// The move constructor.
  ///
  /// @param other - The other component to move.
  ComponentBase(ComponentBase &&) = default;
};

/// The base class for all systems.
class SystemBase {
 public:
  /// The copy assignment operator.
  ///
  /// @param other - The other system to copy.
  auto operator=(const SystemBase &) -> SystemBase & = default;

  /// The move assignment operator.
  ///
  /// @param other - The other system to move.
  auto operator=(SystemBase &&) -> SystemBase & = default;

  /// Initialise the object.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  explicit SystemBase(Registry *registry) : registry(registry) {}

  /// The virtual destructor.
  virtual ~SystemBase() = default;

  /// The copy constructor.
  ///
  /// @param other - The other system to copy.
  SystemBase(const SystemBase &) = default;

  /// The move constructor.
  ///
  /// @param other - The other system to move.
  SystemBase(SystemBase &&) = default;

  /// Get the registry that manages the game objects, components, and systems.
  ///
  /// @return The registry that manages the game objects, components, and systems.
  [[nodiscard]] inline auto get_registry() const -> Registry * { return registry; }

  /// Process update logic for a system.
  ///
  /// @param delta_time - The time interval since the last time the function was called.
  virtual void update(double delta_time) const {};

 private:
  /// The registry that manages the game objects, components, and systems.
  Registry *registry;
};

// ----- EXCEPTIONS ------------------------------
/// Raised when an error occurs with the registry.
struct RegistryError : public std::runtime_error {
  /// Initialise the object
  ///
  /// @param not_registered_type - The type of item that is not registered.
  explicit RegistryError(const std::string &error = "is not registered with the registry")
      : std::runtime_error("The templated type " + error + "."){};

  /// Initialise the object.
  ///
  /// @tparam T - The type of item that is not registered.
  /// @param not_registered_type - The type of item that is not registered.
  /// @param value - The value that is not registered.
  template <typename T>
  RegistryError(const std::string &not_registered_type, const T &value, const std::string &extra = "")
      : std::runtime_error("The " + not_registered_type + " `" + std::to_string(value) +
                           "` is not registered with the registry" + extra + "."){};
};

// ----- CLASSES ------------------------------
/// Manages game objects, components, and systems that are registered.
class Registry {
 public:
  /// Create a new game object.
  ///
  /// @param components - The components to add to the game object.
  /// @param kinematic - Whether the game object should have a kinematic object or not.
  /// @return The game object ID.
  auto create_game_object(const std::vector<std::shared_ptr<ComponentBase>> &&components, bool kinematic = false)
      -> GameObjectID;

  /// Delete a game object.
  ///
  /// @param game_object_id - The game object ID.
  /// @throws RegistryError - If the game object is not registered.
  void delete_game_object(GameObjectID game_object_id);

  /// Checks if a game object has a given component or not.
  ///
  /// @param game_object_id - The game object ID.
  /// @param component_type - The type of component to check for.
  /// @return Whether the game object has the component or not.
  [[nodiscard]] inline auto has_component(const GameObjectID game_object_id,
                                          const std::type_index &component_type) const -> bool {
    return game_objects_.contains(game_object_id) && game_objects_.at(game_object_id).contains(component_type);
  }

  /// Get a component from the registry.
  ///
  /// @tparam T - The type of component to get.
  /// @param game_object_id - The game object ID.
  /// @throws RegistryError - If the game object is not registered or if the game object does not have the component.
  /// @return The component from the registry.
  template <typename T>
  inline auto get_component(const GameObjectID game_object_id) const -> std::shared_ptr<T> {
    return std::static_pointer_cast<T>(get_component(game_object_id, typeid(T)));
  }

  /// Get a component from the registry.
  ///
  /// @param game_object_id - The game object ID.
  /// @param component_type - The type of component to get.
  /// @throws RegistryError - If the game object is not registered or if the game object does not have the component.
  /// @return The component from the registry.
  [[nodiscard]] auto get_component(GameObjectID game_object_id, const std::type_index &component_type) const
      -> std::shared_ptr<ComponentBase>;

  /// Find all the game objects that have the required components.
  ///
  /// @tparam Ts - The types of components to find.
  /// @return A vector of tuples containing the game object ID and the required components.
  template <typename... Ts>
  auto find_components() const -> std::vector<std::tuple<GameObjectID, std::tuple<std::shared_ptr<Ts>...>>> {
    // Create a vector of tuples to store the components
    std::vector<std::tuple<GameObjectID, std::tuple<std::shared_ptr<Ts>...>>> components;

    // Iterate over all game objects
    for (const auto &[game_object_id, game_object_components] : game_objects_) {
      // Check if the game object has all the components using a fold expression
      if (!(has_component(game_object_id, typeid(Ts)) && ...)) {
        continue;
      }

      // Game object has all the components, so cast them to T and add them to the vector
      auto components_result{std::make_tuple(std::static_pointer_cast<Ts>(game_object_components.at(typeid(Ts)))...)};
      components.emplace_back(game_object_id, components_result);
    }

    // Return the components
    return components;
  }

  /// Add a system to the registry.
  ///
  /// @tparam T - The type of system to add.
  /// @throws RegistryError - If the system is already registered.
  template <typename T>
  void add_system() {
    // Check if the system is already registered
    const std::type_index system_type{typeid(T)};
    if (systems_.contains(system_type)) {
      throw RegistryError("is already registered with the registry");
    }

    // Add the system to the registry
    systems_[system_type] = std::make_shared<T>(this);
  }

  /// Get a system from the registry.
  ///
  /// @tparam T - The type of system to get.
  /// @throws RegistryError - If the system is not registered.
  /// @return The system from the registry.
  template <typename T>
  auto get_system() const -> std::shared_ptr<T> {
    // Check if the system is registered
    const std::type_index system_type{typeid(T)};
    auto system_result{systems_.find(system_type)};
    if (system_result == systems_.end()) {
      throw RegistryError();
    }

    // Return the system
    return std::static_pointer_cast<T>(system_result->second);
  }

  /// Update all the systems in the registry.
  ///
  /// @param delta_time - The time interval since the last time the function was called.
  inline void update(const double delta_time) const {
    for (const auto &[_, system] : systems_) {
      system->update(delta_time);
    }
  }

  /// Get the kinematic object for a game object.
  ///
  /// @param game_object_id - The game object ID.
  /// @throws RegistryError - If the game object is not registered or if the game object does not have a kinematic
  /// object.
  /// @return The kinematic object.
  [[nodiscard]] auto get_kinematic_object(GameObjectID game_object_id) const -> std::shared_ptr<KinematicObject>;

  /// Add a wall to the registry.
  ///
  /// @param wall - The wall to add to the registry.
  inline void add_wall(const Vec2d &wall) { walls_.emplace(wall); }

  /// Get the walls in the registry.
  ///
  /// @return The walls in the registry.
  [[nodiscard]] inline auto get_walls() const -> const std::unordered_set<Vec2d> & { return walls_; }

 private:
  /// The next game object ID to use.
  GameObjectID next_game_object_id_{0};

  /// The game objects and their components registered with the registry.
  std::unordered_map<GameObjectID, std::unordered_map<std::type_index, std::shared_ptr<ComponentBase>>> game_objects_;

  /// The systems registered with the registry.
  std::unordered_map<std::type_index, std::shared_ptr<SystemBase>> systems_;

  /// The kinematic objects registered with the registry.
  std::unordered_map<GameObjectID, std::shared_ptr<KinematicObject>> kinematic_objects_;

  /// The walls registered with the registry.
  std::unordered_set<Vec2d> walls_;
};
