// Ensure this file is only included once
#pragma once

// Std headers
#include <functional>
#include <queue>
#include <ranges>
#include <string>
#include <unordered_set>

// Local headers
#include "systems/physics.hpp"

/// Convert a Chipmunk2D shape to a game object ID.
///
/// @param shape - The Chipmunk2D shape to convert.
/// @return The game object ID.
inline auto cpShapeToGameObjectID(const cpShape* shape) -> GameObjectID {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  return static_cast<GameObjectID>(reinterpret_cast<uintptr_t>(cpShapeGetUserData(shape)));
}

/// Get the type name from a type info object.
///
/// @param info - The type info object.
/// @return The type name as a string.
auto type_name_from_info(const std::type_info& info) -> std::string;

/// Raised when an error occurs with the registry.
class RegistryError final : public std::runtime_error {
 public:
  /// Initialise the object.
  ///
  /// @param type - The type that caused the registry error.
  explicit RegistryError(const std::string& type)
      : std::runtime_error("The " + type + " is not registered with the registry.") {}

  /// Initialise the object.
  ///
  /// @param type - The type that caused the registry error.
  explicit RegistryError(const GameObjectID type) : RegistryError("game object ID `" + std::to_string(type) + "`") {}

  /// Initialise the object.
  ///
  /// @param type - The type that caused the registry error.
  /// @param game_object_id - The game object ID which caused the registry error.
  explicit RegistryError(const std::string& type, const GameObjectID game_object_id)
      : std::runtime_error("The component `" + type + "` for the game object ID `" + std::to_string(game_object_id) +
                           "` is not registered with the registry.") {}

  /// Create a RegistryError given a type.
  ///
  /// @tparam T - The type that caused the registry error.
  /// @return The registry error.
  template <typename T>
  static auto for_type() -> RegistryError {
    return RegistryError("`" + type_name_from_info(typeid(T)) + "`");
  }

  /// Create a RegistryError given a type and game object ID.
  ///
  /// @tparam T - The type that caused the registry error.
  /// @param game_object_id - The game object ID which caused the registry error.
  /// @return The registry error.
  template <typename T>
  static auto for_type(const GameObjectID game_object_id) -> RegistryError {
    return RegistryError(type_name_from_info(typeid(T)), game_object_id);
  }
};

/// Manages game objects, components, and systems that are registered.
class Registry {
 public:
  /// Initialise the object.
  explicit Registry();

  /// Create a new game object.
  ///
  /// @param game_object_type - The type of game object to create.
  /// @return The game object ID.
  auto create_game_object(GameObjectType game_object_type) -> GameObjectID;

  /// Check if a game object is registered or not.
  ///
  /// @param game_object_id - The game object ID.
  /// @return Whether the game object is registered or not.
  [[nodiscard]] auto has_game_object(GameObjectID game_object_id) const -> bool;

  /// Mark a game object for deletion after the next update step
  ///
  /// @param game_object_id - The ID of the game object to delete.
  /// @throws RegistryError if the game object does not exist or does not have a kinematic component.
  void mark_for_deletion(GameObjectID game_object_id);

  /// Delete a game object.
  ///
  /// @param game_object_id - The game object ID.
  /// @throws RegistryError - If the game object is not registered.
  void delete_game_object(GameObjectID game_object_id);

  /// Clear all game objects except those specified.
  ///
  /// @param game_object_ids_to_preserve - The game object IDs to preserve.
  void clear_game_objects(const std::unordered_set<GameObjectID>& game_object_ids_to_preserve = {});

  /// Add a component to a game object.
  ///
  /// @tparam Component - The type of component to add.
  /// @tparam Args - The types of arguments to pass to the component constructor.
  /// @param game_object_id - The game object ID of the game object to add the component to.
  /// @param args - The arguments to pass to the component constructor.
  /// @throws RegistryError - If the game object is not registered.
  template <typename Component, typename... Args>
  void add_component(const GameObjectID game_object_id, Args&&... args) {
    const auto component{std::make_shared<Component>(std::forward<Args>(args)...)};
    components_[type_id<Component>()][game_object_id] = component;
    if constexpr (std::is_base_of_v<Component, KinematicComponent>) {
      auto* const body{*component->body};
      auto* const shape{*component->shape};
      const auto game_object_type{get_game_object_type(game_object_id)};
      cpShapeSetCollisionType(shape, static_cast<cpCollisionType>(game_object_type));
      cpShapeSetFilter(shape, {CP_NO_GROUP, static_cast<cpBitmask>(game_object_type), CP_ALL_CATEGORIES});
      cpShapeSetUserData(shape, reinterpret_cast<void*>(static_cast<uintptr_t>(game_object_id)));
      cpShapeSetBody(shape, body);
      cpSpaceAddBody(*space_, body);
      cpSpaceAddShape(*space_, shape);
    }
  }

  /// Get a component from the registry.
  ///
  /// @tparam Component - The type of component to get.
  /// @param game_object_id - The game object ID.
  /// @throws RegistryError - If the game object is not registered or if the game object does not have the component.
  /// @return The component from the registry.
  template <typename Component>
  auto get_component(const GameObjectID game_object_id) const -> std::shared_ptr<Component> {
    const auto component_id{type_id<Component>()};
    if (!components_.contains(component_id)) {
      throw RegistryError::for_type<Component>(game_object_id);
    }
    const auto& component_map{components_.at(component_id)};
    if (!component_map.contains(game_object_id)) {
      throw RegistryError::for_type<Component>(game_object_id);
    }
    return std::static_pointer_cast<Component>(component_map.at(game_object_id));
  }

  /// Checks if a game object has a given component or not.
  ///
  /// @tparam Component - The type of component to check for.
  /// @param game_object_id - The game object ID.
  /// @return Whether the game object has the component or not.
  template <typename Component>
  [[nodiscard]] auto has_component(GameObjectID game_object_id) const -> bool {
    const auto component_id{type_id<Component>()};
    if (!components_.contains(component_id)) {
      return false;
    }
    return components_.at(component_id).contains(game_object_id);
  }

  /// Get all components of a game object.
  ///
  /// @param game_object_id - The game object ID.
  /// @throws RegistryError - If the game object is not registered.
  /// @return A range of all components of the game object.
  [[nodiscard]] auto get_game_object_components(const GameObjectID game_object_id) const {
    if (!has_game_object(game_object_id)) {
      throw RegistryError("game object", game_object_id);
    }
    auto components{components_ |
                    std::views::transform([game_object_id](auto const& pair) -> std::shared_ptr<ComponentBase> {
                      auto const& component_map{pair.second};
                      if (auto component_it{component_map.find(game_object_id)}; component_it != component_map.end()) {
                        return std::static_pointer_cast<ComponentBase>(component_it->second);
                      }
                      return nullptr;
                    }) |
                    std::views::filter([](auto const& ptr) { return ptr != nullptr; })};
    return components;
  }

  /// Get all game objects that have the required components.
  ///
  /// @tparam Component - The types of components to find.
  /// @return The game objects that have the required components.
  template <typename... Component>
  auto get_game_object_components() const {
    auto filtered{game_object_types_ | std::views::filter([this](auto const& pair) {
                    return (has_component<Component>(pair.first) && ...);
                  })};
    auto transformed{filtered | std::views::transform([this](auto const& pair) {
                       return std::make_pair(pair.first, std::tuple<std::shared_ptr<Component>...>{
                                                             get_component<Component>(pair.first)...});
                     })};

    return transformed;
  }

  /// Get the type of a game object.
  ///
  /// @param game_object_id - The game object ID.
  /// @throws RegistryError - If the game object is not registered.
  /// @return The type of the game object.
  [[nodiscard]] auto get_game_object_type(GameObjectID game_object_id) const -> GameObjectType;

  /// Get the game object IDs of a game object type.
  ///
  /// @param game_object_type - The game object type.
  /// @return The game object IDs of the game object type.
  [[nodiscard]] auto get_game_object_ids(GameObjectType game_object_type) const -> std::vector<GameObjectID>;

  /// Add a system to the registry.
  ///
  /// @tparam System - The type of system to add.
  /// @throws RegistryError - If the system is already registered.
  template <typename System>
  void add_system() {
    systems_[type_id<System>()] = std::make_shared<System>(this);
  }

  /// Get a system from the registry.
  ///
  /// @tparam System - The type of system to get.
  /// @throws RegistryError - If the system is not registered.
  /// @return The system from the registry.
  template <typename System>
  auto get_system() const -> std::shared_ptr<System> {
    const auto system_id{type_id<System>()};
    if (!systems_.contains(system_id)) {
      throw RegistryError::for_type<System>();
    }
    return std::static_pointer_cast<System>(systems_.at(system_id));
  }

  /// Update all the systems in the registry.
  ///
  /// @param delta_time - The time interval since the last time the function was called.
  void update(double delta_time);

  /// Get the Chipmunk2D space.
  ///
  /// @return The Chipmunk2D space.
  [[nodiscard]] auto get_space() const -> cpSpace* { return *space_; }

 private:
  /// The next component type ID to use.
  inline static int next_component_type_id_{0};

  /// The next system type ID to use.
  inline static int next_system_type_id_{0};

  /// Get the next type ID to use.
  ///
  /// @tparam T - The type to get the type ID for.
  /// @return The type ID.
  template <typename T>
  [[nodiscard]] static auto type_id() -> int {
    if constexpr (std::is_base_of_v<ComponentBase, T>) {
      static auto current_id{next_component_type_id_++};
      return current_id;
    } else if constexpr (std::is_base_of_v<SystemBase, T>) {
      static auto current_id{next_system_type_id_++};
      return current_id;
    } else {
      static_assert(sizeof(T) == 0, "type_id called on unknown type");
      return -1;
    }
  }

  /// The next game object ID to use.
  GameObjectID next_game_object_id_{0};

  /// The recycled game object IDs that can be reused.
  std::queue<GameObjectID> recycled_ids_;

  /// The components registered with the registry.
  std::unordered_map<int, std::unordered_map<GameObjectID, std::shared_ptr<ComponentBase>>> components_;

  /// The game object types registered with the registry.
  std::unordered_map<GameObjectID, GameObjectType> game_object_types_;

  /// The game object IDs registered with the registry.
  std::unordered_map<GameObjectType, std::vector<GameObjectID>> game_object_ids_;

  /// The game object IDs to delete.
  std::unordered_set<GameObjectID> objects_to_delete_;

  /// The systems registered with the registry.
  std::unordered_map<int, std::shared_ptr<SystemBase>> systems_;

  /// The Chipmunk2D space.
  ChipmunkHandle<cpSpace, cpSpaceFree> space_{cpSpaceNew()};
};
