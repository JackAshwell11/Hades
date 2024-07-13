// Ensure this file is only included once
#pragma once

// Std headers
#ifdef __GNUC__
#include <cxxabi.h>
#endif
#include <ranges>
#include <stdexcept>
#include <string>
#include <typeindex>

// Local headers
#include "steering.hpp"
#include "types.hpp"

// ----- CONSTANTS ------------------------------
// The percentage of velocity a game object will retain after a second.
constexpr double DAMPING = 0.0001;

// ----- FUNCTIONS ------------------------------
/// Calculate the screen position based on a grid position.
///
/// @param position - The position in the grid.
/// @throws std::invalid_argument - If the position is negative.
/// @return The screen position of the grid position.
inline auto grid_pos_to_pixel(const cpVect &position) -> cpVect {
  if (position.x < 0 || position.y < 0) {
    throw std::invalid_argument("The position cannot be negative.");
  }
  return position * SPRITE_SIZE + SPRITE_SIZE / 2;
}

#ifdef __GNUC__
/// Demangle the type name.
///
/// @param type - The type to demangle.
/// @return The demangled type name.
static auto demangle(const std::type_index &type) -> std::string {
  int status;
  const std::unique_ptr<char, void (*)(void *)> res{abi::__cxa_demangle(type.name(), nullptr, nullptr, &status),
                                                    std::free};
  return (status == 0) ? res.get() : type.name();
}
#else
/// Demangle the type name.
///
/// @param type - The type to demangle.
/// @return The demangled type name.
static auto demangle(const std::type_index &type) -> std::string { return std::string(type.name()).substr(7); }
#endif

// ----- EXCEPTIONS ------------------------------
/// Raised when an error occurs with the registry.
class RegistryError final : public std::runtime_error {
 public:
  /// Initialise the object.
  ///
  /// @param not_registered_type - The type of item that is not registered.
  /// @param value - The value that is not registered.
  /// @param extra - Any extra information to add to the error message.
  template <typename T>
  explicit RegistryError(const std::string &not_registered_type, const T &value,
                         const std::string &extra = "is not registered with the registry")
      : std::runtime_error("The " + not_registered_type + " `" + to_string(value) + "` " + extra + ".") {}

  /// Initialise the object.
  ///
  /// @param game_object_id - The game object ID that threw the error.
  /// @param type - The type of item that is not registered.
  explicit RegistryError(const GameObjectID game_object_id, const std::type_index &type)
      : std::runtime_error("The component `" + to_string(type) + "` for the game object ID `" +
                           to_string(game_object_id) + "` is not registered with the registry.") {}

 private:
  /// Convert a value to a string.
  ///
  /// @param value - The value to convert to a string.
  /// @return The value as a string.
  static auto to_string(const std::type_index &value) -> std::string { return demangle(value); }

  /// Convert a value to a string.
  ///
  /// @param value - The value to convert to a string.
  /// @return The value as a string.
  static auto to_string(const GameObjectID value) -> std::string { return std::to_string(value); }
};

// ----- CLASSES ------------------------------
/// Manages game objects, components, and systems that are registered.
class Registry {
 public:
  /// Initialise the object.
  Registry();

  /// Create a new game object.
  ///
  /// @param game_object_type - The type of game object to create.
  /// @param position - The position of the game object.
  /// @param components - The components to add to the game object.
  /// @return The game object ID.
  auto create_game_object(GameObjectType game_object_type, const cpVect &position,
                          const std::vector<std::shared_ptr<ComponentBase>> &&components) -> GameObjectID;

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
  [[nodiscard]] auto has_component(const GameObjectID game_object_id, const std::type_index &component_type) const
      -> bool {
    return game_objects_.contains(game_object_id) && game_objects_.at(game_object_id).contains(component_type);
  }

  /// Get a component from the registry.
  ///
  /// @tparam T - The type of component to get.
  /// @param game_object_id - The game object ID.
  /// @throws RegistryError - If the game object is not registered or if the game object does not have the component.
  /// @return The component from the registry.
  template <typename T>
  auto get_component(const GameObjectID game_object_id) const -> std::shared_ptr<T> {
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
  /// @return The game objects that have the required components.
  template <typename... Ts>
  auto find_components() const {
    // Use ranges::filter to filter out the game objects that have all the components then use ranges::transform to get
    // only the game object ID and the required components
    return game_objects_ | std::views::filter([this](const auto &game_object) {
             const auto &[game_object_id, game_object_components] = game_object;
             return (has_component(game_object_id, typeid(Ts)) && ...);
           }) |
           std::views::transform([](const auto &game_object) {
             const auto &[game_object_id, game_object_components] = game_object;
             return std::make_tuple(
                 game_object_id,
                 std::make_tuple(std::static_pointer_cast<Ts>(game_object_components.at(typeid(Ts)))...));
           });
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
      throw RegistryError("system", system_type, "is already registered with the registry");
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
    const auto system_result{systems_.find(system_type)};
    if (system_result == systems_.end()) {
      throw RegistryError("system", system_type);
    }

    // Return the system
    return std::static_pointer_cast<T>(system_result->second);
  }

  /// Update all the systems in the registry.
  ///
  /// @param delta_time - The time interval since the last time the function was called.
  void update(const double delta_time) const {
    for (const auto &[_, system] : systems_) {
      system->update(delta_time);
    }
  }

  /// Get the Chipmunk2D space.
  ///
  /// @return The Chipmunk2D space.
  [[nodiscard]] auto get_space() const -> cpSpace * { return *space_; }

  /// Add a callback to the registry to listen for events.
  ///
  /// @param event_type - The type of event to listen for.
  /// @param callback - The callback to add.
  void add_callback(const EventType event_type, const std::function<void(GameObjectID)> &callback) {
    callbacks_[event_type] = callback;
  }

  /// Notify all callbacks of an event.
  ///
  /// @param event_type - The type of event to notify callbacks of.
  /// @param game_object_id - The game object ID to pass to the callbacks.
  void notify_callbacks(const EventType event_type, const GameObjectID game_object_id) {
    if (callbacks_.contains(event_type)) {
      callbacks_[event_type](game_object_id);
    }
  }

 private:
  /// Create a Chipmunk2D collision handler to deal with bullet collisions.
  ///
  /// @param game_object_one - The first type of game object to check for collisions.
  /// @param game_object_two - The second type of game object to check for collisions.
  void createCollisionHandlerFunc(GameObjectType game_object_one, GameObjectType game_object_two);

  /// The next game object ID to use.
  GameObjectID next_game_object_id_{0};

  /// The game objects and their components registered with the registry.
  std::unordered_map<GameObjectID, std::unordered_map<std::type_index, std::shared_ptr<ComponentBase>>> game_objects_;

  /// The systems registered with the registry.
  std::unordered_map<std::type_index, std::shared_ptr<SystemBase>> systems_;

  /// The Chipmunk2D space.
  ChipmunkHandle<cpSpace, cpSpaceFree> space_{cpSpaceNew()};

  /// The Chipmunk2D shapes registered with the registry.
  std::unordered_map<cpShape *, GameObjectID> shapes_;

  /// The callbacks registered with the registry to listen for events.
  std::unordered_map<EventType, std::function<void(GameObjectID)>> callbacks_;
};
