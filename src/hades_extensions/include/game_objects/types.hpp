// Ensure this file is only included once
#pragma once

// Std headers
#include <memory>

// ----- ENUMS ------------------------------
/// Stores the different types of game objects available.
enum class GameObjectType : std::uint8_t {
  Bullet = 1U << 0U,  // 1
  Enemy = 1U << 1U,   // 2
  Floor = 1U << 2U,   // 4
  Player = 1U << 3U,  // 8
  Potion = 1U << 4U,  // 16
  Wall = 1U << 5U,    // 32
};

// ----- BASE TYPES ------------------------------
// Add a forward declaration for the registry class
class Registry;

/// The base class for all components.
struct ComponentBase {
  /// The copy assignment operator.
  auto operator=(const ComponentBase &) -> ComponentBase & = default;

  /// The move assignment operator.
  auto operator=(ComponentBase &&) -> ComponentBase & = default;

  /// The default constructor.
  ComponentBase() = default;

  /// The virtual destructor.
  virtual ~ComponentBase() = default;

  /// The copy constructor.
  ComponentBase(const ComponentBase &) = default;

  /// The move constructor.
  ComponentBase(ComponentBase &&) = default;

  /// Checks if the component can have an indicator bar or not.
  ///
  /// @return Whether the component can have an indicator bar or not.
  [[nodiscard]] virtual auto has_indicator_bar() const -> bool { return false; }
};

/// The base class for all systems.
class SystemBase {
 public:
  /// The copy assignment operator.
  auto operator=(const SystemBase &) -> SystemBase & = default;

  /// The move assignment operator.
  auto operator=(SystemBase &&) -> SystemBase & = default;

  /// Initialise the object.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  explicit SystemBase(Registry *registry) : _registry(registry) {}

  /// The virtual destructor.
  virtual ~SystemBase() = default;

  /// The copy constructor.
  SystemBase(const SystemBase &) = default;

  /// The move constructor.
  SystemBase(SystemBase &&) = default;

  /// Get the registry that manages the game objects, components, and systems.
  ///
  /// @return The registry that manages the game objects, components, and systems.
  [[nodiscard]] auto get_registry() const -> Registry * { return _registry; }

  /// Process update logic for a system.
  virtual void update(double /*delta_time*/) const {}

 private:
  /// The registry that manages the game objects, components, and systems.
  Registry *_registry;
};

// ----- RAII TYPES ------------------------------
/// Allows for the RAII management of a Chipmunk2D object.
///
/// @tparam T - The type of Chipmunk2D object to manage.
/// @tparam Destructor - The destructor function for the Chipmunk2D object.
template <typename T, void (*Destructor)(T *)>
class ChipmunkHandle {
 public:
  /// Initialise the object.
  ///
  /// @param obj - The Chipmunk2D object.
  explicit ChipmunkHandle(T *obj) : _obj(obj, Destructor) {}

  /// Destroy the object.
  ~ChipmunkHandle() = default;

  /// The copy constructor.
  ChipmunkHandle(const ChipmunkHandle &) = delete;

  /// The copy assignment operator.
  auto operator=(const ChipmunkHandle &) -> ChipmunkHandle & = delete;

  /// The move constructor.
  ChipmunkHandle(ChipmunkHandle &&other) noexcept : _obj(std::move(other._obj)) {}

  /// The move assignment operator.
  auto operator=(ChipmunkHandle &&other) noexcept -> ChipmunkHandle & {
    if (this != &other) {
      _obj = std::move(other._obj);
    }
    return *this;
  }

  /// The dereference operator.
  auto operator*() const -> T * { return _obj.get(); }

  /// The arrow operator.
  auto operator->() const -> T * { return _obj.get(); }

 private:
  /// The Chipmunk2D object.
  std::unique_ptr<T, void (*)(T *)> _obj;
};
