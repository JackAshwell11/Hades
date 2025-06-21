// Ensure this file is only included once
#pragma once

// Forward declarations
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
  explicit SystemBase(Registry *registry) : registry_(registry) {}

  /// The virtual destructor.
  virtual ~SystemBase() = default;

  /// The copy constructor.
  SystemBase(const SystemBase &) = default;

  /// The move constructor.
  SystemBase(SystemBase &&) = default;

  /// Get the registry that manages the game objects, components, and systems.
  ///
  /// @return The registry that manages the game objects, components, and systems.
  [[nodiscard]] auto get_registry() const -> Registry * { return registry_; }

  /// Process update logic for a system.
  virtual void update(double /*delta_time*/) const {}

 private:
  /// The registry that manages the game objects, components, and systems.
  Registry *registry_;
};
