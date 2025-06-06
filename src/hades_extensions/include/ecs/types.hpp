// Ensure this file is only included once
#pragma once

// Std headers
#include <functional>
#include <memory>
#include <string>

// Avoid having to include headers for this
class Registry;

// Represents unique identifiers for game objects
using GameObjectID = int;

/// Stores the different types of game objects available.
enum class GameObjectType : std::uint8_t {
  Bullet = 1U << 0U,        // 1
  Enemy = 1U << 1U,         // 2
  Floor = 1U << 2U,         // 4
  Player = 1U << 3U,        // 8
  Wall = 1U << 4U,          // 16
  Goal = 1U << 5U,          // 32
  HealthPotion = 1U << 6U,  // 64
  Chest = 1U << 7U,         // 128
};

/// Stores the different types of events that can occur.
enum class EventType : std::uint8_t {
  GameObjectCreation,
  GameObjectDeath,
  InventoryUpdate,
  SpriteRemoval,
  StatusEffectUpdate,
  MoneyUpdate,
  AttackCooldownUpdate,
  RangedAttackSwitch,
  ShopItemLoaded,
  ShopItemPurchased,
};

/// Stores the different types of status effects available.
enum class StatusEffectType : std::uint8_t {
  Regeneration,
  Poison,
};

/// A helper struct to provide the argument types for each event type.
template <EventType>
struct EventTraits;

/// Provides the argument types for the GameObjectCreation event.
template <>
struct EventTraits<EventType::GameObjectCreation> {
  using EventArgs = std::tuple<GameObjectID>;
};

/// Provides the argument types for the GameObjectDeath event.
template <>
struct EventTraits<EventType::GameObjectDeath> {
  using EventArgs = std::tuple<GameObjectID>;
};

/// Provides the argument types for the InventoryUpdate event.
template <>
struct EventTraits<EventType::InventoryUpdate> {
  using EventArgs = std::tuple<std::vector<GameObjectID>>;
};

/// Provides the argument types for the SpriteRemoval event.
template <>
struct EventTraits<EventType::SpriteRemoval> {
  using EventArgs = std::tuple<GameObjectID>;
};

/// Provides the argument types for the StatusEffectUpdate event.
template <>
struct EventTraits<EventType::StatusEffectUpdate> {
  using EventArgs = std::tuple<std::unordered_map<StatusEffectType, double>>;
};

/// Provides the argument types for the MoneyUpdate event.
template <>
struct EventTraits<EventType::MoneyUpdate> {
  using EventArgs = std::tuple<int>;
};

/// Provides the argument types for the AttackCooldownUpdate event.
template <>
struct EventTraits<EventType::AttackCooldownUpdate> {
  using EventArgs = std::tuple<GameObjectID, double, double, double>;
};

/// Provides the argument types for the RangedAttackSwitch event.
template <>
struct EventTraits<EventType::RangedAttackSwitch> {
  using EventArgs = std::tuple<int>;
};

/// Provides the argument types for the ShopItemLoaded event.
template <>
struct EventTraits<EventType::ShopItemLoaded> {
  using EventArgs = std::tuple<int, std::tuple<std::string, std::string, std::string>, int>;
};

/// Provides the argument types for the ShopItemPurchased event.
template <>
struct EventTraits<EventType::ShopItemPurchased> {
  using EventArgs = std::tuple<int, int>;
};

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
  explicit ChipmunkHandle(T *obj) : obj_(obj, Destructor) {}

  /// Destroy the object.
  ~ChipmunkHandle() = default;

  /// The copy constructor.
  ChipmunkHandle(const ChipmunkHandle &) = delete;

  /// The copy assignment operator.
  auto operator=(const ChipmunkHandle &) -> ChipmunkHandle & = delete;

  /// The move constructor.
  ChipmunkHandle(ChipmunkHandle &&other) noexcept : obj_(std::move(other.obj_)) {}

  /// The move assignment operator.
  auto operator=(ChipmunkHandle &&other) noexcept -> ChipmunkHandle & {
    if (this != &other) {
      obj_ = std::move(other.obj_);
    }
    return *this;
  }

  /// The dereference operator.
  auto operator*() const -> T * { return obj_.get(); }

  /// The arrow operator.
  auto operator->() const -> T * { return obj_.get(); }

 private:
  /// The Chipmunk2D object.
  std::unique_ptr<T, void (*)(T *)> obj_;
};
