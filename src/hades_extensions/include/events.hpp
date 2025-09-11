// Ensure this file is only included once
#pragma once

// Std headers
#include <any>
#include <functional>
#include <string>

// Local headers
#include "game_object.hpp"

// Forward declarations
enum class EffectType : std::uint8_t;
struct SaveFileInfo;

/// Stores the different types of events that can occur.
enum class EventType : std::uint8_t {
  GameObjectCreation,
  GameObjectDeath,
  PositionChanged,
  InventoryUpdate,
  SpriteRemoval,
  StatusEffectUpdate,
  MoneyUpdate,
  AttackCooldownUpdate,
  RangedAttackSwitch,
  ShopItemLoaded,
  ShopItemPurchased,
  ShopOpen,
  InventoryOpen,
  GameOptionsOpen,
  SaveFilesUpdated,
  GameOpen,
  HealthChanged,
  ArmourChanged,
};

/// A helper struct to provide the argument types for each event type.
template <EventType>
struct EventTraits;

/// Provides the argument types for the GameObjectCreation event.
template <>
struct EventTraits<EventType::GameObjectCreation> {
  using EventArgs = std::tuple<GameObjectID, GameObjectType>;
};

/// Provides the argument types for the GameObjectDeath event.
template <>
struct EventTraits<EventType::GameObjectDeath> {
  using EventArgs = std::tuple<GameObjectID>;
};

/// Provides the argument types for the PositionChanged event.
template <>
struct EventTraits<EventType::PositionChanged> {
  using EventArgs = std::tuple<GameObjectID, std::pair<double, double>>;
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
  using EventArgs = std::tuple<std::unordered_map<EffectType, double>>;
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

/// Provides the argument types for the ShopOpen event.
template <>
struct EventTraits<EventType::ShopOpen> {
  using EventArgs = std::tuple<>;
};

/// Provides the argument types for the InventoryOpen event.
template <>
struct EventTraits<EventType::InventoryOpen> {
  using EventArgs = std::tuple<>;
};

/// Provides the argument types for the GameOptionsOpen event.
template <>
struct EventTraits<EventType::GameOptionsOpen> {
  using EventArgs = std::tuple<>;
};

/// Provides the argument types for the SaveFilesUpdated event.
template <>
struct EventTraits<EventType::SaveFilesUpdated> {
  using EventArgs = std::tuple<std::vector<SaveFileInfo>>;
};

/// Provides the argument types for the GameOpen event.
template <>
struct EventTraits<EventType::GameOpen> {
  using EventArgs = std::tuple<>;
};

/// Provides the argument types for the HealthChanged event.
template <>
struct EventTraits<EventType::HealthChanged> {
  using EventArgs = std::tuple<GameObjectID, double>;
};

/// Provides the argument types for the ArmourChanged event.
template <>
struct EventTraits<EventType::ArmourChanged> {
  using EventArgs = std::tuple<GameObjectID, double>;
};

/// Get the listeners map for the events.
///
/// @return A reference to the map of listeners for each event type.
inline auto get_listeners() -> auto& {
  static std::unordered_map<EventType, std::vector<std::function<void(std::any)>>> listeners;
  return listeners;
}

/// Clear all listeners for all events.
inline void clear_listeners() { get_listeners().clear(); }

/// Add a callback to the listeners for a specific event type.
///
/// @tparam E - The type of event to listen for.
/// @tparam Func - The callback functions' signature
/// @param callback - The callback to add.
template <EventType E, typename Func>
void add_callback(Func&& callback) {
  get_listeners()[E].emplace_back([callback = std::forward<Func>(callback)](std::any args) {
    std::apply(callback, std::any_cast<typename EventTraits<E>::EventArgs>(args));
  });
}

/// Notify all callbacks of an event.
///
/// @tparam E - The type of event to notify callbacks of.
/// @tparam Args - The types of the arguments to pass to the callbacks.
/// @param args - The arguments to pass to the callbacks.
template <EventType E, typename... Args>
void notify(Args&&... args) {
  using ExpectedArgs = EventTraits<E>::EventArgs;
  static_assert(std::is_same_v<std::tuple<std::decay_t<Args>...>, ExpectedArgs>);
  if (!get_listeners().contains(E)) {
    return;
  }
  const ExpectedArgs tuple_args{std::forward<Args>(args)...};
  for (const auto& callback : get_listeners().at(E)) {
    callback(tuple_args);
  }
}
