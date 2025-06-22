// Ensure this file is only included once
#pragma once

// Custom headers
#include "game_object.hpp"

// Forward declarations
enum class StatusEffectType : std::uint8_t;

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
  ShopOpen,
  InventoryOpen,
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
