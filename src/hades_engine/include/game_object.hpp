// Ensure this file is only included once
#pragma once

// Std headers
#include <cstdint>

// Represents unique identifiers for game objects
using GameObjectID = int;

/// Stores the different types of game objects available.
enum class GameObjectType : std::uint16_t {
  // Only used in generation
  Empty = 0U << 0U,     // 0
  Obstacle = 1U << 0U,  // 1

  // Only used in game engine
  Bullet = 1U << 1U,  // 2

  // Used by game engine and generation
  Enemy = 1U << 2U,         // 4
  Floor = 1U << 3U,         // 8
  Player = 1U << 4U,        // 16
  Wall = 1U << 5U,          // 32
  Goal = 1U << 6U,          // 64
  HealthPotion = 1U << 7U,  // 128
  Chest = 1U << 8U,         // 256
  Shop = 1U << 9U,          // 512
};
