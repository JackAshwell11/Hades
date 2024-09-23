// Ensure this file is only included once
#pragma once

// Std headers
#include <optional>

// Local headers
#include "ecs/registry.hpp"
#include "ecs/types.hpp"
#include "generation/map.hpp"

/// Manages the interaction between Python and C++.
class GameEngine {
 public:
  /// Initialise the object.
  explicit GameEngine(int level, std::optional<unsigned int> seed = std::nullopt);

  /// Get the registry.
  ///
  /// @return The registry.
  [[nodiscard]] auto get_registry() -> std::shared_ptr<Registry> { return registry_; }

  /// Get the game objects.
  ///
  /// @return The game objects.
  [[nodiscard]] auto get_game_object() -> std::vector<std::pair<GameObjectID, GameObjectType>> { return game_object_ids_; }

 private:
  /// Manages game objects, components, and systems that are registered.
  std::shared_ptr<Registry> registry_;

  /// The game objects generated.
  std::vector<std::pair<GameObjectID, GameObjectType>> game_object_ids_;
};
