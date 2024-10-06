// Ensure this file is only included once
#pragma once

// Std headers
#include <optional>

// Local headers
#include "ecs/registry.hpp"
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

  /// Create the game objects from the generator.
  /// @details If this is called twice, the game objects will be duplicated.
  void create_game_objects();

 private:
  /// Manages game objects, components, and systems that are registered.
  std::shared_ptr<Registry> registry_;

  /// The map generator responsible for the generation.
  MapGenerator generator_;
};
