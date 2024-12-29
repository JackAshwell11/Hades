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

  /// Get the player's game object ID.
  ///
  /// @return The player's game object ID.
  [[nodiscard]] auto get_player_id() const -> GameObjectID { return player_id_; }

  /// Create the game objects from the generator.
  ///
  /// @details If this is called twice, the game objects will be duplicated.
  void create_game_objects();

  /// Generate an enemy.
  ///
  /// @param delta_time - The time interval since the last time the function was called.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
  void generate_enemy(double delta_time = 1.0 / 60.0);

 private:
  /// Get the components for a game object type.
  ///
  /// @param game_object_type - The game object type.
  /// @return The components for the game object type.
  [[nodiscard]] auto get_game_object_components(GameObjectType game_object_type)
      -> std::vector<std::shared_ptr<ComponentBase>>;

  /// The current level of the game.
  int level_;

  /// The random generator for the game.
  std::mt19937 random_generator_;

  /// The normal distribution to determine the level of the game objects.
  std::normal_distribution<> level_distribution_;

  /// Manages game objects, components, and systems that are registered.
  std::shared_ptr<Registry> registry_;

  /// The map generator responsible for the generation.
  MapGenerator generator_;

  /// The player's game object ID.
  GameObjectID player_id_;
};
