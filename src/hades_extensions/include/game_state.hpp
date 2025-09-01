// Ensure this file is only included once
#pragma once

// Std headers
#include <memory>
#include <random>

// Local headers
#include "game_object.hpp"

// Forward declarations
class Registry;
struct Grid;
struct cpVect;

/// Stores the different types of levels in the game.
enum class LevelType : std::uint8_t {
  None,
  Lobby,
  FirstDungeon,
  SecondDungeon,
  Boss,
};

/// Stores the state of the game.
class GameState {
 public:
  /// Initialise the object.
  ///
  /// /// @param registry - The registry that manages game objects, components, and systems.
  explicit GameState(const std::shared_ptr<Registry>& registry);

  /// Get the player's game object ID.
  ///
  /// @return The player's game object ID.
  [[nodiscard]] auto get_player_id() const -> GameObjectID;

  /// Get the nearest item to the player.
  ///
  /// @return The nearest item to the player.
  [[nodiscard]] auto get_nearest_item() const -> GameObjectID;

  /// Set the nearest item to the player.
  ///
  /// @param item_id - The ID of the item to set as the nearest item.
  void set_nearest_item(GameObjectID item_id);

  /// Get the level of the current dungeon.
  ///
  /// @return The level of the current dungeon.
  [[nodiscard]] auto get_dungeon_level() const -> LevelType;

  /// Get the level of the game objects.
  ///
  /// @return The level of the game objects.
  [[nodiscard]] auto get_game_level() const -> int;

  /// Check if the current level is a lobby level.
  ///
  /// @return True if the current level is a lobby level, false otherwise.
  [[nodiscard]] auto is_lobby() const -> bool;

  /// Checks if the current level is a boss level.
  ///
  /// @return True if the current level is a boss level, false otherwise.
  [[nodiscard]] auto is_boss() const -> bool;

  /// Get the enemy generation timer.
  ///
  /// @return The enemy generation timer.
  [[nodiscard]] auto get_enemy_generation_timer() const -> double;

  /// Sets the enemy generation timer.
  ///
  /// @param value - The value to set the enemy generation timer to.
  void set_enemy_generation_timer(double value);

  /// Checks if the player is touching the specified game object type.
  ///
  /// @param game_object_type - The type of game object to check for.
  /// @return True if the player is touching the game object type, false otherwise.
  [[nodiscard]] auto is_player_touching_type(GameObjectType game_object_type) const -> bool;

  /// Set the seed for the random generator.
  ///
  /// @param seed - The seed to set for the random generator.
  void set_seed(const std::string& seed);

  /// Initialise the dungeon run.
  void initialise_dungeon_run();

  /// Reset the game engine with a new level.
  ///
  /// @param level_type - The type of level to reset to.
  void reset_level(LevelType level_type);

  /// Generate an enemy.
  void generate_enemy();

 private:
  /// Create the game objects from the generator.
  ///
  /// @details If this is called twice, the game objects will be duplicated.
  /// @param grid - The grid to create the game objects from.
  /// @param store_floor_positions - Whether to store the positions of the floor game objects or not.
  void create_game_objects(const Grid& grid, bool store_floor_positions = true);

  /// The registry that manages game objects, components, and systems.
  std::shared_ptr<Registry> registry_;

  /// Stores the state of the game.
  struct {
    /// Stores state which should persist across the entire game.
    struct {
      /// The player's game object ID.
      GameObjectID player_id{-1};
    } game;

    /// Store state which should persist across the entire level.
    struct {
      /// The random generator for the current level.
      std::mt19937 random_generator{std::random_device{}()};

      /// The level of the game objects.
      int game_level{-1};

      /// The normal distribution to determine the level of the game objects.
      std::normal_distribution<> level_distribution;
    } dungeon_run;

    /// Stores state about the current dungeon.
    struct {
      /// The positions of the floor game objects in the game.
      std::vector<cpVect> floor_positions;

      /// The current dungeon level.
      LevelType dungeon_level{LevelType::None};

      /// The nearest item to the player.
      GameObjectID nearest_item{-1};

      /// The timer for enemy generation.
      double enemy_generation_timer{0.0};
    } current_level;
  } game_state_;
};
