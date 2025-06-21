// Ensure this file is only included once
#pragma once

// Local headers
#include "ecs/registry.hpp"
#include "generation/map.hpp"

/// The W key code.
constexpr int KEY_W{119};

/// The A key code.
constexpr int KEY_A{97};

/// The S key code.
constexpr int KEY_S{115};

/// The D key code.
constexpr int KEY_D{100};

/// The C key code.
constexpr int KEY_C{99};

/// The E key code.
constexpr int KEY_E{101};

/// The Z key code.
constexpr int KEY_Z{122};

/// The X key code.
constexpr int KEY_X{120};

/// The left mouse button code.
constexpr int MOUSE_BUTTON_LEFT{1};

/// Stores the different types of levels in the game.
enum class LevelType : std::uint8_t { Normal, Boss, Lobby };

/// Manages the interaction between Python and C++.
class GameEngine {
 public:
  /// Initialise the object.
  explicit GameEngine();

  /// Get the registry.
  ///
  /// @return The registry.
  [[nodiscard]] auto get_registry() -> Registry & { return registry_; }

  /// Get the player's game object ID.
  ///
  /// @return The player's game object ID.
  [[nodiscard]] auto get_player_id() const -> GameObjectID { return game_state_.dungeon_run.player_id; }

  /// Get the nearest item to the player.
  ///
  /// @return The nearest item to the player.
  [[nodiscard]] auto get_nearest_item() const -> GameObjectID { return game_state_.current_level.nearest_item; }

  /// Get the level of the current dungeon.
  ///
  /// @return The level of the current dungeon.
  [[nodiscard]] auto get_dungeon_level() const -> int { return game_state_.dungeon_run.dungeon_level; }

  /// Get the level of the game objects.
  ///
  /// @return The level of the game objects.
  [[nodiscard]] auto get_game_level() const -> int { return game_state_.dungeon_run.game_level; }

  /// Checks if the player is touching the specified game object type.
  ///
  /// @param game_object_type - The type of game object to check for.
  /// @return True if the player is touching the game object type, false otherwise.
  [[nodiscard]] auto is_player_touching_type(GameObjectType game_object_type) const -> bool;

  /// Set the seed for the random generator.
  ///
  /// @param seed - The seed to set for the random generator.
  void set_seed(unsigned int seed);

  /// Reset the game engine with a new level.
  ///
  /// @param level_type - The type of level to reset to.
  void reset_level(LevelType level_type);

  /// Set up the shop offerings.
  ///
  /// @param stream - The input stream containing the JSON data for the shop offerings.
  /// @throws std::runtime_error if there was an error parsing the JSON file or the offering type is unknown.
  void setup_shop(std::istream &stream) const;

  /// Process update logic for the game engine.
  ///
  /// @param delta_time - The time interval since the last time the function was called.
  void on_update(double delta_time);

  /// Process fixed update logic for the game engine.
  ///
  /// @param delta_time - The time interval since the last time the function was called.
  void on_fixed_update(double delta_time);

  /// Process key press functionality.
  ///
  /// @param symbol - The key that was hit.
  /// @param modifiers - Bitwise AND of all modifiers (shift, ctrl, num lock) pressed during this event.
  void on_key_press(int symbol, int modifiers) const;

  /// Process key release functionality.
  ///
  /// @param symbol - The key that was released.
  /// @param modifiers - Bitwise AND of all modifiers (shift, ctrl, num lock) pressed during this event.
  void on_key_release(int symbol, int modifiers);

  /// Process mouse press functionality.
  ///
  /// @param x - The x position of the mouse.
  /// @param y - The y position of the mouse.
  /// @param button - The button that was pressed.
  /// @param modifiers - Bitwise AND of all modifiers (shift, ctrl, num lock) pressed during this event.
  /// @return Whether the attack was successful or not.
  [[nodiscard]] auto on_mouse_press(double x, double y, int button, int modifiers) const -> bool;

  /// Use an item on a target game object.
  ///
  /// @param target_id - The game object ID of the target.
  /// @param item_id - The game object ID of the item to use.
  void use_item(GameObjectID target_id, GameObjectID item_id);

 private:
  /// Create the player game object.
  void create_player();

  /// Create the game objects from the generator.
  ///
  /// @details If this is called twice, the game objects will be duplicated.
  /// @param grid - The grid to create the game objects from.
  /// @param store_floor_positions - Whether to store the positions of the floor game objects or not.
  void create_game_objects(const Grid &grid, bool store_floor_positions = true);

  /// Generate an enemy.
  void generate_enemy();

  /// Get the components for a game object type.
  ///
  /// @param game_object_type - The game object type.
  /// @return The components for the game object type.
  [[nodiscard]] auto get_game_object_components(GameObjectType game_object_type)
      -> std::vector<std::shared_ptr<ComponentBase>>;

  /// Manages game objects, components, and systems that are registered.
  Registry registry_;

  struct GameState {
    /// Stores state which should persist across the entire game.
    struct {
    } game;

    /// Store state which should persist across the entire level.
    struct {
      /// The player's game object ID.
      GameObjectID player_id{-1};

      /// The random generator for the current level.
      std::mt19937 random_generator{std::random_device{}()};

      /// The level of the game objects.
      int game_level{-1};

      /// The normal distribution to determine the level of the game objects.
      std::normal_distribution<> level_distribution;

      /// The current dungeon level.
      int dungeon_level{-1};
    } dungeon_run;

    /// Stores state about the current dungeon.
    struct {
      /// The positions of the floor game objects in the game.
      std::vector<cpVect> floor_positions;

      /// Whether the player is in the lobby or not.
      bool is_lobby{false};

      /// Whether the current level is a boss level or not.
      bool is_boss_level{false};

      /// The nearest item to the player.
      GameObjectID nearest_item{-1};

      /// The timer for enemy generation.
      double enemy_generation_timer{0.0};
    } current_level;
  } game_state_;
};
