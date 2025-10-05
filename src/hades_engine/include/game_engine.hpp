// Ensure this file is only included once
#pragma once

// Std headers
#include <filesystem>

// Local headers
#include "ecs/registry.hpp"
#include "game_state.hpp"
#include "input_handler.hpp"

/// Manages the interaction between Python and C++.
class GameEngine {
 public:
  /// Initialise the object.
  explicit GameEngine();

  /// Get the registry.
  ///
  /// @return The registry.
  [[nodiscard]] auto get_registry() -> std::shared_ptr<Registry> { return registry_; }

  /// Get the game state.
  ///
  /// @return The game state.
  [[nodiscard]] auto get_game_state() -> std::shared_ptr<GameState> { return game_state_; }

  /// Get the input handler.
  ///
  /// @return The input handler.
  [[nodiscard]] auto get_input_handler() -> std::shared_ptr<InputHandler> { return input_handler_; }

  /// Process update logic for the game engine.
  ///
  /// @param delta_time - The time interval since the last time the function was called.
  void on_update(double delta_time) const;

  /// Process fixed update logic for the game engine.
  ///
  /// @param delta_time - The time interval since the last time the function was called.
  void on_fixed_update(double delta_time) const;

  /// Enter the dungeon from the lobby.
  void enter_dungeon() const;

 private:
  /// Manages game objects, components, and systems that are registered.
  std::shared_ptr<Registry> registry_;

  /// Stores the state of the game.
  std::shared_ptr<GameState> game_state_;

  /// Handles input events such as key presses, releases, and mouse clicks.
  std::shared_ptr<InputHandler> input_handler_;
};
