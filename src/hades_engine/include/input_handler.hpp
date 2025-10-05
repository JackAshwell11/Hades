// Ensure this file is only included once
#pragma once

// Std headers
#include <memory>

// Forward declarations
class Registry;
class GameState;

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

/// The I key code.
constexpr int KEY_I{105};

/// The R key code.
constexpr int KEY_R{114};

/// The left mouse button code.
constexpr int MOUSE_BUTTON_LEFT{1};

/// Handles input events such as key presses, releases, and mouse clicks.
class InputHandler {
 public:
  /// Initialise the object.
  ///
  /// @param registry - The registry which manages the game objects, components, and systems.
  /// @param game_state - The storage for the state of the game.
  explicit InputHandler(const std::shared_ptr<Registry>& registry, const std::shared_ptr<GameState>& game_state);

  /// Process key press functionality.
  ///
  /// @param symbol - The key that was hit.
  /// @param modifiers - Bitwise AND of all modifiers (shift, ctrl, num lock) pressed during this event.
  void on_key_press(int symbol, int modifiers) const;

  /// Process key release functionality.
  ///
  /// @param symbol - The key that was released.
  /// @param modifiers - Bitwise AND of all modifiers (shift, ctrl, num lock) pressed during this event.
  void on_key_release(int symbol, int modifiers) const;

  /// Process mouse press functionality.
  ///
  /// @param x - The x position of the mouse.
  /// @param y - The y position of the mouse.
  /// @param button - The button that was pressed.
  /// @param modifiers - Bitwise AND of all modifiers (shift, ctrl, num lock) pressed during this event.
  /// @return Whether the attack was successful or not.
  [[nodiscard]] auto on_mouse_press(double x, double y, int button, int modifiers) const -> bool;

  /// Process mouse motion functionality.
  ///
  /// @param x - The x position of the mouse.
  /// @param y - The y position of the mouse.
  /// @param delta_x - The change in x position of the mouse.
  /// @param delta_y - The change in y position of the mouse.
  void on_mouse_motion(double x, double y, double delta_x, double delta_y) const;

 private:
  /// The registry that manages game objects, components, and systems.
  std::shared_ptr<Registry> registry_;

  /// Stores the state of the game.
  std::shared_ptr<GameState> game_state_;
};
