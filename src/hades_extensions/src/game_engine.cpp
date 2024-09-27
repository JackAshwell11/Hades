// Related header
#include "game_engine.hpp"

// External headers
#include <chipmunk/chipmunk_structs.h>

// Local headers
#include "ecs/systems/movements.hpp"
#include "ecs/systems/physics.hpp"

// ----- FUNCTIONS ------------------------------
void GameEngine::on_key_press(const int symbol, const int /*modifiers*/) const {
  const auto keyboard_movement{registry_.get_component<KeyboardMovement>(player_id_)};
  if (const auto symbol_ord{std::to_string(symbol)}; symbol_ord == "w") {
    keyboard_movement->moving_north = true;
  } else if (symbol_ord == "a") {
    keyboard_movement->moving_west = true;
  } else if (symbol_ord == "s") {
    keyboard_movement->moving_south = true;
  } else if (symbol_ord == "d") {
    keyboard_movement->moving_east = true;
  }
}

void GameEngine::on_key_release(const int symbol, const int /*modifiers*/) const {
  const auto keyboard_movement{registry_.get_component<KeyboardMovement>(player_id_)};
  if (const auto symbol_ord{std::to_string(symbol)}; symbol_ord == "w") {
    keyboard_movement->moving_north = false;
  } else if (symbol_ord == "a") {
    keyboard_movement->moving_west = false;
  } else if (symbol_ord == "s") {
    keyboard_movement->moving_south = false;
  } else if (symbol_ord == "d") {
    keyboard_movement->moving_east = false;
  }
}
