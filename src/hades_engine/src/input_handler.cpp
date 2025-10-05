// Related header
#include "input_handler.hpp"

// Local headers
#include "ecs/registry.hpp"
#include "ecs/systems/attacks.hpp"
#include "ecs/systems/inventory.hpp"
#include "ecs/systems/movements.hpp"
#include "events.hpp"
#include "game_state.hpp"

InputHandler::InputHandler(const std::shared_ptr<Registry>& registry, const std::shared_ptr<GameState>& game_state)
    : registry_(registry), game_state_(game_state) {}

void InputHandler::on_key_press(const int symbol, const int /*modifiers*/) const {
  const auto player_movement{registry_->get_component<KeyboardMovement>(game_state_->get_player_id())};
  switch (symbol) {
    case KEY_W:
      player_movement->moving_north = true;
      break;
    case KEY_A:
      player_movement->moving_west = true;
      break;
    case KEY_S:
      player_movement->moving_south = true;
      break;
    case KEY_D:
      player_movement->moving_east = true;
      break;
    default:
      break;
  }
}

void InputHandler::on_key_release(const int symbol, const int /*modifiers*/) const {
  const auto player_movement{registry_->get_component<KeyboardMovement>(game_state_->get_player_id())};
  switch (symbol) {
    case KEY_W:
      player_movement->moving_north = false;
      break;
    case KEY_A:
      player_movement->moving_west = false;
      break;
    case KEY_S:
      player_movement->moving_south = false;
      break;
    case KEY_D:
      player_movement->moving_east = false;
      break;
    case KEY_C:
      registry_->get_system<InventorySystem>()->add_item_to_inventory(game_state_->get_player_id(),
                                                                      game_state_->get_nearest_item());
      break;
    case KEY_E:
      registry_->get_system<InventorySystem>()->use_item(game_state_->get_player_id(), game_state_->get_nearest_item());
      break;
    case KEY_Z:
      registry_->get_system<AttackSystem>()->previous_ranged_attack(game_state_->get_player_id());
      break;
    case KEY_X:
      registry_->get_system<AttackSystem>()->next_ranged_attack(game_state_->get_player_id());
      break;
    case KEY_I:
      notify<EventType::InventoryOpen>();
    case KEY_R:
      notify<EventType::ShopOpen>();
    default:
      break;
  }
}

auto InputHandler::on_mouse_press(const double /*x*/, const double /*y*/, const int button,
                                  const int /*modifiers*/) const -> bool {
  if (button == MOUSE_BUTTON_LEFT) {
    return registry_->get_system<AttackSystem>()->do_attack(game_state_->get_player_id());
  }
  return false;
}

void InputHandler::on_mouse_motion(const double x, const double y, const double /*delta_x*/,
                                   const double /*delta_y*/) const {
  const auto [width, height]{game_state_->get_window_size()};
  const auto kinematic_component{registry_->get_component<KinematicComponent>(game_state_->get_player_id())};
  const auto [player_x, player_y]{cpBodyGetPosition(*kinematic_component->body)};
  kinematic_component->rotation = std::atan2(y + player_y - (static_cast<double>(height) / 2) - player_y,
                                             x + player_x - (static_cast<double>(width) / 2) - player_x);
}
