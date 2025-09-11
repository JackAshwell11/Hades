// Local headers
#include "ecs/systems/attacks.hpp"
#include "ecs/systems/effects.hpp"
#include "ecs/systems/inventory.hpp"
#include "ecs/systems/movements.hpp"
#include "events.hpp"
#include "factories.hpp"
#include "game_helpers.hpp"
#include "input_handler.hpp"
#include "macros.hpp"

/// Implements the fixture for the input_handler.hpp tests.
class InputHandlerFixture : public testing::Test {  // NOLINT
 protected:
  /// The registry which manages the game objects, components, and systems.
  std::shared_ptr<Registry> registry;

  /// The game state which stores the state of the game.
  std::shared_ptr<GameState> game_state;

  /// The input handler which handles input events.
  std::shared_ptr<InputHandler> input_handler;

  /// Set up shared logic for the tests.
  static void SetUpTestSuite() {
    load_hitbox(GameObjectType::Player, {{0.0, 1.0}, {1.0, 2.0}, {2.0, 0.0}});
    load_hitbox(GameObjectType::Enemy, {{0.0, 1.0}, {1.0, 2.0}, {2.0, 0.0}});
  }

  /// Set up the fixture for the tests.
  void SetUp() override {
    registry = std::make_shared<Registry>();
    game_state = std::make_shared<GameState>(registry);
    input_handler = std::make_shared<InputHandler>(registry, game_state);
    registry->add_system<AttackSystem>();
    registry->add_system<EffectSystem>();
    registry->add_system<InventorySystem>();
    registry->add_system<PhysicsSystem>();
    game_state->reset_level(LevelType::FirstDungeon);
  }

  /// Tear down the fixture after the tests.
  void TearDown() override { clear_listeners(); }
};

/// Test that the input handler processes a "W" key press correctly.
TEST_F(InputHandlerFixture, TestInputHandlerOnKeyPressW) {
  input_handler->on_key_press(KEY_W, 0);
  const auto player_movement{registry->get_component<KeyboardMovement>(0)};
  ASSERT_TRUE(player_movement->moving_north);
}

/// Test that the input handler processes an "A" key press correctly.
TEST_F(InputHandlerFixture, TestInputHandlerOnKeyPressA) {
  input_handler->on_key_press(KEY_A, 0);
  const auto player_movement{registry->get_component<KeyboardMovement>(0)};
  ASSERT_TRUE(player_movement->moving_west);
}

/// Test that the input handler processes an "S" key press correctly.
TEST_F(InputHandlerFixture, TestInputHandlerOnKeyPressS) {
  input_handler->on_key_press(KEY_S, 0);
  const auto player_movement{registry->get_component<KeyboardMovement>(0)};
  ASSERT_TRUE(player_movement->moving_south);
}

/// Test that the input handler processes a "D" key press correctly.
TEST_F(InputHandlerFixture, TestInputHandlerOnKeyPressD) {
  input_handler->on_key_press(KEY_D, 0);
  const auto player_movement{registry->get_component<KeyboardMovement>(0)};
  ASSERT_TRUE(player_movement->moving_east);
}

/// Test that the input handler processes an unknown key press correctly.
TEST_F(InputHandlerFixture, TestInputHandlerOnKeyPressUnknown) { ASSERT_NO_THROW(input_handler->on_key_press(0, 0)); }

/// Test that the input handler processes a "W" key release correctly.
TEST_F(InputHandlerFixture, TestInputHandlerOnKeyReleaseW) {
  input_handler->on_key_press(KEY_W, 0);
  input_handler->on_key_release(KEY_W, 0);
  const auto player_movement{registry->get_component<KeyboardMovement>(0)};
  ASSERT_FALSE(player_movement->moving_north);
}

/// Test that the input handler processes an "A" key release correctly.
TEST_F(InputHandlerFixture, TestInputHandlerOnKeyReleaseA) {
  input_handler->on_key_press(KEY_A, 0);
  input_handler->on_key_release(KEY_A, 0);
  const auto player_movement{registry->get_component<KeyboardMovement>(0)};
  ASSERT_FALSE(player_movement->moving_west);
}

/// Test that the input handler processes an "S" key release correctly.
TEST_F(InputHandlerFixture, TestInputHandlerOnKeyReleaseS) {
  input_handler->on_key_press(KEY_S, 0);
  input_handler->on_key_release(KEY_S, 0);
  const auto player_movement{registry->get_component<KeyboardMovement>(0)};
  ASSERT_FALSE(player_movement->moving_south);
}

/// Test that the input handler processes a "D" key release correctly.
TEST_F(InputHandlerFixture, TestInputHandlerOnKeyReleaseD) {
  input_handler->on_key_press(KEY_D, 0);
  input_handler->on_key_release(KEY_D, 0);
  const auto player_movement{registry->get_component<KeyboardMovement>(0)};
  ASSERT_FALSE(player_movement->moving_east);
}

/// Test that the input handler processes a "C" key release correctly.
TEST_F(InputHandlerFixture, TestInputHandlerOnKeyReleaseC) {
  const auto item_id{move_player_to_item(registry, game_state, GameObjectType::HealthPotion)};
  input_handler->on_key_release(KEY_C, 0);
  const auto inventory{registry->get_component<Inventory>(0)};
  ASSERT_EQ(inventory->items.front(), item_id);
}

/// Test that the input handler processes an "E" key release correctly when the player is in the lobby and is touching
/// the goal.
TEST_F(InputHandlerFixture, TestInputHandlerOnKeyReleaseEInLobbyTouchingGoal) {
  game_state->reset_level(LevelType::Lobby);
  int called{-1};
  add_callback<EventType::GameObjectCreation>(
      [&called](const GameObjectID event, const GameObjectType) { called = event; });
  move_player_to_item(registry, game_state, GameObjectType::Goal);
  input_handler->on_key_release(KEY_E, 0);
  ASSERT_NE(called, -1);
  ASSERT_EQ(game_state->get_dungeon_level(), LevelType::FirstDungeon);
}

/// Test that the input handler processes an "E" key release correctly when the player is not in the lobby and is
/// touching the goal.
TEST_F(InputHandlerFixture, TestInputHandlerOnKeyReleaseENotInLobbyTouchingGoal) {
  int called{-1};
  add_callback<EventType::GameObjectCreation>(
      [&called](const GameObjectID event, const GameObjectType) { called = event; });
  move_player_to_item(registry, game_state, GameObjectType::Goal, false);
  input_handler->on_key_release(KEY_E, 0);
  ASSERT_EQ(called, -1);
}

/// Test that the input handler processes an "E" key release correctly when the player is in the lobby not touching the
/// goal.
TEST_F(InputHandlerFixture, TestInputHandlerOnKeyReleaseEInLobbyNotTouchingGoal) {
  game_state->reset_level(LevelType::Lobby);
  int called{-1};
  add_callback<EventType::GameObjectCreation>(
      [&called](const GameObjectID event, const GameObjectType) { called = event; });
  input_handler->on_key_release(KEY_E, 0);
  ASSERT_EQ(called, -1);
}

/// Test that the input handler processes an "E" key release correctly when the player is touching a health potion and
/// not in the lobby.
TEST_F(InputHandlerFixture, TestInputHandlerOnKeyReleaseETouchingHealthPotionNotInLobby) {
  move_player_to_item(registry, game_state, GameObjectType::HealthPotion);
  const auto health{registry->get_component<Health>(0)};
  health->set_value(50);
  input_handler->on_key_release(KEY_E, 0);
  ASSERT_EQ(health->get_value(), 55);
  ASSERT_FALSE(registry->has_game_object(game_state->get_nearest_item()));
}

/// Test that the input handler processes an "E" key release correctly when the player is in the lobby and touching the
/// shop.
TEST_F(InputHandlerFixture, TestInputHandlerOnKeyReleaseETouchingShopInLobby) {
  game_state->reset_level(LevelType::Lobby);
  auto called{false};
  add_callback<EventType::ShopOpen>([&called] { called = true; });
  move_player_to_item(registry, game_state, GameObjectType::Shop);
  input_handler->on_key_release(KEY_E, 0);
  ASSERT_TRUE(called);
}

/// Test that the input handler processes a "Q" key release correctly if the player is in the lobby.
TEST_F(InputHandlerFixture, TestInputHandlerOnKeyReleaseQInLobby) {
  game_state->reset_level(LevelType::Lobby);
  auto called{false};
  add_callback<EventType::GameOptionsOpen>([&called] { called = true; });
  input_handler->on_key_release(KEY_Q, 0);
  ASSERT_TRUE(called);
}

/// Test that the input handler processes a "Q" key release correctly if the player is not in the lobby.
TEST_F(InputHandlerFixture, TestInputHandlerOnKeyReleaseQNotInLobby) {
  auto called{false};
  add_callback<EventType::GameOptionsOpen>([&called] { called = true; });
  input_handler->on_key_release(KEY_Q, 0);
  ASSERT_FALSE(called);
}

/// Test that the input handler processes a "Z" key release correctly.
TEST_F(InputHandlerFixture, TestInputHandlerOnKeyReleaseZ) {
  input_handler->on_key_release(KEY_X, 0);
  input_handler->on_key_release(KEY_Z, 0);
  ASSERT_EQ(registry->get_component<Attack>(0)->selected_ranged_attack, 0);
}

/// Test that the input handler processes an "X" key release correctly.
TEST_F(InputHandlerFixture, TestInputHandlerOnKeyReleaseX) {
  input_handler->on_key_release(KEY_X, 0);
  ASSERT_EQ(registry->get_component<Attack>(0)->selected_ranged_attack, 1);
}

/// Test that the input handler processes an "I" key release correctly if the player is in the lobby.
TEST_F(InputHandlerFixture, TestInputHandlerOnKeyReleaseIInLobby) {
  game_state->reset_level(LevelType::Lobby);
  auto called{false};
  add_callback<EventType::InventoryOpen>([&called] { called = true; });
  input_handler->on_key_release(KEY_I, 0);
  ASSERT_FALSE(called);
}

/// Test that the input handler processes an "I" key release correctly if the player is not in the lobby.
TEST_F(InputHandlerFixture, TestInputHandlerOnKeyReleaseINotInLobby) {
  auto called{false};
  add_callback<EventType::InventoryOpen>([&called] { called = true; });
  input_handler->on_key_release(KEY_I, 0);
  ASSERT_TRUE(called);
}

/// Test that the input handler processes an unknown key release correctly.
TEST_F(InputHandlerFixture, TestInputHandlerOnKeyReleaseUnknown) {
  ASSERT_NO_THROW(input_handler->on_key_release(0, 0));
}

/// Test that the input handler processes a left mouse press correctly.
TEST_F(InputHandlerFixture, TestInputHandlerOnMousePressLeft) {
  registry->get_system<AttackSystem>()->update(10);
  int called{-1};
  add_callback<EventType::GameObjectCreation>(
      [&called](const GameObjectID event, const GameObjectType) { called = event; });
  ASSERT_TRUE(input_handler->on_mouse_press(0, 0, MOUSE_BUTTON_LEFT, 0));
  ASSERT_NE(called, -1);
}

/// Test that the input handler processes an unknown mouse press correctly.
TEST_F(InputHandlerFixture, TestInputHandlerOnMousePressUnknown) {
  ASSERT_FALSE(input_handler->on_mouse_press(0, 0, -1, 0));
}
