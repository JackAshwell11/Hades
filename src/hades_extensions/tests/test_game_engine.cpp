// Local headers
#include "ecs/systems/attacks.hpp"
#include "ecs/systems/inventory.hpp"
#include "ecs/systems/movements.hpp"
#include "ecs/systems/physics.hpp"
#include "factories.hpp"
#include "game_engine.hpp"
#include "macros.hpp"

/// Implements the fixture for the game_engine.hpp tests.
class GameEngineFixture : public testing::Test {  // NOLINT
 protected:
  /// The game engine object.
  GameEngine game_engine{0, 10};

  /// The level constants.
  std::tuple<int, int, int> level_constants{30, 20, 5};

  /// Set up the fixture for the tests.
  void SetUp() override {
    load_hitbox(GameObjectType::Player, {{0.0, 1.0}, {1.0, 2.0}, {2.0, 0.0}});
    load_hitbox(GameObjectType::Enemy, {{0.0, 1.0}, {1.0, 2.0}, {2.0, 0.0}});
  }

  /// Move the player to the position of the nearest item.
  ///
  /// @return The game object ID of the item that the player was moved to.
  GameObjectID move_player_to_nearest_item(const GameObjectType game_object_type = GameObjectType::HealthPotion) {
    const auto item_id{game_engine.get_registry()->get_game_object_ids(game_object_type).front()};
    const auto item_pos{
        cpBodyGetPosition(*game_engine.get_registry()->get_component<KinematicComponent>(item_id)->body)};
    cpBodySetPosition(*game_engine.get_registry()->get_component<KinematicComponent>(game_engine.get_player_id())->body,
                      item_pos);
    return item_id;
  }
};

/// Test that the game engine is created correctly.
TEST_F(GameEngineFixture, TestGameEngineZeroLevel) {
  ASSERT_NE(game_engine.get_registry(), nullptr);
  ASSERT_EQ(game_engine.get_player_id(), -1);
  ASSERT_EQ(game_engine.get_level_constants(), level_constants);
}

/// Test that the game engine throws an exception when given a negative level.
TEST_F(GameEngineFixture, TestGameEngineNegativeLevel) {
  ASSERT_THROW_MESSAGE(GameEngine{-1}, std::length_error, "Level must be bigger than or equal to 0.");
}

/// Test that the game engine creates game objects correctly.
TEST_F(GameEngineFixture, TestGameEngineCreateGameObjects) {
  game_engine.create_game_objects();
  ASSERT_NE(game_engine.get_player_id(), -1);
  ASSERT_EQ(game_engine.get_level_constants(), level_constants);
}

/// Test that the game engine creates game objects correctly given no seed.
TEST_F(GameEngineFixture, TestGameEngineCreateGameObjectsNoSeed) {
  GameEngine game_engine_no_seed{0};
  game_engine.create_game_objects();
  game_engine_no_seed.create_game_objects();
  ASSERT_NE(game_engine.get_player_id(), game_engine_no_seed.get_player_id());
  ASSERT_EQ(game_engine.get_level_constants(), game_engine_no_seed.get_level_constants());
}

/// Test that the game engine generates an enemy correctly.
TEST_F(GameEngineFixture, TestGameEngineGenerateEnemy) {
  auto enemy_created{-1};
  auto enemy_creation{[&](const GameObjectID enemy_id) { enemy_created = enemy_id; }};
  game_engine.create_game_objects();
  game_engine.get_registry()->add_callback<EventType::GameObjectCreation>(enemy_creation);
  game_engine.generate_enemy();
  ASSERT_NE(enemy_created, -1);
}

/// Test that the game engine throws an exception if the game objects haven't been created yet.
TEST_F(GameEngineFixture, TestGameEngineGenerateEnemyNoGameObjects) {
  ASSERT_THROW_MESSAGE(
      game_engine.generate_enemy(), RegistryError,
      "The component `KinematicComponent` for the game object ID `-1` is not registered with the registry.");
}

/// Test that the game engine throws an exception if the player is dead.
TEST_F(GameEngineFixture, TestGameEngineGenerateEnemyPlayerDead) {
  game_engine.create_game_objects();
  game_engine.get_registry()->delete_game_object(game_engine.get_player_id());
  ASSERT_THROW(game_engine.generate_enemy(), RegistryError);
}

/// Test that the game engine doesn't generate an enemy correctly if the enemy limit has been reached.
TEST_F(GameEngineFixture, TestGameEngineGenerateEnemyLimit) {
  game_engine.create_game_objects();
  for (auto i{0}; i < 10; i++) {
    game_engine.generate_enemy();
  }
  auto enemy_created{-1};
  auto enemy_creation{[&](const GameObjectID enemy_id) { enemy_created = enemy_id; }};
  game_engine.get_registry()->add_callback<EventType::GameObjectCreation>(enemy_creation);
  game_engine.generate_enemy();
  ASSERT_EQ(enemy_created, -1);
}

/// Test that the game engine processes an update correctly when there is no nearest item.
TEST_F(GameEngineFixture, TestGameEngineOnUpdateNoNearestItem) {
  game_engine.create_game_objects();
  game_engine.on_update(0);
  ASSERT_EQ(game_engine.get_nearest_item(), -1);
}

/// Test that the game engine processes an update correctly when the nearest item is not a goal.
TEST_F(GameEngineFixture, TestGameEngineOnUpdateNearestItemNotGoal) {
  game_engine.create_game_objects();
  const auto item_id{move_player_to_nearest_item()};
  game_engine.on_update(0);
  ASSERT_EQ(game_engine.get_nearest_item(), item_id);
}

/// Test that the game engine processes an update correctly when the nearest item is a goal.
TEST_F(GameEngineFixture, TestGameEngineOnUpdateNearestItemIsGoal) {
  game_engine.create_game_objects();
  move_player_to_nearest_item(GameObjectType::Goal);
  game_engine.on_update(0);
  ASSERT_FALSE(game_engine.get_registry()->has_game_object(game_engine.get_player_id()));
}

/// Test that the game engine processes a fixed update correctly.
TEST_F(GameEngineFixture, TestGameEngineOnFixedUpdateDeletePlayer) {
  game_engine.create_game_objects();
  game_engine.get_registry()->mark_for_deletion(game_engine.get_player_id());
  game_engine.on_fixed_update(0.0);
  ASSERT_FALSE(game_engine.get_registry()->has_game_object(game_engine.get_player_id()));
}

/// Test that the game engine processes a 'W' key press correctly.
TEST_F(GameEngineFixture, TestGameEngineOnKeyPressW) {
  game_engine.create_game_objects();
  game_engine.on_key_press(KEY_W, 0);
  const auto player_movement{game_engine.get_registry()->get_component<KeyboardMovement>(game_engine.get_player_id())};
  ASSERT_TRUE(player_movement->moving_north);
}

/// Test that the game engine processes an 'A' key press correctly.
TEST_F(GameEngineFixture, TestGameEngineOnKeyPressA) {
  game_engine.create_game_objects();
  game_engine.on_key_press(KEY_A, 0);
  const auto player_movement{game_engine.get_registry()->get_component<KeyboardMovement>(game_engine.get_player_id())};
  ASSERT_TRUE(player_movement->moving_west);
}

/// Test that the game engine processes an 'S' key press correctly.
TEST_F(GameEngineFixture, TestGameEngineOnKeyPressS) {
  game_engine.create_game_objects();
  game_engine.on_key_press(KEY_S, 0);
  const auto player_movement{game_engine.get_registry()->get_component<KeyboardMovement>(game_engine.get_player_id())};
  ASSERT_TRUE(player_movement->moving_south);
}

/// Test that the game engine processes a 'D' key press correctly.
TEST_F(GameEngineFixture, TestGameEngineOnKeyPressD) {
  game_engine.create_game_objects();
  game_engine.on_key_press(KEY_D, 0);
  const auto player_movement{game_engine.get_registry()->get_component<KeyboardMovement>(game_engine.get_player_id())};
  ASSERT_TRUE(player_movement->moving_east);
}

/// Test that the game engine processes an unknown key press correctly.
TEST_F(GameEngineFixture, TestGameEngineOnKeyPressUnknown) {
  game_engine.create_game_objects();
  ASSERT_NO_THROW(game_engine.on_key_press(0, 0));
}

/// Test that the game engine processes a 'W' key release correctly.
TEST_F(GameEngineFixture, TestGameEngineOnKeyReleaseW) {
  game_engine.create_game_objects();
  game_engine.on_key_press(KEY_W, 0);
  game_engine.on_key_release(KEY_W, 0);
  const auto player_movement{game_engine.get_registry()->get_component<KeyboardMovement>(game_engine.get_player_id())};
  ASSERT_FALSE(player_movement->moving_north);
}

/// Test that the game engine processes an 'A' key release correctly.
TEST_F(GameEngineFixture, TestGameEngineOnKeyReleaseA) {
  game_engine.create_game_objects();
  game_engine.on_key_press(KEY_A, 0);
  game_engine.on_key_release(KEY_A, 0);
  const auto player_movement{game_engine.get_registry()->get_component<KeyboardMovement>(game_engine.get_player_id())};
  ASSERT_FALSE(player_movement->moving_west);
}

/// Test that the game engine processes an 'S' key release correctly.
TEST_F(GameEngineFixture, TestGameEngineOnKeyReleaseS) {
  game_engine.create_game_objects();
  game_engine.on_key_press(KEY_S, 0);
  game_engine.on_key_release(KEY_S, 0);
  const auto player_movement{game_engine.get_registry()->get_component<KeyboardMovement>(game_engine.get_player_id())};
  ASSERT_FALSE(player_movement->moving_south);
}

/// Test that the game engine processes a 'D' key release correctly.
TEST_F(GameEngineFixture, TestGameEngineOnKeyReleaseD) {
  game_engine.create_game_objects();
  game_engine.on_key_press(KEY_D, 0);
  game_engine.on_key_release(KEY_D, 0);
  const auto player_movement{game_engine.get_registry()->get_component<KeyboardMovement>(game_engine.get_player_id())};
  ASSERT_FALSE(player_movement->moving_east);
}

/// Test that the game engine processes a 'C' key release correctly.
TEST_F(GameEngineFixture, TestGameEngineOnKeyReleaseC) {
  game_engine.create_game_objects();
  const auto item_id{move_player_to_nearest_item()};
  game_engine.on_update(0);
  game_engine.on_key_release(KEY_C, 0);
  const auto inventory{game_engine.get_registry()->get_component<Inventory>(game_engine.get_player_id())};
  ASSERT_EQ(inventory->items.front(), item_id);
}

/// Test that the game engine processes an 'E' key release correctly.
TEST_F(GameEngineFixture, TestGameEngineOnKeyReleaseE) {
  game_engine.create_game_objects();
  game_engine.generate_enemy();
  move_player_to_nearest_item();
  game_engine.on_update(0);
  const auto health{game_engine.get_registry()->get_component<Health>(game_engine.get_player_id())};
  health->set_value(50);
  game_engine.on_key_release(KEY_E, 0);
  ASSERT_EQ(health->get_value(), 55);
}

/// Test that the game engine processes a 'Z' key release correctly.
TEST_F(GameEngineFixture, TestGameEngineOnKeyReleaseZ) {
  game_engine.create_game_objects();
  game_engine.generate_enemy();
  game_engine.on_key_release(KEY_X, 0);
  game_engine.on_key_release(KEY_Z, 0);
  ASSERT_EQ(game_engine.get_registry()->get_component<Attack>(game_engine.get_player_id())->selected_ranged_attack, 0);
}

/// Test that the game engine processes a 'X' key release correctly.
TEST_F(GameEngineFixture, TestGameEngineOnKeyReleaseX) {
  game_engine.create_game_objects();
  game_engine.generate_enemy();
  game_engine.on_key_release(KEY_X, 0);
  ASSERT_EQ(game_engine.get_registry()->get_component<Attack>(game_engine.get_player_id())->selected_ranged_attack, 1);
}

/// Test that the game engine processes an unknown key release correctly.
TEST_F(GameEngineFixture, TestGameEngineOnKeyReleaseUnknown) {
  game_engine.create_game_objects();
  ASSERT_NO_THROW(game_engine.on_key_release(0, 0));
}

/// Test that the game engine processes a left mouse press correctly.
TEST_F(GameEngineFixture, TestGameEngineOnMousePressLeft) {
  // Set up the game engine so that the player can attack an enemy
  game_engine.create_game_objects();
  game_engine.generate_enemy();
  game_engine.get_registry()->get_system<AttackSystem>()->update(10);

  // Test that processing the left mouse press works correctly
  int called{-1};
  game_engine.get_registry()->add_callback<EventType::GameObjectCreation>(
      [&called](const GameObjectID event) { called = event; });
  ASSERT_TRUE(game_engine.on_mouse_press(0, 0, MOUSE_BUTTON_LEFT, 0));
  ASSERT_NE(called, -1);
}

/// Test that the game engine processes a unknown mouse press correctly.
TEST_F(GameEngineFixture, TestGameEngineOnMousePressUnknown) {
  game_engine.create_game_objects();
  ASSERT_FALSE(game_engine.on_mouse_press(0, 0, -1, 0));
}
