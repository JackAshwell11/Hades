// Local headers
#include "events.hpp"
#include "factories.hpp"
#include "game_engine.hpp"
#include "game_helpers.hpp"
#include "macros.hpp"

/// Implements the fixture for the game_engine.hpp tests.
class GameEngineFixture : public testing::Test {  // NOLINT
 protected:
  /// The game engine object.
  GameEngine game_engine;

  /// Set up shared logic for the tests.
  static void SetUpTestSuite() {
    load_hitbox(GameObjectType::Player, {{0.0, 1.0}, {1.0, 2.0}, {2.0, 0.0}});
    load_hitbox(GameObjectType::Enemy, {{0.0, 1.0}, {1.0, 2.0}, {2.0, 0.0}});
  }

  /// Set up the fixture for the tests.
  void SetUp() override { get_game_state()->reset_level(LevelType::FirstDungeon); }

  /// Tear down the fixture after the tests.
  void TearDown() override { clear_listeners(); }

  /// Get the registry of the game engine.
  ///
  /// @return The registry of the game engine.
  std::shared_ptr<Registry> get_registry() { return game_engine.get_registry(); }

  /// Get the game state of the game engine.
  ///
  /// @return The game state of the game engine.
  std::shared_ptr<GameState> get_game_state() { return game_engine.get_game_state(); }
};

/// Test that the game engine processes an update correctly when there is no player.
TEST_F(GameEngineFixture, TestGameEngineOnUpdateNoPlayer) {
  get_registry()->delete_game_object(get_game_state()->get_player_id());
  ASSERT_THROW_MESSAGE(
      game_engine.on_update(0), RegistryError,
      "The component `KinematicComponent` for the game object ID `0` is not registered with the registry.");
}

/// Test that the game engine processes an update correctly when there is no nearest item.
TEST_F(GameEngineFixture, TestGameEngineOnUpdateNoNearestItem) {
  game_engine.on_update(0);
  ASSERT_EQ(get_game_state()->get_nearest_item(), -1);
}

/// Test that the game engine processes an update correctly when the nearest item is not a goal.
TEST_F(GameEngineFixture, TestGameEngineOnUpdateNearestItemNotGoal) {
  const auto item_id{move_player_to_item(get_registry(), get_game_state(), GameObjectType::HealthPotion)};
  game_engine.on_update(0);
  ASSERT_EQ(get_game_state()->get_nearest_item(), item_id);
}

/// Test that the game engine processes an update correctly when the nearest item is a goal and the player is in the
/// lobby.
TEST_F(GameEngineFixture, TestGameEngineOnUpdateNearestItemIsGoalInLobby) {
  get_game_state()->reset_level(LevelType::Lobby);
  auto called{-1};
  add_callback<EventType::GameObjectCreation>(
      [&](const GameObjectID event, const GameObjectType, const std::pair<double, double>&) { called = event; });
  move_player_to_item(get_registry(), get_game_state(), GameObjectType::Goal);
  game_engine.on_update(0);
  ASSERT_EQ(called, -1);
  ASSERT_EQ(get_game_state()->get_dungeon_level(), LevelType::Lobby);
}

/// Test that the game engine processes an update correctly when the nearest item is a goal and the player hasn't
/// completed any levels.
TEST_F(GameEngineFixture, TestGameEngineOnUpdateNearestItemIsGoalFirstLevel) {
  auto called{-1};
  add_callback<EventType::GameObjectCreation>(
      [&](const GameObjectID event, const GameObjectType, const std::pair<double, double>&) { called = event; });
  move_player_to_item(get_registry(), get_game_state(), GameObjectType::Goal);
  game_engine.on_update(0);
  ASSERT_NE(called, -1);
  ASSERT_EQ(get_game_state()->get_dungeon_level(), LevelType::SecondDungeon);
}

/// Test that the game engine processes an update correctly when the nearest item is a goal and the player has completed
/// all game levels.
TEST_F(GameEngineFixture, TestGameEngineOnUpdateNearestItemIsGoalCompletedFirstSecondLevels) {
  get_game_state()->reset_level(LevelType::SecondDungeon);
  auto called{-1};
  add_callback<EventType::GameObjectCreation>(
      [&](const GameObjectID event, const GameObjectType, const std::pair<double, double>&) { called = event; });
  move_player_to_item(get_registry(), get_game_state(), GameObjectType::Goal);
  game_engine.on_update(0);
  ASSERT_NE(called, -1);
  ASSERT_EQ(get_game_state()->get_dungeon_level(), LevelType::Boss);
}

/// Test that the game engine processes an update correctly when the nearest item is a goal and the player has completed
/// the last level.
TEST_F(GameEngineFixture, TestGameEngineOnUpdateNearestItemIsGoalCompletedLastLevel) {
  get_game_state()->reset_level(LevelType::Boss);
  auto called{-1};
  add_callback<EventType::GameObjectCreation>(
      [&](const GameObjectID event, const GameObjectType, const std::pair<double, double>&) { called = event; });
  move_player_to_item(get_registry(), get_game_state(), GameObjectType::Goal);
  game_engine.on_update(0);
  ASSERT_NE(called, -1);
  ASSERT_EQ(get_game_state()->get_dungeon_level(), LevelType::Lobby);
}

/// Test that the game engine generates an enemy correctly.
TEST_F(GameEngineFixture, TestGameEngineOnUpdateGenerateEnemyTimerReached) {
  auto enemy_created{-1};
  auto enemy_creation{[&](const GameObjectID enemy_id, const GameObjectType, const std::pair<double, double>&) {
    enemy_created = enemy_id;
  }};
  add_callback<EventType::GameObjectCreation>(enemy_creation);
  game_engine.on_update(1);
  ASSERT_NE(enemy_created, -1);
  ASSERT_EQ(get_game_state()->get_enemy_generation_timer(), 0.0);
}

/// Test that the game engine can't generate an enemy if the timer is not reached.
TEST_F(GameEngineFixture, TestGameEngineOnUpdateGenerateEnemyTimerNotReached) {
  auto enemy_created{-1};
  auto enemy_creation{[&](const GameObjectID enemy_id, const GameObjectType, const std::pair<double, double>&) {
    enemy_created = enemy_id;
  }};
  add_callback<EventType::GameObjectCreation>(enemy_creation);
  game_engine.on_update(0.5);
  ASSERT_EQ(enemy_created, -1);
  ASSERT_EQ(get_game_state()->get_enemy_generation_timer(), 0.5);
}

/// Test that the game engine processes a fixed physics update correctly.
TEST_F(GameEngineFixture, TestGameEngineOnFixedUpdatePhysicsStep) {
  const auto kinematic_component{get_registry()->get_component<KinematicComponent>(get_game_state()->get_player_id())};
  cpBodySetPosition(*kinematic_component->body, cpv(1.0, 1.0));
  cpBodySetVelocity(*kinematic_component->body, cpv(5.0, 0.0));
  game_engine.on_fixed_update(1.0);
  ASSERT_EQ(cpBodyGetPosition(*kinematic_component->body), cpv(6.0, 1.0));
  ASSERT_NEAR(cpBodyGetVelocity(*kinematic_component->body).x, 0.0005, 1e-6);
  ASSERT_NEAR(cpBodyGetVelocity(*kinematic_component->body).y, 0.0, 1e-6);
}

/// Test that the game engine processes a fixed update to delete a player correctly.
TEST_F(GameEngineFixture, TestGameEngineOnFixedUpdateDeletePlayer) {
  get_registry()->mark_for_deletion(get_game_state()->get_player_id());
  game_engine.on_fixed_update(0.0);
  ASSERT_FALSE(get_registry()->has_game_object(get_game_state()->get_player_id()));
}
