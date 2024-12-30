// Local headers
#include "factories.hpp"
#include "game_engine.hpp"
#include "macros.hpp"

/// Implements the fixture for the game_engine.hpp tests.
class GameEngineFixture : public testing::Test {  // NOLINT
 protected:
  /// The game engine object.
  GameEngine game_engine{0, 10};

  void SetUp() override {
    load_hitbox(GameObjectType::Player, {{0, 0}});
    load_hitbox(GameObjectType::Enemy, {{0, 0}});
  }
};

/// Test that the game engine is created correctly.
TEST_F(GameEngineFixture, TestGameEngineZeroLevel) {
  ASSERT_NE(game_engine.get_registry(), nullptr);
  ASSERT_EQ(game_engine.get_player_id(), -1);
}

/// Test that the game engine throws an exception when given a negative level.
TEST_F(GameEngineFixture, TestGameEngineNegativeLevel) {
  ASSERT_THROW_MESSAGE(GameEngine{-1}, std::length_error, "Level must be bigger than or equal to 0.");
}

/// Test that the game engine creates game objects correctly.
TEST_F(GameEngineFixture, TestGameEngineCreateGameObjects) {
  game_engine.create_game_objects();
  ASSERT_NE(game_engine.get_player_id(), -1);
}

/// Test that the game engine creates game objects correctly given no seed.
TEST_F(GameEngineFixture, TestGameEngineCreateGameObjectsNoSeed) {
  GameEngine game_engine_no_seed{0};
  game_engine.create_game_objects();
  game_engine_no_seed.create_game_objects();
  ASSERT_NE(game_engine.get_player_id(), game_engine_no_seed.get_player_id());
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
