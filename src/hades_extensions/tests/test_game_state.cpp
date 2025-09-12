// Local headers
#include "ecs/systems/inventory.hpp"
#include "ecs/systems/level.hpp"
#include "events.hpp"
#include "factories.hpp"
#include "game_helpers.hpp"
#include "macros.hpp"

/// Implements the fixture for the game_state.hpp tests.
class GameStateFixture : public testing::Test {  // NOLINT
 protected:
  /// The registry which manages the game objects, components, and systems.
  std::shared_ptr<Registry> registry;

  /// The game state which stores the state of the game.
  std::shared_ptr<GameState> game_state;

  /// Set up shared logic for the tests.
  static void SetUpTestSuite() {
    load_hitbox(GameObjectType::Player, {{0.0, 1.0}, {1.0, 2.0}, {2.0, 0.0}});
    load_hitbox(GameObjectType::Enemy, {{0.0, 1.0}, {1.0, 2.0}, {2.0, 0.0}});
  }

  /// Set up the fixture for the tests.
  void SetUp() override {
    registry = std::make_shared<Registry>();
    game_state = std::make_shared<GameState>(registry);
    registry->add_system<InventorySystem>();
    registry->add_system<PhysicsSystem>();
  }

  /// Tear down the fixture after the tests.
  void TearDown() override { clear_listeners(); }
};

/// Test that getting the nearest item works correctly.
TEST_F(GameStateFixture, TestGameStateNearestItem) {
  ASSERT_EQ(game_state->get_nearest_item(), -1);
  game_state->set_nearest_item(123);
  ASSERT_EQ(game_state->get_nearest_item(), 123);
}

/// Test that the enemy generation timer works correctly.
TEST_F(GameStateFixture, TestGameStateEnemyGenerationTimer) {
  ASSERT_EQ(game_state->get_enemy_generation_timer(), 0.0);
  game_state->set_enemy_generation_timer(1.0);
  ASSERT_EQ(game_state->get_enemy_generation_timer(), 1.0);
  game_state->set_enemy_generation_timer(game_state->get_enemy_generation_timer() + 2.0);
  ASSERT_EQ(game_state->get_enemy_generation_timer(), 3.0);
  game_state->set_enemy_generation_timer(0.0);
  ASSERT_EQ(game_state->get_enemy_generation_timer(), 0.0);
}

/// Test that the player is not touching a game object type when there is no nearest item.
TEST_F(GameStateFixture, TestGameStateIsPlayerTouchingTypeNoNearest) {
  game_state->reset_level(LevelType::FirstDungeon);
  ASSERT_FALSE(game_state->is_player_touching_type(GameObjectType::Goal));
}

/// Test that the player is not touching a game object type when the nearest item is not the specified type.
TEST_F(GameStateFixture, TestGameStateIsPlayerTouchingTypeNotSpecifiedType) {
  game_state->reset_level(LevelType::FirstDungeon);
  move_player_to_item(registry, game_state, GameObjectType::HealthPotion);
  ASSERT_FALSE(game_state->is_player_touching_type(GameObjectType::Goal));
}

/// Test that the player is touching a game object type when the nearest item is the specified type.
TEST_F(GameStateFixture, TestGameStateIsPlayerTouchingTypeIsSpecifiedType) {
  game_state->reset_level(LevelType::Lobby);
  move_player_to_item(registry, game_state, GameObjectType::Goal);
  ASSERT_TRUE(game_state->is_player_touching_type(GameObjectType::Goal));
}

/// Test that the player is initialised on constructing the game state.
TEST_F(GameStateFixture, TestGameStatePlayerInitialised) {
  ASSERT_EQ(game_state->get_player_id(), 0);
  ASSERT_TRUE(registry->has_game_object(game_state->get_player_id()));
  ASSERT_EQ(registry->get_game_object_type(game_state->get_player_id()), GameObjectType::Player);
}

/// Test that the dungeon is initialised on constructing the game state.
TEST_F(GameStateFixture, TestGameStateDungeonInitialised) {
  ASSERT_EQ(game_state->get_game_level(), 1);
  ASSERT_EQ(game_state->get_dungeon_level(), LevelType::None);
}

/// Test that setting the seed works correctly.
TEST_F(GameStateFixture, TestGameStateSetSeed) {
  const std::string seed{"test_seed"};
  ASSERT_NO_THROW(game_state->set_seed(seed));
}

/// Test setting the window size works correctly.
TEST_F(GameStateFixture, TestGameStateSetWindowSize) {
  constexpr std::pair original_window_size{0, 0};
  constexpr std::pair expected_window_size{1024, 768};
  ASSERT_EQ(game_state->get_window_size(), original_window_size);
  game_state->set_window_size(1024, 768);
  ASSERT_EQ(game_state->get_window_size(), expected_window_size);
}

/// Test that the dungeon can be initialised with an arbitrary player level.
TEST_F(GameStateFixture, TestGameStateInitialiseDungeonRun) {
  registry->get_component<PlayerLevel>(game_state->get_player_id())->level = 5;
  game_state->initialise_dungeon_run();
  ASSERT_EQ(game_state->get_game_level(), 5);
}

/// Test that resetting to a lobby creates the lobby game objects.
TEST_F(GameStateFixture, TestGameStateResetLevelLobby) {
  game_state->reset_level(LevelType::Lobby);
  ASSERT_EQ(game_state->get_player_id(), 0);
  ASSERT_EQ(game_state->get_nearest_item(), -1);
  ASSERT_EQ(game_state->get_dungeon_level(), LevelType::Lobby);
  ASSERT_EQ(game_state->get_game_level(), 1);
  ASSERT_TRUE(game_state->is_lobby());
  ASSERT_FALSE(game_state->is_boss());
  ASSERT_EQ(game_state->get_enemy_generation_timer(), 0.0);
  ASSERT_TRUE(registry->get_game_object_ids(GameObjectType::Enemy).empty());
  ASSERT_TRUE(registry->get_game_object_ids(GameObjectType::HealthPotion).empty());
  ASSERT_FALSE(registry->get_game_object_ids(GameObjectType::Goal).empty());
}

/// Test that resetting to a lobby clears all game objects and inventory items except the player.
TEST_F(GameStateFixture, TestGameStateResetLevelLobbyClearEverything) {
  game_state->reset_level(LevelType::FirstDungeon);
  const auto item_id{get_item(registry, GameObjectType::HealthPotion)};
  registry->get_system<InventorySystem>()->add_item_to_inventory(game_state->get_player_id(), item_id);
  game_state->reset_level(LevelType::Lobby);
  const auto health_potion_ids{registry->get_game_object_ids(GameObjectType::HealthPotion)};
  ASSERT_TRUE(health_potion_ids.empty());
  ASSERT_TRUE(registry->get_component<Inventory>(game_state->get_player_id())->items.empty());
  ASSERT_TRUE(registry->get_game_object_ids(GameObjectType::HealthPotion).empty());
}

/// Test that resetting to a game level works correctly.
TEST_F(GameStateFixture, TestGameStateResetLevelFirstSecond) {
  game_state->reset_level(LevelType::FirstDungeon);
  ASSERT_EQ(game_state->get_player_id(), 0);
  ASSERT_EQ(game_state->get_nearest_item(), -1);
  ASSERT_EQ(game_state->get_dungeon_level(), LevelType::FirstDungeon);
  ASSERT_EQ(game_state->get_game_level(), 1);
  ASSERT_FALSE(game_state->is_lobby());
  ASSERT_FALSE(game_state->is_boss());
  ASSERT_EQ(game_state->get_enemy_generation_timer(), 0.0);
  ASSERT_TRUE(registry->get_game_object_ids(GameObjectType::Enemy).empty());
  ASSERT_FALSE(registry->get_game_object_ids(GameObjectType::HealthPotion).empty());
  ASSERT_FALSE(registry->get_game_object_ids(GameObjectType::Goal).empty());
}

/// Test that resetting to a game level clears all game objects except the player and their inventory.
TEST_F(GameStateFixture, TestGameStateResetLevelFirstSecondClearObjects) {
  game_state->reset_level(LevelType::FirstDungeon);
  const auto item_id{get_item(registry, GameObjectType::HealthPotion)};
  registry->get_system<InventorySystem>()->add_item_to_inventory(game_state->get_player_id(), item_id);
  game_state->reset_level(LevelType::FirstDungeon);
  const auto health_potion_ids{registry->get_game_object_ids(GameObjectType::HealthPotion)};
  ASSERT_TRUE(std::ranges::find(health_potion_ids.begin(), health_potion_ids.end(), item_id) !=
              health_potion_ids.end());
  const auto items{registry->get_component<Inventory>(game_state->get_player_id())->items};
  ASSERT_TRUE(std::ranges::find(items.begin(), items.end(), item_id) != items.end());
  ASSERT_FALSE(registry->get_game_object_ids(GameObjectType::HealthPotion).empty());
}

/// Test that resetting the level to a lobby calls the correct callbacks.
TEST_F(GameStateFixture, TestGameStateResetLevelLobbyCallbacks) {
  auto inventory_update{false};
  auto ranged_attack_switch{false};
  auto attack_cooldown_update{false};
  auto status_effect_update{false};
  auto game_open{false};
  add_callback<EventType::InventoryUpdate>(
      [&inventory_update](const std::vector<GameObjectID>&) { inventory_update = true; });
  add_callback<EventType::RangedAttackSwitch>([&ranged_attack_switch](const int) { ranged_attack_switch = true; });
  add_callback<EventType::AttackCooldownUpdate>(
      [&attack_cooldown_update](const GameObjectID, const double, const double, const double) {
        attack_cooldown_update = true;
      });
  add_callback<EventType::StatusEffectUpdate>(
      [&status_effect_update](const std::unordered_map<EffectType, double>&) { status_effect_update = true; });
  add_callback<EventType::GameOpen>([&game_open] { game_open = true; });
  game_state->reset_level(LevelType::Lobby);
  ASSERT_TRUE(inventory_update);
  ASSERT_TRUE(ranged_attack_switch);
  ASSERT_TRUE(attack_cooldown_update);
  ASSERT_TRUE(status_effect_update);
  ASSERT_TRUE(game_open);
}

/// Test that resetting the level to a game level calls the correct callbacks.
TEST_F(GameStateFixture, TestGameStateResetLevelGameLevelCallbacks) {
  auto game_open{false};
  add_callback<EventType::GameOpen>([&game_open] { game_open = true; });
  game_state->reset_level(LevelType::FirstDungeon);
  ASSERT_TRUE(game_open);
}

/// Test that resetting the level to a game level when it hasn't been reset to the lobby works correctly.
TEST_F(GameStateFixture, TestGameStateResetLevelFirstSecondWithoutLobby) {
  GameState game_state_no_lobby{registry};
  game_state_no_lobby.reset_level(LevelType::FirstDungeon);
  ASSERT_EQ(game_state_no_lobby.get_player_id(), 0);
  ASSERT_EQ(game_state_no_lobby.get_dungeon_level(), LevelType::FirstDungeon);
  ASSERT_EQ(game_state_no_lobby.get_game_level(), 1);
  ASSERT_FALSE(game_state_no_lobby.is_lobby());
  ASSERT_FALSE(game_state_no_lobby.is_boss());
}

/// Test that the game state generates an enemy correctly.
TEST_F(GameStateFixture, TestGameStateGenerateEnemyValid) {
  game_state->reset_level(LevelType::FirstDungeon);
  auto enemy_created{-1};
  auto enemy_creation{[&](const GameObjectID enemy_id, const GameObjectType) { enemy_created = enemy_id; }};
  add_callback<EventType::GameObjectCreation>(enemy_creation);
  game_state->generate_enemy();
  ASSERT_NE(enemy_created, -1);
}

/// Test that the game state doesn't generate an enemy correctly if there are no valid positions.
TEST_F(GameStateFixture, TestGameStateGenerateEnemyNoValidPositions) {
  auto enemy_created{-1};
  auto enemy_creation{[&](const GameObjectID enemy_id, const GameObjectType) { enemy_created = enemy_id; }};
  add_callback<EventType::GameObjectCreation>(enemy_creation);
  game_state->generate_enemy();
  ASSERT_EQ(enemy_created, -1);
}
