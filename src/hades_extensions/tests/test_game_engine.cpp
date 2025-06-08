// External headers
#include <nlohmann/json.hpp>

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

  /// Get an item from the game engine's registry.
  ///
  /// @param game_object_type - The type of the item to find.
  /// @return The game object ID of the item.
  auto get_item(const GameObjectType game_object_type) -> GameObjectID {
    return game_engine.get_registry().get_game_object_ids(game_object_type).front();
  }

  /// Move the player to the position of an item.
  ///
  /// @param item_id - The ID of the item to move the player to.
  void move_player_to_item(const GameObjectID item_id) {
    const auto item_pos{
        cpBodyGetPosition(*game_engine.get_registry().get_component<KinematicComponent>(item_id)->body)};
    cpBodySetPosition(*game_engine.get_registry().get_component<KinematicComponent>(game_engine.get_player_id())->body,
                      item_pos);
  }
};

/// Test that the game engine is created correctly.
TEST_F(GameEngineFixture, TestGameEngineZeroLevel) { ASSERT_EQ(game_engine.get_player_id(), -1); }

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

/// Test that setting up the shop with a stat offering works correctly.
TEST_F(GameEngineFixture, TestGameEngineSetupShopSingleValidStat) {
  game_engine.create_game_objects();
  std::tuple<int, std::tuple<std::string, std::string, std::string>, int> callback_args;
  game_engine.get_registry().add_callback<EventType::ShopItemLoaded>(
      [&callback_args](const int offering_index, const std::tuple<std::string, std::string, std::string> &data,
                       const int cost) { callback_args = std::make_tuple(offering_index, data, cost); });
  std::istringstream shop_stream{R"([
        {
            "type": "stat",
            "name": "Health Boost",
            "description": "Increases health",
            "icon_type": "health",
            "base_cost": 100.0,
            "cost_multiplier": 1.5,
            "stat_type": "Health",
            "base_value": 10.0,
            "value_multiplier": 1.2
        }
    ])"};
  game_engine.setup_shop(shop_stream);
  ASSERT_EQ(std::get<0>(callback_args), 0);
  ASSERT_EQ(std::get<1>(callback_args), std::make_tuple("Health Boost", "Increases health", "health"));
  ASSERT_EQ(std::get<2>(callback_args), 100);
}

/// Test that setting up the shop with a component offering works correctly.
TEST_F(GameEngineFixture, TestGameEngineSetupShopSingleValidComponent) {
  game_engine.create_game_objects();
  std::tuple<int, std::tuple<std::string, std::string, std::string>, int> callback_args;
  game_engine.get_registry().add_callback<EventType::ShopItemLoaded>(
      [&callback_args](const int offering_index, const std::tuple<std::string, std::string, std::string> &data,
                       const int cost) { callback_args = std::make_tuple(offering_index, data, cost); });
  std::istringstream shop_stream{R"([
        {
            "type": "component",
            "name": "Speed Boost",
            "description": "Increases speed",
            "icon_type": "speed",
            "base_cost": 150.0,
            "cost_multiplier": 1.2
        }
    ])"};
  game_engine.setup_shop(shop_stream);
  ASSERT_EQ(std::get<0>(callback_args), 0);
  ASSERT_EQ(std::get<1>(callback_args), std::make_tuple("Speed Boost", "Increases speed", "speed"));
  ASSERT_EQ(std::get<2>(callback_args), 150);
}

/// Test that setting up the shop with an item offering works correctly.
TEST_F(GameEngineFixture, TestGameEngineSetupShopSingleValidItem) {
  game_engine.create_game_objects();
  std::tuple<int, std::tuple<std::string, std::string, std::string>, int> callback_args;
  game_engine.get_registry().add_callback<EventType::ShopItemLoaded>(
      [&callback_args](const int offering_index, const std::tuple<std::string, std::string, std::string> &data,
                       const int cost) { callback_args = std::make_tuple(offering_index, data, cost); });
  std::istringstream shop_stream{R"([
        {
            "type": "item",
            "name": "Mana Potion",
            "description": "Restores mana",
            "icon_type": "mana",
            "base_cost": 50.0,
            "cost_multiplier": 1.1
        }
    ])"};
  game_engine.setup_shop(shop_stream);
  ASSERT_EQ(std::get<0>(callback_args), 0);
  ASSERT_EQ(std::get<1>(callback_args), std::make_tuple("Mana Potion", "Restores mana", "mana"));
  ASSERT_EQ(std::get<2>(callback_args), 50);
}

/// Test that setting up the shop with multiple offerings works correctly.
TEST_F(GameEngineFixture, TestGameEngineSetupShopMultipleValidOfferings) {
  game_engine.create_game_objects();
  std::vector<std::tuple<int, std::tuple<std::string, std::string, std::string>, int>> callback_args;
  game_engine.get_registry().add_callback<EventType::ShopItemLoaded>(
      [&callback_args](const int offering_index, const std::tuple<std::string, std::string, std::string> &data,
                       const int cost) { callback_args.emplace_back(offering_index, data, cost); });
  std::istringstream shop_stream{R"([
        {
            "type": "stat",
            "name": "Health Boost",
            "description": "Increases health",
            "icon_type": "health",
            "base_cost": 100.0,
            "cost_multiplier": 1.5,
            "stat_type": "Health",
            "base_value": 10.0,
            "value_multiplier": 1.2
        },
        {
            "type": "component",
            "name": "Speed Boost",
            "description": "Increases speed",
            "icon_type": "speed",
            "base_cost": 150.0,
            "cost_multiplier": 1.2
        },
        {
            "type": "item",
            "name": "Mana Potion",
            "description": "Restores mana",
            "icon_type": "mana",
            "base_cost": 50.0,
            "cost_multiplier": 1.1
        }
    ])"};
  game_engine.setup_shop(shop_stream);
  ASSERT_EQ(callback_args.size(), 3);

  // Check the stat offering
  ASSERT_EQ(std::get<0>(callback_args[0]), 0);
  ASSERT_EQ(std::get<1>(callback_args[0]), std::make_tuple("Health Boost", "Increases health", "health"));
  ASSERT_EQ(std::get<2>(callback_args[0]), 100);

  // Check the component offering
  ASSERT_EQ(std::get<0>(callback_args[1]), 1);
  ASSERT_EQ(std::get<1>(callback_args[1]), std::make_tuple("Speed Boost", "Increases speed", "speed"));
  ASSERT_EQ(std::get<2>(callback_args[1]), 150);

  // Check the item offering
  ASSERT_EQ(std::get<0>(callback_args[2]), 2);
  ASSERT_EQ(std::get<1>(callback_args[2]), std::make_tuple("Mana Potion", "Restores mana", "mana"));
  ASSERT_EQ(std::get<2>(callback_args[2]), 50);
}

/// Test that setting up the shop with an unknown type throws an exception.
TEST_F(GameEngineFixture, TestGameEngineSetupShopUnknownType) {
  game_engine.create_game_objects();
  std::istringstream shop_stream{R"([
        {
            "type": "unknown",
            "name": "Mystery Item",
            "description": "An item of unknown type",
            "icon_type": "mystery",
            "base_cost": 100.0,
            "cost_multiplier": 1.5
        }
    ])"};
  ASSERT_THROW_MESSAGE(game_engine.setup_shop(shop_stream), std::runtime_error, "Unknown offering type: unknown");
}

/// Test that setting up the shop with an invalid JSON format throws an exception.
TEST_F(GameEngineFixture, TestGameEngineSetupShopInvalidJSON) {
  game_engine.create_game_objects();
  std::istringstream shop_stream{R"([
        {
            "type": "stat",
            "name": "Invalid Item",
            "description": "This item has an invalid JSON format",
            "icon_type": "invalid",
            "base_cost": 100.0,
            "cost_multiplier": 1.5,
        }
    ])"};
  ASSERT_THROW_MESSAGE(game_engine.setup_shop(shop_stream), nlohmann::json::exception,
                       "[json.exception.parse_error.101] parse error at line 9, column 9: syntax error while parsing "
                       "object key - unexpected '}'; expected string literal");
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
  const auto item_id{get_item(GameObjectType::HealthPotion)};
  move_player_to_item(item_id);
  game_engine.on_update(0);
  ASSERT_EQ(game_engine.get_nearest_item(), item_id);
}

/// Test that the game engine processes an update correctly when the nearest item is a goal.
TEST_F(GameEngineFixture, TestGameEngineOnUpdateNearestItemIsGoal) {
  game_engine.create_game_objects();
  move_player_to_item(get_item(GameObjectType::Goal));
  game_engine.on_update(0);
  ASSERT_FALSE(game_engine.get_registry().has_game_object(game_engine.get_player_id()));
}

/// Test that the game engine generates an enemy correctly.
TEST_F(GameEngineFixture, TestGameEngineOnUpdateGenerateEnemy) {
  auto enemy_created{-1};
  auto enemy_creation{[&](const GameObjectID enemy_id) { enemy_created = enemy_id; }};
  game_engine.create_game_objects();
  game_engine.get_registry().add_callback<EventType::GameObjectCreation>(enemy_creation);
  game_engine.on_update(1);
  ASSERT_NE(enemy_created, -1);
}

/// Test that the game engine throws an exception if the game objects haven't been created yet.
TEST_F(GameEngineFixture, TestGameEngineOnUpdateGenerateEnemyNoGameObjects) {
  GameEngine engine{0};
  ASSERT_THROW_MESSAGE(
      engine.on_update(1), RegistryError,
      "The component `KinematicComponent` for the game object ID `-1` is not registered with the registry.");
}

/// Test that the game engine throws an exception if the player is dead.
TEST_F(GameEngineFixture, TestGameEngineOnUpdateGenerateEnemyPlayerDead) {
  game_engine.create_game_objects();
  game_engine.get_registry().delete_game_object(game_engine.get_player_id());
  ASSERT_THROW(game_engine.on_update(1), RegistryError);
}

/// Test that the game engine doesn't generate an enemy correctly if the enemy limit has been reached.
TEST_F(GameEngineFixture, TestGameEngineOnUpdateGenerateEnemyLimit) {
  game_engine.create_game_objects();
  for (auto i{0}; i < 10; i++) {
    game_engine.on_update(1);
  }
  auto enemy_created{-1};
  auto enemy_creation{[&](const GameObjectID enemy_id) { enemy_created = enemy_id; }};
  game_engine.get_registry().add_callback<EventType::GameObjectCreation>(enemy_creation);
  game_engine.on_update(1);
  ASSERT_EQ(enemy_created, -1);
}

/// Test that the game engine processes a fixed update correctly.
TEST_F(GameEngineFixture, TestGameEngineOnFixedUpdateDeletePlayer) {
  game_engine.create_game_objects();
  game_engine.get_registry().mark_for_deletion(game_engine.get_player_id());
  game_engine.on_fixed_update(0.0);
  ASSERT_FALSE(game_engine.get_registry().has_game_object(game_engine.get_player_id()));
}

/// Test that the game engine processes a 'W' key press correctly.
TEST_F(GameEngineFixture, TestGameEngineOnKeyPressW) {
  game_engine.create_game_objects();
  game_engine.on_key_press(KEY_W, 0);
  const auto player_movement{game_engine.get_registry().get_component<KeyboardMovement>(game_engine.get_player_id())};
  ASSERT_TRUE(player_movement->moving_north);
}

/// Test that the game engine processes an 'A' key press correctly.
TEST_F(GameEngineFixture, TestGameEngineOnKeyPressA) {
  game_engine.create_game_objects();
  game_engine.on_key_press(KEY_A, 0);
  const auto player_movement{game_engine.get_registry().get_component<KeyboardMovement>(game_engine.get_player_id())};
  ASSERT_TRUE(player_movement->moving_west);
}

/// Test that the game engine processes an 'S' key press correctly.
TEST_F(GameEngineFixture, TestGameEngineOnKeyPressS) {
  game_engine.create_game_objects();
  game_engine.on_key_press(KEY_S, 0);
  const auto player_movement{game_engine.get_registry().get_component<KeyboardMovement>(game_engine.get_player_id())};
  ASSERT_TRUE(player_movement->moving_south);
}

/// Test that the game engine processes a 'D' key press correctly.
TEST_F(GameEngineFixture, TestGameEngineOnKeyPressD) {
  game_engine.create_game_objects();
  game_engine.on_key_press(KEY_D, 0);
  const auto player_movement{game_engine.get_registry().get_component<KeyboardMovement>(game_engine.get_player_id())};
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
  const auto player_movement{game_engine.get_registry().get_component<KeyboardMovement>(game_engine.get_player_id())};
  ASSERT_FALSE(player_movement->moving_north);
}

/// Test that the game engine processes an 'A' key release correctly.
TEST_F(GameEngineFixture, TestGameEngineOnKeyReleaseA) {
  game_engine.create_game_objects();
  game_engine.on_key_press(KEY_A, 0);
  game_engine.on_key_release(KEY_A, 0);
  const auto player_movement{game_engine.get_registry().get_component<KeyboardMovement>(game_engine.get_player_id())};
  ASSERT_FALSE(player_movement->moving_west);
}

/// Test that the game engine processes an 'S' key release correctly.
TEST_F(GameEngineFixture, TestGameEngineOnKeyReleaseS) {
  game_engine.create_game_objects();
  game_engine.on_key_press(KEY_S, 0);
  game_engine.on_key_release(KEY_S, 0);
  const auto player_movement{game_engine.get_registry().get_component<KeyboardMovement>(game_engine.get_player_id())};
  ASSERT_FALSE(player_movement->moving_south);
}

/// Test that the game engine processes a 'D' key release correctly.
TEST_F(GameEngineFixture, TestGameEngineOnKeyReleaseD) {
  game_engine.create_game_objects();
  game_engine.on_key_press(KEY_D, 0);
  game_engine.on_key_release(KEY_D, 0);
  const auto player_movement{game_engine.get_registry().get_component<KeyboardMovement>(game_engine.get_player_id())};
  ASSERT_FALSE(player_movement->moving_east);
}

/// Test that the game engine processes a 'C' key release correctly.
TEST_F(GameEngineFixture, TestGameEngineOnKeyReleaseC) {
  game_engine.create_game_objects();
  const auto item_id{get_item(GameObjectType::HealthPotion)};
  move_player_to_item(item_id);
  game_engine.on_update(0);
  game_engine.on_key_release(KEY_C, 0);
  const auto inventory{game_engine.get_registry().get_component<Inventory>(game_engine.get_player_id())};
  ASSERT_EQ(inventory->items.front(), item_id);
}

/// Test that the game engine processes an 'E' key release correctly.
TEST_F(GameEngineFixture, TestGameEngineOnKeyReleaseE) {
  game_engine.create_game_objects();
  game_engine.on_update(1);
  move_player_to_item(get_item(GameObjectType::HealthPotion));
  game_engine.on_update(0);
  const auto health{game_engine.get_registry().get_component<Health>(game_engine.get_player_id())};
  health->set_value(50);
  game_engine.on_key_release(KEY_E, 0);
  ASSERT_EQ(health->get_value(), 55);
  ASSERT_FALSE(game_engine.get_registry().has_game_object(game_engine.get_nearest_item()));
}

/// Test that the game engine processes a 'Z' key release correctly.
TEST_F(GameEngineFixture, TestGameEngineOnKeyReleaseZ) {
  game_engine.create_game_objects();
  game_engine.on_update(1);
  game_engine.on_key_release(KEY_X, 0);
  game_engine.on_key_release(KEY_Z, 0);
  ASSERT_EQ(game_engine.get_registry().get_component<Attack>(game_engine.get_player_id())->selected_ranged_attack, 0);
}

/// Test that the game engine processes a 'X' key release correctly.
TEST_F(GameEngineFixture, TestGameEngineOnKeyReleaseX) {
  game_engine.create_game_objects();
  game_engine.on_update(1);
  game_engine.on_key_release(KEY_X, 0);
  ASSERT_EQ(game_engine.get_registry().get_component<Attack>(game_engine.get_player_id())->selected_ranged_attack, 1);
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
  game_engine.on_update(1);
  game_engine.get_registry().get_system<AttackSystem>()->update(10);

  // Test that processing the left mouse press works correctly
  int called{-1};
  game_engine.get_registry().add_callback<EventType::GameObjectCreation>(
      [&called](const GameObjectID event) { called = event; });
  ASSERT_TRUE(game_engine.on_mouse_press(0, 0, MOUSE_BUTTON_LEFT, 0));
  ASSERT_NE(called, -1);
}

/// Test that the game engine processes a unknown mouse press correctly.
TEST_F(GameEngineFixture, TestGameEngineOnMousePressUnknown) {
  game_engine.create_game_objects();
  ASSERT_FALSE(game_engine.on_mouse_press(0, 0, -1, 0));
}

/// Test that a game object's effects are applied correctly when an item is used.
TEST_F(GameEngineFixture, TestGameEngineUseItemEffects) {
  game_engine.create_game_objects();
  const auto item_id{get_item(GameObjectType::HealthPotion)};
  const auto health{game_engine.get_registry().get_component<Health>(game_engine.get_player_id())};
  health->set_value(50);
  game_engine.use_item(game_engine.get_player_id(), item_id);
  ASSERT_EQ(health->get_value(), 55);
  ASSERT_FALSE(game_engine.get_registry().has_game_object(item_id));
}

/// Test that a game object is removed from the inventory after it is used.
TEST_F(GameEngineFixture, TestGameEngineUseItemRemoveFromInventory) {
  game_engine.create_game_objects();
  const auto item_id{get_item(GameObjectType::HealthPotion)};
  game_engine.get_registry().get_system<InventorySystem>()->add_item_to_inventory(game_engine.get_player_id(), item_id);
  game_engine.get_registry().get_component<Health>(game_engine.get_player_id())->set_value(50);
  game_engine.use_item(game_engine.get_player_id(), item_id);
  ASSERT_FALSE(game_engine.get_registry().has_game_object(item_id));
  ASSERT_FALSE(game_engine.get_registry().get_system<InventorySystem>()->has_item_in_inventory(
      game_engine.get_player_id(), item_id));
}

/// Test that an item is not used if it doesn't match any of the strategies.
TEST_F(GameEngineFixture, TestGameEngineUseItemNoEffect) {
  game_engine.create_game_objects();
  game_engine.use_item(game_engine.get_player_id(), get_item(GameObjectType::Wall));
  ASSERT_EQ(game_engine.get_registry().get_component<Health>(game_engine.get_player_id())->get_value(), 200);
}

/// Test that nothing happens if the item game object does not exist.
TEST_F(GameEngineFixture, TestGameEngineUseItemInvalidItemID) {
  game_engine.create_game_objects();
  game_engine.use_item(game_engine.get_player_id(), -1);
  ASSERT_EQ(game_engine.get_registry().get_component<Health>(game_engine.get_player_id())->get_value(), 200);
}

/// Test that an exception is thrown if the target game object does not have the required components.
TEST_F(GameEngineFixture, TestGameEngineUseItemEffectsInvalidTarget) {
  game_engine.create_game_objects();
  ASSERT_THROW_MESSAGE(
      game_engine.use_item(-1, get_item(GameObjectType::HealthPotion)), RegistryError,
      "The component `StatusEffects` for the game object ID `-1` is not registered with the registry.");
}
