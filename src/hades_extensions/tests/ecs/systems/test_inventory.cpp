// External headers
#include <nlohmann/json.hpp>

// Local headers
#include "ecs/registry.hpp"
#include "ecs/systems/effects.hpp"
#include "ecs/systems/inventory.hpp"
#include "ecs/systems/physics.hpp"
#include "events.hpp"
#include "macros.hpp"

/// Implements the fixture for the InventorySystem fixture.
class InventorySystemFixture : public testing::Test {
 protected:
  /// The configuration for creating an item
  struct ItemConfig {
    /// Whether the item is kinematic or not.
    bool kinematic = false;

    /// Whether the item is an effect applier or not.
    bool effect_applier = false;
  };

  /// The registry that manages the game objects, components, and systems.
  Registry registry;

  /// Set up the fixture for the tests.
  void SetUp() override {
    registry.create_game_object(GameObjectType::Player, cpvzero,
                                {std::make_shared<Health>(200, -1), std::make_shared<Inventory>(),
                                 std::make_shared<InventorySize>(8, -1), std::make_shared<StatusEffects>()});
    registry.add_system<EffectSystem>();
    registry.add_system<InventorySystem>();
  }

  /// Tear down the fixture after the tests.
  void TearDown() override { clear_listeners(); }

  /// Create an item with the specified type.
  ///
  /// @param type - The type of the game object.
  /// @param config - The configuration for the item.
  /// @return The game object ID of the item.
  [[nodiscard]] auto create_item(const GameObjectType type, const ItemConfig &config) -> GameObjectID {
    std::vector<std::shared_ptr<ComponentBase>> components;
    if (config.kinematic) {
      components.push_back(std::make_shared<KinematicComponent>());
    }
    if (config.effect_applier) {
      const auto effect_applier{std::make_shared<EffectApplier>()};
      effect_applier->add_instant_effect(EffectType::Regeneration, 5);
      components.push_back(std::move(effect_applier));
    }
    return registry.create_game_object(type, cpvzero, std::move(components));
  }

  /// Get the inventory system from the registry.
  ///
  /// @return The inventory system.
  [[nodiscard]] auto get_inventory_system() const -> std::shared_ptr<InventorySystem> {
    return registry.get_system<InventorySystem>();
  }
};

/// Test that the inventory component is reset correctly.
TEST_F(InventorySystemFixture, TestInventoryReset) {
  const auto inventory{registry.get_component<Inventory>(0)};
  inventory->items = {1, 2, 3};
  inventory->reset();
  ASSERT_TRUE(inventory->items.empty());
}

/// Test that the inventory component is serialised to a file correctly.
TEST_F(InventorySystemFixture, TestInventoryToFile) {
  registry.get_system<InventorySystem>()->add_item_to_inventory(0, create_item(GameObjectType::HealthPotion, {}));
  registry.get_system<InventorySystem>()->add_item_to_inventory(0, create_item(GameObjectType::HealthPotion, {}));
  registry.get_system<InventorySystem>()->add_item_to_inventory(0, create_item(GameObjectType::HealthPotion, {}));
  nlohmann::json json;
  registry.get_component<Inventory>(0)->to_file(json, &registry);
  ASSERT_EQ(json.at("items"), nlohmann::json::array({128, 128, 128}));
}

/// Test that the inventory component is deserialised from a file correctly.
TEST_F(InventorySystemFixture, TestInventoryFromFile) {
  const nlohmann::json json(nlohmann::json::parse(R"({"items":[128, 128]})"));
  const auto inventory{registry.get_component<Inventory>(0)};
  inventory->from_file(json, &registry);
  ASSERT_EQ(inventory->items.size(), 2);
  ASSERT_EQ(inventory->items[0], 1);
  ASSERT_EQ(registry.get_game_object_type(inventory->items[0]), GameObjectType::HealthPotion);
  ASSERT_EQ(inventory->items[1], 2);
  ASSERT_EQ(registry.get_game_object_type(inventory->items[1]), GameObjectType::HealthPotion);
}

/// Test that the inventory size component is serialised to a file correctly.
TEST_F(InventorySystemFixture, TestInventorySizeToFile) {
  nlohmann::json json;
  registry.get_component<InventorySize>(0)->to_file(json);
  ASSERT_EQ(json, nlohmann::json::parse(
                      R"({"inventory_size":{"current_level":0,"max_level":-1,"max_value":8.0,"value":8.0}})"));
}

/// Test that the inventory size component is deserialised from a file correctly.
TEST_F(InventorySystemFixture, TestInventorySizeFromFile) {
  const nlohmann::json json(
      nlohmann::json::parse(R"({"inventory_size":{"current_level":0,"max_level":10,"max_value":5.0,"value":5.0}})"));
  const auto inventory_size{std::make_shared<InventorySize>(0, -1)};
  inventory_size->from_file(json);
  ASSERT_EQ(inventory_size->get_value(), 5);
  ASSERT_EQ(inventory_size->get_max_value(), 5);
  ASSERT_EQ(inventory_size->get_max_level(), 10);
  ASSERT_EQ(inventory_size->get_current_level(), 0);
}

/// Test that an item does not exist in the inventory if it has not been added.
TEST_F(InventorySystemFixture, TestInventorySystemHasItemInInventoryNotAdded) {
  const auto item_id{create_item(GameObjectType::HealthPotion, {})};
  ASSERT_FALSE(get_inventory_system()->has_item_in_inventory(0, item_id));
}

/// Test that an item exists in the inventory if it has been added.
TEST_F(InventorySystemFixture, TestInventorySystemHasItemInInventoryAdded) {
  const auto item_id{create_item(GameObjectType::HealthPotion, {})};
  get_inventory_system()->add_item_to_inventory(0, item_id);
  ASSERT_TRUE(get_inventory_system()->has_item_in_inventory(0, item_id));
}

/// Test that multiple items exist in the inventory if they have been added.
TEST_F(InventorySystemFixture, TestInventorySystemHasItemInInventoryMultipleAdded) {
  const auto item_id_one{create_item(GameObjectType::HealthPotion, {})};
  const auto item_id_two{create_item(GameObjectType::HealthPotion, {})};
  get_inventory_system()->add_item_to_inventory(0, item_id_one);
  get_inventory_system()->add_item_to_inventory(0, item_id_two);
  ASSERT_TRUE(get_inventory_system()->has_item_in_inventory(0, item_id_one));
  ASSERT_TRUE(get_inventory_system()->has_item_in_inventory(0, item_id_two));
}

/// Test that an item does not exist in the inventory if it is not a valid game object.
TEST_F(InventorySystemFixture, TestInventorySystemHasItemInInventoryInvalidItem) {
  ASSERT_FALSE(get_inventory_system()->has_item_in_inventory(0, -1));
}

/// Test that an item does not exist in the inventory if it is not a collectible item.
TEST_F(InventorySystemFixture, TestInventorySystemHasItemInInventoryInvalidType) {
  const auto item_id{create_item(GameObjectType::Player, {})};
  ASSERT_FALSE(get_inventory_system()->has_item_in_inventory(0, item_id));
}

/// Test that an item does not exist in the inventory if the inventory component is not registered.
TEST_F(InventorySystemFixture, TestInventorySystemHasItemInInventoryInvalidGameObjectID) {
  ASSERT_THROW_MESSAGE(get_inventory_system()->has_item_in_inventory(-1, 0), RegistryError,
                       "The component `Inventory` for the game object ID `-1` is not registered with the registry.");
}

/// Test that a valid item is added to the inventory correctly.
TEST_F(InventorySystemFixture, TestInventorySystemAddItemToInventoryValid) {
  // Add the callbacks to the registry
  std::vector<GameObjectID> inventory_update;
  auto inventory_update_callback{[&](const std::vector<GameObjectID> &items) { inventory_update = items; }};
  add_callback<EventType::InventoryUpdate>(inventory_update_callback);
  auto sprite_removal{-1};
  auto sprite_removal_callback{[&](const GameObjectID game_object_id) { sprite_removal = game_object_id; }};
  add_callback<EventType::SpriteRemoval>(sprite_removal_callback);

  // Add the item to the inventory and check the results
  const auto game_object_id{create_item(GameObjectType::HealthPotion, {})};
  get_inventory_system()->add_item_to_inventory(0, game_object_id);
  ASSERT_EQ(registry.get_component<Inventory>(0)->items, std::vector{game_object_id});
  const std::vector expected_items{game_object_id};
  ASSERT_EQ(inventory_update, expected_items);
  ASSERT_EQ(sprite_removal, game_object_id);
}

/// Test that an item with a kinematic component is added to the inventory correctly.
TEST_F(InventorySystemFixture, TestInventorySystemAddItemToInventoryKinematic) {
  const auto game_object_id{create_item(GameObjectType::HealthPotion, {.kinematic = true})};
  get_inventory_system()->add_item_to_inventory(0, game_object_id);
  ASSERT_TRUE(registry.get_component<KinematicComponent>(game_object_id)->collected);
}

/// Test that an item is not added to the inventory if it is not a collectible item.
TEST_F(InventorySystemFixture, TestInventorySystemAddItemToInventoryInvalidType) {
  const auto game_object_id{create_item(GameObjectType::Player, {})};
  get_inventory_system()->add_item_to_inventory(0, game_object_id);
  ASSERT_TRUE(registry.get_component<Inventory>(0)->items.empty());
}

/// Test that an item is not added to the inventory if it is not a valid game object.
TEST_F(InventorySystemFixture, TestInventorySystemAddItemToInventoryInvalidItem) {
  get_inventory_system()->add_item_to_inventory(0, -1);
  ASSERT_TRUE(registry.get_component<Inventory>(0)->items.empty());
}

/// Test that a valid item is not added to a zero size inventory.
TEST_F(InventorySystemFixture, TestInventorySystemAddItemToInventoryZeroSize) {
  const auto game_object_id{create_item(GameObjectType::HealthPotion, {})};
  registry.get_component<InventorySize>(0)->set_value(0);
  ASSERT_THROW_MESSAGE(get_inventory_system()->add_item_to_inventory(0, game_object_id), std::runtime_error,
                       "The inventory is full.")
}

/// Test that a valid item is removed from the inventory correctly.
TEST_F(InventorySystemFixture, TestInventorySystemRemoveItemFromInventoryValid) {
  // Add the callbacks to the registry
  std::vector<GameObjectID> inventory_update;
  auto inventory_update_callback{[&](const std::vector<GameObjectID> &items) { inventory_update = items; }};
  add_callback<EventType::InventoryUpdate>(inventory_update_callback);
  auto sprite_removal{-1};
  auto sprite_removal_callback{[&](const GameObjectID game_object_id) { sprite_removal = game_object_id; }};
  add_callback<EventType::SpriteRemoval>(sprite_removal_callback);

  // Add two items and remove one of them from the inventory and check the results
  const auto item_id_one{create_item(GameObjectType::HealthPotion, {})};
  const auto item_id_two{create_item(GameObjectType::HealthPotion, {})};
  get_inventory_system()->add_item_to_inventory(0, item_id_one);
  get_inventory_system()->add_item_to_inventory(0, item_id_two);
  get_inventory_system()->remove_item_from_inventory(0, item_id_one);
  const std::vector result{item_id_two};
  ASSERT_EQ(registry.get_component<Inventory>(0)->items, result);
  const std::vector expected_items{item_id_two};
  ASSERT_EQ(inventory_update, expected_items);
  ASSERT_EQ(sprite_removal, item_id_two);
}

/// Test that an item is not removed from the inventory if it is not added.
TEST_F(InventorySystemFixture, TestInventorySystemRemoveItemFromInventoryNotAdded) {
  const auto item_id_one{create_item(GameObjectType::HealthPotion, {})};
  const auto item_id_two{create_item(GameObjectType::HealthPotion, {})};
  get_inventory_system()->add_item_to_inventory(0, item_id_one);
  const std::vector result{item_id_one};
  ASSERT_EQ(registry.get_component<Inventory>(0)->items, result);
  get_inventory_system()->remove_item_from_inventory(0, item_id_two);
  ASSERT_EQ(registry.get_component<Inventory>(0)->items, result);
}

/// Test that an item is not removed from the inventory if it is not a valid game object.
TEST_F(InventorySystemFixture, TestInventorySystemRemoveItemFromInventoryInvalidItem) {
  get_inventory_system()->remove_item_from_inventory(0, -1);
  ASSERT_TRUE(registry.get_component<Inventory>(0)->items.empty());
}

/// Test that an exception is thrown if the game object does not have an inventory component.
TEST_F(InventorySystemFixture, TestInventorySystemRemoveItemFromInventoryNonexistentComponent) {
  const auto game_object_id{create_item(GameObjectType::Player, {})};
  ASSERT_THROW_MESSAGE(get_inventory_system()->remove_item_from_inventory(game_object_id, 0), RegistryError,
                       "The component `Inventory` for the game object ID `1` is not registered with the registry.")
}

/// Test that an exception is thrown if an invalid game object ID is provided.
TEST_F(InventorySystemFixture, TestInventorySystemRemoveItemFromInventoryInvalidGameObjectID){
    ASSERT_THROW_MESSAGE(get_inventory_system()->remove_item_from_inventory(-1, 0), RegistryError,
                         "The component `Inventory` for the game object ID `-1` is not registered with the registry.")}

/// Test that a game object's effects are applied correctly when an item is used.
TEST_F(InventorySystemFixture, TestInventorySystemEngineUseItemEffects) {
  const auto item_id{create_item(GameObjectType::HealthPotion, {.effect_applier = true})};
  const auto health{registry.get_component<Health>(0)};
  health->set_value(50);
  get_inventory_system()->use_item(0, item_id);
  ASSERT_EQ(health->get_value(), 55);
  ASSERT_FALSE(registry.has_game_object(item_id));
}

/// Test that a game object is removed from the inventory after it is used.
TEST_F(InventorySystemFixture, TestInventorySystemEngineUseItemRemoveFromInventory) {
  const auto item_id{create_item(GameObjectType::HealthPotion, {.effect_applier = true})};
  registry.get_system<InventorySystem>()->add_item_to_inventory(0, item_id);
  registry.get_component<Health>(0)->set_value(50);
  get_inventory_system()->use_item(0, item_id);
  ASSERT_FALSE(registry.has_game_object(item_id));
  ASSERT_FALSE(registry.get_system<InventorySystem>()->has_item_in_inventory(0, item_id));
}

/// Test that an item is not used if it doesn't match any of the strategies.
TEST_F(InventorySystemFixture, TestInventorySystemEngineUseItemNoEffect) {
  get_inventory_system()->use_item(0, create_item(GameObjectType::Wall, {}));
  ASSERT_EQ(registry.get_component<Health>(0)->get_value(), 200);
}

/// Test that nothing happens if the target game object does not exist.
TEST_F(InventorySystemFixture, TestInventorySystemEngineUseItemEffectsInvalidTarget) {
  get_inventory_system()->use_item(-1, create_item(GameObjectType::HealthPotion, {}));
  ASSERT_TRUE(registry.has_game_object(1));
}

/// Test that nothing happens if the item game object does not exist.
TEST_F(InventorySystemFixture, TestInventorySystemEngineUseItemInvalidItemID) {
  get_inventory_system()->use_item(0, -1);
  ASSERT_EQ(registry.get_component<Health>(0)->get_value(), 200);
}
