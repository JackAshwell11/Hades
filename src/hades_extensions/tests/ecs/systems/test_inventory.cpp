// Local headers
#include "ecs/registry.hpp"
#include "ecs/stats.hpp"
#include "ecs/systems/effects.hpp"
#include "ecs/systems/inventory.hpp"
#include "macros.hpp"

// ----- FIXTURES ------------------------------
/// Implements the fixture for the InventorySystem fixture.
class InventorySystemFixture : public testing::Test {
 protected:
  /// The registry that manages the game objects, components, and systems.
  Registry registry;

  /// Set up the fixture for the tests.
  void SetUp() override {
    registry.create_game_object(GameObjectType::Player, cpvzero,
                                {std::make_shared<Health>(200, -1), std::make_shared<Inventory>(),
                                 std::make_shared<InventorySize>(8, -1), std::make_shared<StatusEffect>()});
    registry.add_system<InventorySystem>();
    registry.add_system<EffectSystem>();
  }

  /// Get the inventory system from the registry.
  ///
  /// @return The inventory system.
  [[nodiscard]] auto get_inventory_system() const -> std::shared_ptr<InventorySystem> {
    return registry.get_system<InventorySystem>();
  }

  /// Create a status effect item.
  ///
  /// @return The game object ID of the status effect item.
  [[nodiscard]] auto create_status_effect_item() -> GameObjectID {
    return registry.create_game_object(GameObjectType::Player, cpvzero,
                                       {std::make_shared<EffectApplier>(
                                            std::unordered_map<std::type_index, ActionFunction>{
                                                {typeid(Health), [](const int level) { return level * 3 + 5; }}},
                                            std::unordered_map<std::type_index, StatusEffectData>{}),
                                        std::make_shared<EffectLevel>(1, -1)});
  }
};

// ----- TESTS ----------------------------------
/// Test that a valid item is added to the inventory correctly.
TEST_F(InventorySystemFixture, TestInventorySystemAddItemToInventoryValid) {
  bool inventory_update{false};
  auto inventory_update_callback{[&](const GameObjectID /*game_object_id*/) { inventory_update = true; }};
  registry.add_callback(EventType::InventoryUpdate, inventory_update_callback);
  ASSERT_TRUE(get_inventory_system()->add_item_to_inventory(0, 50));
  ASSERT_EQ(registry.get_component<Inventory>(0)->items, std::vector{50});
  ASSERT_TRUE(inventory_update);
}

/// Test that a valid item is not added to a zero size inventory.
TEST_F(InventorySystemFixture, TestInventorySystemAddItemToInventoryZeroSize) {
  registry.get_component<InventorySize>(0)->set_value(0);
  ASSERT_THROW_MESSAGE(get_inventory_system()->add_item_to_inventory(0, 50), std::runtime_error,
                       "The inventory is full.")
}

/// Test that an exception is thrown if the game object does not have an inventory component.
TEST_F(InventorySystemFixture, TestInventorySystemAddItemToInventoryNonexistentComponent) {
  registry.create_game_object(GameObjectType::Player, cpvzero, {});
  ASSERT_THROW_MESSAGE(get_inventory_system()->add_item_to_inventory(1, 0), RegistryError,
                       "The component `Inventory` for the game object ID `1` is not registered with the registry.")
}

/// Test that an exception is thrown if an invalid game object ID is provided.
TEST_F(InventorySystemFixture, TestInventorySystemAddItemToInventoryInvalidGameObjectID){
    ASSERT_THROW_MESSAGE(get_inventory_system()->add_item_to_inventory(-1, 50), RegistryError,
                         "The component `Inventory` for the game object ID `-1` is not registered with the registry.")}

/// Test that a valid item is removed from the inventory correctly.
TEST_F(InventorySystemFixture, TestInventorySystemRemoveItemFromInventoryValid) {
  auto inventory_update{false};
  auto inventory_update_callback{[&](const GameObjectID /*game_object_id*/) { inventory_update = true; }};
  registry.add_callback(EventType::InventoryUpdate, inventory_update_callback);
  const auto item_id{registry.create_game_object(GameObjectType::Potion, cpvzero, {})};
  const std::vector result{1, 4};
  ASSERT_TRUE(get_inventory_system()->add_item_to_inventory(0, 1));
  ASSERT_TRUE(get_inventory_system()->add_item_to_inventory(0, item_id));
  ASSERT_TRUE(get_inventory_system()->add_item_to_inventory(0, 4));
  ASSERT_TRUE(get_inventory_system()->remove_item_from_inventory(0, item_id));
  ASSERT_EQ(registry.get_component<Inventory>(0)->items, result);
  ASSERT_TRUE(inventory_update);
}

/// Test that an item is not removed from the inventory if it is not added.
TEST_F(InventorySystemFixture, TestInventorySystemRemoveItemFromInventoryLargeIndex) {
  ASSERT_TRUE(get_inventory_system()->add_item_to_inventory(0, 5));
  ASSERT_TRUE(get_inventory_system()->add_item_to_inventory(0, 10));
  ASSERT_TRUE(get_inventory_system()->add_item_to_inventory(0, 50));
  ASSERT_FALSE(get_inventory_system()->remove_item_from_inventory(0, 100));
}

/// Test that an exception is thrown if the game object does not have an inventory component.
TEST_F(InventorySystemFixture, TestInventorySystemRemoveItemFromInventoryNonexistentComponent) {
  registry.create_game_object(GameObjectType::Player, cpvzero, {});
  ASSERT_THROW_MESSAGE((get_inventory_system()->remove_item_from_inventory(1, 0)), RegistryError,
                       "The component `Inventory` for the game object ID `1` is not registered with the registry.")
}

/// Test that an exception is thrown if an invalid game object ID is provided.
TEST_F(InventorySystemFixture, TestInventorySystemRemoveItemFromInventoryInvalidGameObjectID){
    ASSERT_THROW_MESSAGE((get_inventory_system()->remove_item_from_inventory(-1, 0)), RegistryError,
                         "The component `Inventory` for the game object ID `-1` is not registered with the registry.")}

/// Test that a status effect item is used correctly if the target can be affected.
TEST_F(InventorySystemFixture, TestInventorySystemUseItemStatusEffectValid) {
  const auto item_id{create_status_effect_item()};
  registry.get_component<Health>(0)->set_value(50);
  ASSERT_TRUE(get_inventory_system()->add_item_to_inventory(0, item_id));
  ASSERT_TRUE(get_inventory_system()->use_item(0, item_id));
  ASSERT_EQ(registry.get_component<Health>(0)->get_value(), 58);
  ASSERT_EQ(registry.get_component<Inventory>(0)->items.size(), 0);
}

/// Test that a status effect item is not used if the target cannot be affected.
TEST_F(InventorySystemFixture, TestInventorySystemUseItemStatusEffectInvalid) {
  const auto item_id{create_status_effect_item()};
  ASSERT_FALSE(get_inventory_system()->use_item(0, item_id));
  ASSERT_EQ(registry.get_component<Health>(0)->get_value(), 200);
}

/// Test that an item is not used if it doesn't match any of the strategies.
TEST_F(InventorySystemFixture, TestInventorySystemUseItemNoEffect) {
  const auto item_id{registry.create_game_object(GameObjectType::Potion, cpvzero, {})};
  ASSERT_FALSE(get_inventory_system()->use_item(0, item_id));
}

/// Test that nothing happens if the item game object does not exist.
TEST_F(InventorySystemFixture, TestInventorySystemUseItemStatusEffectInvalidItemID) {
  ASSERT_FALSE(get_inventory_system()->use_item(0, -1));
}

/// Test that an exception is thrown if the game object does not have the required components.
TEST_F(InventorySystemFixture, TestInventorySystemUseItemStatusEffectInvalidGameObjectID) {
  const auto item_id{create_status_effect_item()};
  ASSERT_THROW_MESSAGE(get_inventory_system()->use_item(-1, item_id), RegistryError,
                       "The component `StatusEffect` for the game object ID `-1` is not registered with the registry.")
}
