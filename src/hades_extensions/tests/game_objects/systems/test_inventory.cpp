// Local headers
#include "game_objects/systems/inventory.hpp"
#include "macros.hpp"

// ----- FIXTURES ------------------------------
/// Implements the fixture for the InventorySystem fixture.
class InventorySystemFixture : public testing::Test {
 protected:
  /// The registry that manages the game objects, components, and systems.
  Registry registry;

  /// Set up the fixture for the tests.
  void SetUp() override {
    registry.create_game_object(cpvzero, {std::make_shared<Inventory>(3, 6)});
    registry.add_system<InventorySystem>();
  }

  /// Get the inventory system from the registry.
  ///
  /// @return The inventory system.
  [[nodiscard]] auto get_inventory_system() const -> std::shared_ptr<InventorySystem> {
    return registry.get_system<InventorySystem>();
  }
};

// ----- TESTS ----------------------------------
/// Test that the required components return the correct value for has_indicator_bar.
TEST(Tests, TestInventorySystemComponentsHasIndicatorBar) { ASSERT_FALSE(Inventory(-1, -1).has_indicator_bar()); }

/// Test that a valid item is added to the inventory correctly.
TEST_F(InventorySystemFixture, TestInventorySystemAddItemToInventoryValid) {
  ASSERT_TRUE(get_inventory_system()->add_item_to_inventory(0, 50));
  ASSERT_EQ(registry.get_component<Inventory>(0)->items, std::vector{50});
}

/// Test that a valid item is not added to a zero size inventory.
TEST_F(InventorySystemFixture, TestInventorySystemAddItemToInventoryZeroSize) {
  registry.get_component<Inventory>(0)->width = 0;
  ASSERT_THROW_MESSAGE(get_inventory_system()->add_item_to_inventory(0, 50), std::runtime_error,
                       "The inventory is full.")
}

/// Test that an exception is thrown if the game object does not have an inventory component.
TEST_F(InventorySystemFixture, TestInventorySystemAddItemToInventoryNonexistentComponent) {
  registry.create_game_object(cpvzero, {});
  ASSERT_THROW_MESSAGE(
      get_inventory_system()->add_item_to_inventory(1, 0), RegistryError,
      "The game object `1` is not registered with the registry or does not have the required component.")
}

/// Test that an exception is thrown if an invalid game object ID is provided.
TEST_F(InventorySystemFixture, TestInventorySystemAddItemToInventoryInvalidGameObjectID){ASSERT_THROW_MESSAGE(
    get_inventory_system()->add_item_to_inventory(-1, 50), RegistryError,
    "The game object `-1` is not registered with the registry or does not have the required component.")}

/// Test that a valid item is removed from the inventory correctly.
TEST_F(InventorySystemFixture, TestInventorySystemRemoveItemFromInventoryValid) {
  const std::vector result{1, 4};
  ASSERT_TRUE(get_inventory_system()->add_item_to_inventory(0, 1));
  ASSERT_TRUE(get_inventory_system()->add_item_to_inventory(0, 7));
  ASSERT_TRUE(get_inventory_system()->add_item_to_inventory(0, 4));
  ASSERT_EQ(get_inventory_system()->remove_item_from_inventory(0, 1), 7);
  ASSERT_EQ(registry.get_component<Inventory>(0)->items, result);
}

/// Test that an exception is thrown if a larger index is provided.
TEST_F(InventorySystemFixture, TestInventorySystemRemoveItemFromInventoryLargeIndex) {
  ASSERT_TRUE(get_inventory_system()->add_item_to_inventory(0, 5));
  ASSERT_TRUE(get_inventory_system()->add_item_to_inventory(0, 10));
  ASSERT_TRUE(get_inventory_system()->add_item_to_inventory(0, 50));
  ASSERT_THROW_MESSAGE((get_inventory_system()->remove_item_from_inventory(0, 10)), std::runtime_error,
                       "The index is out of range.")
}

/// Test that an exception is thrown if the game object does not have an inventory component.
TEST_F(InventorySystemFixture, TestInventorySystemRemoveItemFromInventoryNonexistentComponent) {
  registry.create_game_object(cpvzero, {});
  ASSERT_THROW_MESSAGE(
      (get_inventory_system()->remove_item_from_inventory(1, 0)), RegistryError,
      "The game object `1` is not registered with the registry or does not have the required component.")
}

/// Test that an exception is thrown if an invalid game object ID is provided.
TEST_F(InventorySystemFixture, TestInventorySystemRemoveItemFromInventoryInvalidGameObjectID) {
  ASSERT_THROW_MESSAGE(
      (get_inventory_system()->remove_item_from_inventory(-1, 0)), RegistryError,
      "The game object `-1` is not registered with the registry or does not have the required component.")
}
