// External includes
#include "gtest/gtest.h"

// Custom includes
#include "macros.hpp"
#include "game_objects/systems/inventory.hpp"

// ----- FIXTURES ------------------------------
/// Implements the Inventory fixture for the game_objects/systems/inventory.hpp tests.
class InventoryFixture : public testing::Test {
 protected:
  /// The registry that manages the game objects, components, and systems.
  Registry registry{};

  /// Set up the fixture for the tests.
  void SetUp() override {
    std::vector<std::unique_ptr<ComponentBase>> components;
    components.push_back(std::make_unique<Inventory>(3, 6));
    registry.create_game_object(false, std::move(components));
    registry.add_system(std::make_shared<InventorySystem>(registry));
  }

  /// Get the inventory system from the registry.
  ///
  /// @return The inventory system.
  std::shared_ptr<InventorySystem> get_inventory_system() {
    return registry.find_system<InventorySystem>();
  }
};

// ----- TESTS ----------------------------------
/// Test that InventorySpaceError is raised correctly when full.
TEST(Tests, TestThrowInventorySpaceErrorFull) {
  ASSERT_THROW_MESSAGE(throw InventorySpaceException(true), InventorySpaceException, "The inventory is full.")
}

/// Test that InventorySpaceError is raised correctly when empty.
TEST(Tests, TestThrowInventorySpaceErrorEmpty) {
  ASSERT_THROW_MESSAGE(throw InventorySpaceException(false), InventorySpaceException, "The inventory is empty.")
}

/// Test that a valid item is added to the inventory correctly.
TEST_F(InventoryFixture, TestInventorySystemAddItemToInventoryValid) {
  get_inventory_system()->add_item_to_inventory(0, 50);
  ASSERT_EQ(registry.get_component<Inventory>(0)->items, std::vector<int>{50});
}

/// Test that a valid item is not added to a zero size inventory.
TEST_F(InventoryFixture, TestInventorySystemAddItemToInventoryZeroSize) {
  registry.get_component<Inventory>(0)->width = 0;
  ASSERT_THROW_MESSAGE(get_inventory_system()->add_item_to_inventory(0, 50),
                       InventorySpaceException,
                       "The inventory is full.")
}

/// Test that an exception is raised if an invalid game object ID is provided.
TEST_F(InventoryFixture, TestInventorySystemAddItemToInventoryInvalidGameObjectID) {
  ASSERT_THROW_MESSAGE(get_inventory_system()->add_item_to_inventory(-1, 50),
                       RegistryException,
                       "The game object `-1` is not registered with the registry.")
}

/// Test that a valid item is removed from the inventory correctly.
TEST_F(InventoryFixture, TestInventorySystemRemoveItemFromInventoryValid) {
  std::vector<int> result{1, 4};
  get_inventory_system()->add_item_to_inventory(0, 1);
  get_inventory_system()->add_item_to_inventory(0, 7);
  get_inventory_system()->add_item_to_inventory(0, 4);
  ASSERT_EQ(get_inventory_system()->remove_item_from_inventory(0, 1), 7);
  ASSERT_EQ(registry.get_component<Inventory>(0)->items, result);
}

/// Test that an exception is raised if a larger index is provided.
TEST_F(InventoryFixture, TestInventorySystemRemoveItemFromInventoryLargeIndex) {
  get_inventory_system()->add_item_to_inventory(0, 5);
  get_inventory_system()->add_item_to_inventory(0, 10);
  get_inventory_system()->add_item_to_inventory(0, 50);
  ASSERT_THROW_MESSAGE(get_inventory_system()->remove_item_from_inventory(0, 10),
                       InventorySpaceException,
                       "The index is out of range.")
}

/// Test that an exception is raised if an invalid game object ID is provided.
TEST_F(InventoryFixture, TestInventorySystemRemoveItemFromInventoryInvalidGameObjectID) {
  ASSERT_THROW_MESSAGE(get_inventory_system()->remove_item_from_inventory(-1, 0),
                       RegistryException,
                       "The game object `-1` is not registered with the registry.")
}
