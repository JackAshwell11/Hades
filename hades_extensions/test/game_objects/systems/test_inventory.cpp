// External includes
#include "gtest/gtest.h"

// Custom includes
#include "macros.hpp"
#include "game_objects/systems/inventory.hpp"

// TODO: Rename all Fixtures to Fixture

// ----- FIXTURES ------------------------------
/// Implements the fixture for the game_objects/systems/inventory.hpp tests.
class InventoryFixture : public testing::Test {
 protected:
  /// The registry that manages the game objects, components, and systems.
  Registry registry{};

  /// Set up the fixture for the tests.
  void SetUp() override {
    std::vector<std::unique_ptr<ComponentBase>> components;
    components.push_back(std::make_unique<Inventory>(3, 6));
    registry.create_game_object(false, std::move(components));
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
  InventorySystem::add_item_to_inventory(registry, 0, 50);
  ASSERT_EQ(registry.get_component<Inventory>(0)->items, std::vector<int>{50});
}

/// Test that a valid item is not added to a zero size inventory.
TEST_F(InventoryFixture, TestInventorySystemAddItemToInventoryZeroSize) {
  registry.get_component<Inventory>(0)->width = 0;
  ASSERT_THROW_MESSAGE(InventorySystem::add_item_to_inventory(registry, 0, 50),
                       InventorySpaceException,
                       "The inventory is full.")
}

/// Test that an exception is raised if an invalid game object ID is provided.
TEST_F(InventoryFixture, TestInventorySystemAddItemToInventoryInvalidGameObjectID) {
  ASSERT_THROW_MESSAGE(InventorySystem::add_item_to_inventory(registry, -1, 50),
                       RegistryException,
                       "The game object `-1` is not registered with the registry.")
}

/// Test that a valid item is removed from the inventory correctly.
TEST_F(InventoryFixture, TestInventorySystemRemoveItemFromInventoryValid) {
  std::vector<int> result{1, 4};
  InventorySystem::add_item_to_inventory(registry, 0, 1);
  InventorySystem::add_item_to_inventory(registry, 0, 7);
  InventorySystem::add_item_to_inventory(registry, 0, 4);
  ASSERT_EQ(InventorySystem::remove_item_from_inventory(registry, 0, 1), 7);
  ASSERT_EQ(registry.get_component<Inventory>(0)->items, result);
}

/// Test that an exception is raised if a larger index is provided.
TEST_F(InventoryFixture, TestInventorySystemRemoveItemFromInventoryLargeIndex) {
  InventorySystem::add_item_to_inventory(registry, 0, 5);
  InventorySystem::add_item_to_inventory(registry, 0, 10);
  InventorySystem::add_item_to_inventory(registry, 0, 50);
  ASSERT_THROW_MESSAGE(InventorySystem::remove_item_from_inventory(registry, 0, 10),
                       InventorySpaceException,
                       "The index is out of range.")
}

/// Test that an exception is raised if an invalid game object ID is provided.
TEST_F(InventoryFixture, TestInventorySystemRemoveItemFromInventoryInvalidGameObjectID) {
  ASSERT_THROW_MESSAGE(InventorySystem::remove_item_from_inventory(registry, -1, 0),
                       RegistryException,
                       "The game object `-1` is not registered with the registry.")
}

// TODO: Move all fixtures to local files (for independence)
// TODO: Switch to docstrings for explaining tests
