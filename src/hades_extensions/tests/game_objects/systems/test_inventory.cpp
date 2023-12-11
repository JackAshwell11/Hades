// Local headers
#include "game_objects/systems/inventory.hpp"
#include "macros.hpp"

// ----- FIXTURES ------------------------------
/// Implements the fixture for the InventorySystem fixture.
class InventorySystemFixture : public testing::Test {
 protected:
  /// The registry that manages the game objects, components, and systems.
  Registry registry{};

  /// Set up the fixture for the tests.
  void SetUp() override {
    registry.create_game_object({std::make_shared<Inventory>(3, 6)});
    registry.add_system<InventorySystem>();
  }

  /// Get the inventory system from the registry.
  ///
  /// @return The inventory system.
  [[nodiscard]] auto get_inventory_system() const -> std::shared_ptr<InventorySystem> {
    return registry.get_system<InventorySystem>();
  }
};

// ----- HELPER FUNCTIONS -----------------------------
/// Throw an InventorySpaceError with a given boolean.
///
/// @param val - The value to throw the error with.
/// @throws InventorySpaceError - Always for testing.
template <typename T>
void throw_inventory_space_error(const T val) {
  throw InventorySpaceError(val);
}

// ----- TESTS ----------------------------------
/// Test that InventorySpaceError is thrown correctly when given a message.
TEST(Tests, TestThrowInventorySpaceErrorMessage){
    ASSERT_THROW_MESSAGE(throw_inventory_space_error("Test message."), InventorySpaceError, "Test message.")}

/// Test that InventorySpaceError is thrown correctly when given an empty message.
TEST(Tests, TestThrowInventorySpaceErrorEmptyMessage){
    ASSERT_THROW_MESSAGE(throw_inventory_space_error(""), InventorySpaceError, "")}

/// Test that InventorySpaceError is thrown correctly when full.
TEST(Tests, TestThrowInventorySpaceErrorFull){
    ASSERT_THROW_MESSAGE(throw_inventory_space_error(true), InventorySpaceError, "The inventory is full.")}

/// Test that InventorySpaceError is thrown correctly when empty.
TEST(Tests, TestThrowInventorySpaceErrorEmpty){
    ASSERT_THROW_MESSAGE(throw_inventory_space_error(false), InventorySpaceError, "The inventory is empty.")}

/// Test that a valid item is added to the inventory correctly.
TEST_F(InventorySystemFixture, TestInventorySystemAddItemToInventoryValid) {
  get_inventory_system()->add_item_to_inventory(0, 50);
  ASSERT_EQ(registry.get_component<Inventory>(0)->items, std::vector{50});
}

/// Test that a valid item is not added to a zero size inventory.
TEST_F(InventorySystemFixture, TestInventorySystemAddItemToInventoryZeroSize) {
  registry.get_component<Inventory>(0)->width = 0;
  ASSERT_THROW_MESSAGE(get_inventory_system()->add_item_to_inventory(0, 50), InventorySpaceError,
                       "The inventory is full.")
}

/// Test that an exception is thrown if an invalid game object ID is provided.
TEST_F(InventorySystemFixture, TestInventorySystemAddItemToInventoryInvalidGameObjectID){ASSERT_THROW_MESSAGE(
    get_inventory_system()->add_item_to_inventory(-1, 50), RegistryError,
    "The game object `-1` is not registered with the registry or does not have the required component.")}

/// Test that a valid item is removed from the inventory correctly.
TEST_F(InventorySystemFixture, TestInventorySystemRemoveItemFromInventoryValid) {
  const std::vector result{1, 4};
  get_inventory_system()->add_item_to_inventory(0, 1);
  get_inventory_system()->add_item_to_inventory(0, 7);
  get_inventory_system()->add_item_to_inventory(0, 4);
  ASSERT_EQ(get_inventory_system()->remove_item_from_inventory(0, 1), 7);
  ASSERT_EQ(registry.get_component<Inventory>(0)->items, result);
}

/// Test that an exception is thrown if a larger index is provided.
TEST_F(InventorySystemFixture, TestInventorySystemRemoveItemFromInventoryLargeIndex) {
  get_inventory_system()->add_item_to_inventory(0, 5);
  get_inventory_system()->add_item_to_inventory(0, 10);
  get_inventory_system()->add_item_to_inventory(0, 50);
  ASSERT_THROW_MESSAGE((get_inventory_system()->remove_item_from_inventory(0, 10)), InventorySpaceError,
                       "The index is out of range.")
}

/// Test that an exception is thrown if an invalid game object ID is provided.
TEST_F(InventorySystemFixture, TestInventorySystemRemoveItemFromInventoryInvalidGameObjectID) {
  ASSERT_THROW_MESSAGE(
      (get_inventory_system()->remove_item_from_inventory(-1, 0)), RegistryError,
      "The game object `-1` is not registered with the registry or does not have the required component.")
}
