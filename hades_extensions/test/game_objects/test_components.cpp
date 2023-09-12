// External includes
#include "gtest/gtest.h"

// Custom includes
#include "game_objects/components.hpp"

// ----- CLASSES ------------------------------
/// Represents a game object attribute useful for testing.
class TestGameObjectAttribute : public GameObjectAttributeBase {
 public:
  /// Initialise the object.
  ///
  /// @param initial_value - The initial value of the movement force attribute.
  /// @param level_limit - The level limit of the movement force attribute.
  TestGameObjectAttribute(float initial_value, int level_limit) : GameObjectAttributeBase(initial_value, level_limit) {}
};

// ----- FIXTURES ------------------------------
/// A test fixture for the game_objects/components.hpp tests.
class ComponentsFixtures : public testing::Test {
 protected:
  /// A test game object attribute.
  TestGameObjectAttribute test_game_object_attribute{150, 3};
};

// ----- TESTS ------------------------------
/// Test that a game object attribute is set with a higher value correctly.
TEST_F(ComponentsFixtures, TestGameObjectAttributeSetterHigher) {
  test_game_object_attribute.value(200);
  ASSERT_EQ(test_game_object_attribute.value(), 150);
}

/// Test that a game object attribute is set with a lower value correctly.
TEST_F(ComponentsFixtures, TestGameObjectAttributeSetterLower) {
  test_game_object_attribute.value(100);
  ASSERT_EQ(test_game_object_attribute.value(), 100);
}

/// Test that adding a value to the game object attribute is correct.
TEST_F(ComponentsFixtures, TestGameObjectAttributeSetterAdd) {
  test_game_object_attribute.value(test_game_object_attribute.value() + 100);
  ASSERT_EQ(test_game_object_attribute.value(), 150);
}

/// Test that subtracting a value from the game object attribute is correct.
TEST_F(ComponentsFixtures, TestGameObjectAttributeSetterSub) {
  test_game_object_attribute.value(test_game_object_attribute.value() - 200);
  ASSERT_EQ(test_game_object_attribute.value(), 0);
}

/// Test that the inventory capacity calculation is correct.
TEST(Tests, TestInventoryCapacity) {
  // Test if an inventory with zero width and height has a zero capacity
  Inventory inventory_zero_capacity{0, 0};
  ASSERT_EQ(inventory_zero_capacity.capacity(), 0);

  // Test if an inventory with zero width and non-zero height has a zero
  // capacity
  Inventory inventory_zero_width{0, 10};
  ASSERT_EQ(inventory_zero_width.capacity(), 0);

  // Test if an inventory with non-zero width and zero height has a zero
  // capacity
  Inventory inventory_zero_height{10, 0};
  ASSERT_EQ(inventory_zero_height.capacity(), 0);

  // Test if an inventory with non-zero width and non-zero height has a
  // non-zero capacity
  Inventory inventory_non_zero{5, 10};
  ASSERT_EQ(inventory_non_zero.capacity(), 50);
}
