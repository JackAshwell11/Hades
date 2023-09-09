// External includes
#include "gtest/gtest.h"

// Custom includes
#include "fixtures.hpp"
#include "game_objects/components.hpp"

// ----- TESTS ------------------------------
TEST_F(GameObjectsFixtures, TestGameObjectAttributeSetterHigher) {
  // Test that a game object attribute is set with a higher value correctly
  test_game_object_attribute.value(200);
  ASSERT_EQ(test_game_object_attribute.value(), 150);
}

TEST_F(GameObjectsFixtures, TestGameObjectAttributeSetterLower) {
  // Test that a game object attribute is set with a lower value correctly
  test_game_object_attribute.value(100);
  ASSERT_EQ(test_game_object_attribute.value(), 100);
}

TEST_F(GameObjectsFixtures, TestGameObjectAttributeSetterIadd) {
  // Test that adding a value to the game object attribute is correct
  test_game_object_attribute.value(test_game_object_attribute.value() + 100);
  ASSERT_EQ(test_game_object_attribute.value(), 150);
}

TEST_F(GameObjectsFixtures, TestGameObjectAttributeSetterIsub) {
  // Test that subtracting a value from the game object attribute is correct
  test_game_object_attribute.value(test_game_object_attribute.value() - 200);
  ASSERT_EQ(test_game_object_attribute.value(), 0);
}

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
