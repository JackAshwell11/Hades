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
