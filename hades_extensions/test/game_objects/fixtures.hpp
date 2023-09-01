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
class GameObjectsFixtures : public testing::Test {
  /// Hold fixtures relating to the game_objects/ C++ tests.
 protected:
  TestGameObjectAttribute test_game_object_attribute{150, 3};
};
