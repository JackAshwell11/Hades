// Local headers
#include "factories.hpp"
#include "game_object.hpp"
#include "macros.hpp"

/// Implements the fixture for the factories.hpp tests.
class FactoriesFixture : public testing::Test {  // NOLINT
 protected:
  /// Tear down the fixture for the tests.
  void TearDown() override { clear_hitboxes(); }
};

/// Test that loading a hitbox for a game object type works.
TEST_F(FactoriesFixture, TestLoadHitboxOnce) { ASSERT_TRUE(load_hitbox(GameObjectType::Player, {{0.0, 0.0}})); }

/// Test that loading two hitboxes for the same game object type doesn't do anything.
TEST_F(FactoriesFixture, TestLoadHitboxTwice) {
  load_hitbox(GameObjectType::Player, {{0.0, 0.0}});
  ASSERT_FALSE(load_hitbox(GameObjectType::Player, {{1.0, 1.0}}));
}

/// Test that getting components for a game object that doesn't require a hitbox works.
TEST_F(FactoriesFixture, TestGetGameObjectComponentsNoHitboxRequired) {
  ASSERT_NO_THROW(get_game_object_components(GameObjectType::Floor));
}

/// Test that getting components for a game object that requires a hitbox works when the hitbox is loaded.
TEST_F(FactoriesFixture, TestGetGameObjectComponentsHitboxLoaded) {
  load_hitbox(GameObjectType::Player, {{0.0, 1.0}, {1.0, 2.0}, {2.0, 0.0}});
  ASSERT_NO_THROW(get_game_object_components(GameObjectType::Player));
}

/// Test that getting components for a game object which doesn't have a loaded hitbox throws an exception.
TEST_F(FactoriesFixture, TestGetFactoriesHitboxNotLoaded) {
  ASSERT_THROW(get_game_object_components(GameObjectType::Player), std::out_of_range);
}

/// Test that getting components for a game object with a level works.
TEST_F(FactoriesFixture, TestGetGameObjectComponentsWithLevel) {
  load_hitbox(GameObjectType::Enemy, {{0.0, 1.0}, {1.0, 2.0}, {2.0, 0.0}});
  ASSERT_NO_THROW(get_game_object_components(GameObjectType::Enemy, 1));
}
