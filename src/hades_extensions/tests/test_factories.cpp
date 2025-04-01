// Local headers
#include "factories.hpp"
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

/// Test that loading a factory that doesn't require a hitbox works.
TEST_F(FactoriesFixture, TestGetFactoryNoHitboxRequired) {
  ASSERT_NO_THROW(get_factories().at(GameObjectType::Floor)(0));
}

/// Test that loading a factory that requires a hitbox works when the hitbox is loaded.
TEST_F(FactoriesFixture, TestGetFactoryHitboxLoaded) {
  load_hitbox(GameObjectType::Player, {{0.0, 0.0}});
  ASSERT_NO_THROW(get_factories().at(GameObjectType::Player)(0));
}

/// Test that loading a factory that requires a hitbox which isn't loaded throws an exception.
TEST_F(FactoriesFixture, TestGetFactoriesHitboxNotLoaded) {
  ASSERT_THROW(get_factories().at(GameObjectType::Player)(0), std::out_of_range);
}
