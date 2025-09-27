// Local headers
#include "ecs/registry.hpp"
#include "events.hpp"
#include "factories.hpp"
#include "game_object.hpp"
#include "macros.hpp"

/// Implements the fixture for the factories.hpp tests.
class FactoriesFixture : public testing::Test {  // NOLINT
 protected:
  /// The registry that manages the game objects, components, and systems.
  Registry registry;

  /// Tear down the fixture for the tests.
  void TearDown() override {
    clear_hitboxes();
    clear_listeners();
  }

  /// Load hitboxes for the factories.
  static void load_hitboxes() {
    load_hitbox(GameObjectType::Player, {{0.0, 1.0}, {1.0, 2.0}, {2.0, 0.0}});
    load_hitbox(GameObjectType::Enemy, {{0.0, 1.0}, {1.0, 2.0}, {2.0, 0.0}});
  }
};

/// Test that a valid position is converted correctly.
TEST(Tests, TestGridPosToPixelPositivePosition) { ASSERT_EQ(grid_pos_to_pixel({.x = 100, .y = 100}), cpv(6432, 6432)); }

/// Test that a zero position is converted correctly.
TEST(Tests, TestGridPosToPixelZeroPosition) { ASSERT_EQ(grid_pos_to_pixel(cpvzero), cpv(32, 32)); }

/// Test that a negative x position raises an error.
TEST(Tests, TestGridPosToPixelNegativeXPosition){
    ASSERT_THROW_MESSAGE(grid_pos_to_pixel({-100, 100}), std::invalid_argument, "The position cannot be negative.")}

/// Test that a negative y position raises an error.
TEST(Tests, TestGridPosToPixelNegativeYPosition){
    ASSERT_THROW_MESSAGE(grid_pos_to_pixel({100, -100}), std::invalid_argument, "The position cannot be negative.")}

/// Test that a negative x and y position raises an error.
TEST(Tests, TestGridPosToPixelNegativeXYPosition){
    ASSERT_THROW_MESSAGE(grid_pos_to_pixel({-100, -100}), std::invalid_argument, "The position cannot be negative.")}

/// Test that loading a hitbox for a game object type works.
TEST_F(FactoriesFixture, TestLoadHitboxOnce) {
  ASSERT_TRUE(load_hitbox(GameObjectType::Player, {{0.0, 0.0}}));
}

/// Test that loading two hitboxes for the same game object type doesn't do anything.
TEST_F(FactoriesFixture, TestLoadHitboxTwice) {
  load_hitboxes();
  ASSERT_FALSE(load_hitbox(GameObjectType::Player, {{1.0, 1.0}}));
}

/// Test that creating a game object which doesn't require a hitbox works correctly.
TEST_F(FactoriesFixture, TestCreateGameObjectNoHitboxRequired) {
  ASSERT_NO_THROW(create_game_object(&registry, GameObjectType::Floor, {0.0, 0.0}));
}

/// Test that creating a game object that requires a hitbox works when the hitbox is loaded.
TEST_F(FactoriesFixture, TestCreateGameObjectHitboxLoaded) {
  load_hitboxes();
  ASSERT_NO_THROW(create_game_object(&registry, GameObjectType::Player, {0.0, 0.0}));
}

/// Test that creating a game object which doesn't have a loaded hitbox throws an exception.
TEST_F(FactoriesFixture, TestCreateGameObjectHitboxNotLoaded) {
  ASSERT_THROW(create_game_object(&registry, GameObjectType::Player, {0.0, 0.0}), std::out_of_range);
}

/// Test that creating a game object with a negative position throws an exception.
TEST_F(FactoriesFixture, TestCreateGameObjectNegativePosition) {
  load_hitboxes();
  ASSERT_THROW_MESSAGE(create_game_object(&registry, GameObjectType::Player, {-1.0, 0.0}), std::invalid_argument,
                       "The position cannot be negative.")
  ASSERT_THROW_MESSAGE(create_game_object(&registry, GameObjectType::Player, {0.0, -1.0}), std::invalid_argument,
                       "The position cannot be negative.")
  ASSERT_THROW_MESSAGE(create_game_object(&registry, GameObjectType::Player, {-1.0, -1.0}), std::invalid_argument,
                       "The position cannot be negative.")
}

/// Test that creating a bullet game object with a negative position works correctly.
TEST_F(FactoriesFixture, TestCreateGameObjectBulletNegativePosition) {
  ASSERT_NO_THROW(create_game_object(&registry, GameObjectType::Bullet, {-1.0, 0.0}));
  ASSERT_NO_THROW(create_game_object(&registry, GameObjectType::Bullet, {0.0, -1.0}));
  ASSERT_NO_THROW(create_game_object(&registry, GameObjectType::Bullet, {-1.0, -1.0}));
}

/// Test that creating a game object with a level works correctly.
TEST_F(FactoriesFixture, TestCreateGameObjectWithLevel) {
  load_hitboxes();
  ASSERT_NO_THROW(create_game_object(&registry, GameObjectType::Enemy, {0.0, 0.0}, 1));
}

/// Test that the position changed callback is called correctly.
TEST_F(FactoriesFixture, TestCreateGameObjectPositionChangedCallback) {
  std::pair<double, double> called{-1, -1};
  add_callback<EventType::PositionChanged>(
      [&called](const GameObjectID, const std::pair<double, double>& event) { called = event; });
  create_game_object(&registry, GameObjectType::Floor, {5.0, 10.0});
  ASSERT_EQ(called, std::make_pair(352.0, 672.0));
}
