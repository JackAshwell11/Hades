// External includes
#include "gtest/gtest.h"

// Custom includes
#include "macros.hpp"
#include "game_objects/stats.hpp"
#include "game_objects/systems/upgrade.hpp"

// ----- FIXTURES ------------------------------
/// Implements the fixture for the UpgradeSystem tests.
class UpgradeSystemFixture : public testing::Test {
 protected:
  /// The registry that manages the game objects, components, and systems.
  Registry registry{};

  /// Set up the fixture for the tests.
  void SetUp() override {
    registry.add_system<UpgradeSystem>();
  }

  /// Create a game object with a given value and maximum level
  ///
  /// @param value - The value of the game object.
  /// @param max_level - The maximum level of the game object.
  void create_game_object(int value, int max_level) {
    std::unordered_map<std::type_index, std::function<double(int)>>
        upgrades{{typeid(Stat), [](int level) { return 150 * (level + 1); }}};
    registry.create_game_object();
    registry.add_component<Stat>(0, value, max_level);
    registry.add_component<Upgrades>(0, upgrades);
  }

  /// Create a game object that is upgradable multiple times.
  void create_upgradeable_game_object() {
    create_game_object(200, 3);
  }

  /// Get the upgrade system from the registry.
  ///
  /// @return The upgrade system.
  std::shared_ptr<UpgradeSystem> get_upgrade_system() {
    return registry.find_system<UpgradeSystem>();
  }
};

// ----- TESTS ----------------------------------
/// Test that a stat is upgraded correctly if the value equals the maximum.
TEST_F(UpgradeSystemFixture, TestUpgradeSystemUpgradeValueEqualMax) {
  create_upgradeable_game_object();
  ASSERT_TRUE(get_upgrade_system()->upgrade_component(0, typeid(Stat)));
  ASSERT_EQ(registry.get_component<Stat>(0)->get_value(), 350);
  ASSERT_EQ(registry.get_component<Stat>(0)->max_value, 350);
  ASSERT_EQ(registry.get_component<Stat>(0)->current_level, 1);
}

/// Test that a stat is upgraded correctly if the value is lower than the maximum.
TEST_F(UpgradeSystemFixture, TestUpgradeSystemUpgradeValueLowerMax) {
  create_upgradeable_game_object();
  registry.get_component<Stat>(0)->set_value(150);
  ASSERT_TRUE(get_upgrade_system()->upgrade_component(0, typeid(Stat)));
  ASSERT_EQ(registry.get_component<Stat>(0)->get_value(), 300);
  ASSERT_EQ(registry.get_component<Stat>(0)->max_value, 350);
  ASSERT_EQ(registry.get_component<Stat>(0)->current_level, 1);
}

/// Test that a stat that can be upgraded multiple times is upgraded correctly.
TEST_F(UpgradeSystemFixture, TestUpgradeSystemUpgradeMultipleTimes) {
  create_upgradeable_game_object();
  ASSERT_TRUE(get_upgrade_system()->upgrade_component(0, typeid(Stat)));
  ASSERT_TRUE(get_upgrade_system()->upgrade_component(0, typeid(Stat)));
  ASSERT_TRUE(get_upgrade_system()->upgrade_component(0, typeid(Stat)));
  ASSERT_FALSE(get_upgrade_system()->upgrade_component(0, typeid(Stat)));
  ASSERT_EQ(registry.get_component<Stat>(0)->get_value(), 1100);
  ASSERT_EQ(registry.get_component<Stat>(0)->max_value, 1100);
  ASSERT_EQ(registry.get_component<Stat>(0)->current_level, 3);
}

/// Test that a stat is upgraded only one time.
TEST_F(UpgradeSystemFixture, TestUpgradeSystemUpgradeOnce) {
  create_game_object(150, 1);
  ASSERT_TRUE(get_upgrade_system()->upgrade_component(0, typeid(Stat)));
  ASSERT_FALSE(get_upgrade_system()->upgrade_component(0, typeid(Stat)));
}

/// Test that a stat is not upgraded if it is not allowed.
TEST_F(UpgradeSystemFixture, TestUpgradeSystemUpgradeNonUpgradeable) {
  create_game_object(100, -1);
  ASSERT_FALSE(get_upgrade_system()->upgrade_component(0, typeid(Stat)));
}

/// Test that an exception is raised if an invalid game object ID is provided.
TEST_F(UpgradeSystemFixture, TestUpgradeSystemUpgradeInvalidGameObjectId) {
  ASSERT_THROW_MESSAGE(get_upgrade_system()->upgrade_component(-1, typeid(Stat)),
                       RegistryException,
                       "The game object `-1` is not registered with the registry.")
}
