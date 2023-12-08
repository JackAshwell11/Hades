// Local headers
#include "game_objects/stats.hpp"
#include "game_objects/systems/upgrade.hpp"
#include "macros.hpp"

// ----- STRUCTURES -----------------------------
/// Represents a test stat useful for testing.
struct TestStat : public Stat {
  /// Initialise the object.
  ///
  /// @param value - The initial and maximum value of the test stat.
  /// @param maximum_level - The maximum level of the test stat.
  TestStat(const double value, const int maximum_level) : Stat(value, maximum_level) {}
};

// ----- FIXTURES ------------------------------
/// Implements the fixture for the UpgradeSystem tests.
class UpgradeSystemFixture : public testing::Test {
 protected:
  /// The registry that manages the game objects, components, and systems.
  Registry registry{};

  /// Set up the fixture for the tests.
  void SetUp() override { registry.add_system<UpgradeSystem>(); }

  /// Create a game object with a given value and maximum level
  ///
  /// @param value - The value of the game object.
  /// @param max_level - The maximum level of the game object.
  void create_game_object(const int value, const int max_level) {
    const std::unordered_map<std::type_index, ActionFunction> upgrades{
        {typeid(TestStat), [](int level) { return 150 * (level + 1); }}};
    registry.create_game_object({std::make_shared<TestStat>(value, max_level), std::make_shared<Upgrades>(upgrades)});
  }

  /// Create a game object that is upgradable multiple times.
  void create_upgradeable_game_object() { create_game_object(200, 3); }

  /// Get the upgrade system from the registry.
  ///
  /// @return The upgrade system.
  [[nodiscard]] auto get_upgrade_system() const -> std::shared_ptr<UpgradeSystem> {
    return registry.get_system<UpgradeSystem>();
  }
};

// ----- TESTS ----------------------------------
/// Test that a test stat is upgraded correctly if the value equals the maximum.
TEST_F(UpgradeSystemFixture, TestUpgradeSystemUpgradeValueEqualMax) {
  create_upgradeable_game_object();
  ASSERT_TRUE(get_upgrade_system()->upgrade_component(0, typeid(TestStat)));
  ASSERT_EQ(registry.get_component<TestStat>(0)->get_value(), 350);
  ASSERT_EQ(registry.get_component<TestStat>(0)->get_max_value(), 350);
  ASSERT_EQ(registry.get_component<TestStat>(0)->get_current_level(), 1);
}

/// Test that a test stat is upgraded correctly if the value is lower than the maximum.
TEST_F(UpgradeSystemFixture, TestUpgradeSystemUpgradeValueLowerMax) {
  create_upgradeable_game_object();
  registry.get_component<TestStat>(0)->set_value(150);
  ASSERT_TRUE(get_upgrade_system()->upgrade_component(0, typeid(TestStat)));
  ASSERT_EQ(registry.get_component<TestStat>(0)->get_value(), 300);
  ASSERT_EQ(registry.get_component<TestStat>(0)->get_max_value(), 350);
  ASSERT_EQ(registry.get_component<TestStat>(0)->get_current_level(), 1);
}

/// Test that a test stat that can be upgraded multiple times is upgraded correctly.
TEST_F(UpgradeSystemFixture, TestUpgradeSystemUpgradeMultipleTimes) {
  create_upgradeable_game_object();
  ASSERT_TRUE(get_upgrade_system()->upgrade_component(0, typeid(TestStat)));
  ASSERT_TRUE(get_upgrade_system()->upgrade_component(0, typeid(TestStat)));
  ASSERT_TRUE(get_upgrade_system()->upgrade_component(0, typeid(TestStat)));
  ASSERT_FALSE(get_upgrade_system()->upgrade_component(0, typeid(TestStat)));
  ASSERT_EQ(registry.get_component<TestStat>(0)->get_value(), 1100);
  ASSERT_EQ(registry.get_component<TestStat>(0)->get_max_value(), 1100);
  ASSERT_EQ(registry.get_component<TestStat>(0)->get_current_level(), 3);
}

/// Test that a test stat is upgraded only one time.
TEST_F(UpgradeSystemFixture, TestUpgradeSystemUpgradeOnce) {
  create_game_object(150, 1);
  ASSERT_TRUE(get_upgrade_system()->upgrade_component(0, typeid(TestStat)));
  ASSERT_FALSE(get_upgrade_system()->upgrade_component(0, typeid(TestStat)));
}

/// Test that a test stat is not upgraded if it is not allowed.
TEST_F(UpgradeSystemFixture, TestUpgradeSystemUpgradeNonUpgradeable) {
  create_game_object(100, -1);
  ASSERT_FALSE(get_upgrade_system()->upgrade_component(0, typeid(TestStat)));
}

/// Test that a test stat is not upgraded if the target component is not initialised.
TEST_F(UpgradeSystemFixture, TestEffectSystemApplyStatusEffectNonexistentTargetComponent) {
  registry.create_game_object({});
  ASSERT_THROW_MESSAGE(
      get_upgrade_system()->upgrade_component(0, typeid(TestStat)), RegistryError,
      "The game object `0` is not registered with the registry or does not have the required component.")
}

/// Test that an exception is thrown if an invalid game object ID is provided.
TEST_F(UpgradeSystemFixture, TestUpgradeSystemUpgradeInvalidGameObjectId) {
  ASSERT_THROW_MESSAGE(
      (get_upgrade_system()->upgrade_component(-1, typeid(TestStat))), RegistryError,
      "The game object `-1` is not registered with the registry or does not have the required component.")
}
