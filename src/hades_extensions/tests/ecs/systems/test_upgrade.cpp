// Local headers
#include "ecs/registry.hpp"
#include "ecs/stats.hpp"
#include "ecs/systems/upgrade.hpp"
#include "macros.hpp"

/// Represents a test stat useful for testing.
struct TestStat final : Stat {
  /// Initialise the object.
  ///
  /// @param value - The initial and maximum value of the test stat.
  /// @param maximum_level - The maximum level of the test stat.
  TestStat(const double value, const int maximum_level) : Stat(value, maximum_level) {}
};

/// Implements the fixture for the UpgradeSystem tests.
class UpgradeSystemFixture : public testing::Test {
 protected:
  /// The registry that manages the game objects, components, and systems.
  Registry registry;

  /// Set up the fixture for the tests.
  void SetUp() override { registry.add_system<UpgradeSystem>(); }

  /// Create a game object with a given value and maximum level
  ///
  /// @param value - The value of the game object.
  /// @param max_level - The maximum level of the game object.
  void create_game_object(const int value, const int max_level) {
    const std::unordered_map<std::type_index, std::pair<ActionFunction, ActionFunction>> upgrades{
        {typeid(TestStat), std::make_pair([](const int level) { return 150 * (level + 1); },
                                          [](const int level) { return std::pow(level, 2) + 30; })}};
    registry.create_game_object(GameObjectType::Player, cpvzero,
                                {std::make_shared<TestStat>(value, max_level), std::make_shared<Upgrades>(upgrades),
                                 std::make_shared<Money>()});
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

/// Test that a test stat is upgraded correctly if the value equals the maximum.
TEST_F(UpgradeSystemFixture, TestUpgradeSystemUpgradeValueEqualMax) {
  create_upgradeable_game_object();
  registry.get_component<Money>(0)->money = 1000;
  ASSERT_TRUE(get_upgrade_system()->upgrade_component(0, typeid(TestStat)));
  ASSERT_EQ(registry.get_component<TestStat>(0)->get_value(), 350);
  ASSERT_EQ(registry.get_component<TestStat>(0)->get_max_value(), 350);
  ASSERT_EQ(registry.get_component<TestStat>(0)->get_current_level(), 1);
}

/// Test that a test stat is upgraded correctly if the value is lower than the maximum.
TEST_F(UpgradeSystemFixture, TestUpgradeSystemUpgradeValueLowerMax) {
  create_upgradeable_game_object();
  registry.get_component<Money>(0)->money = 1000;
  registry.get_component<TestStat>(0)->set_value(150);
  ASSERT_TRUE(get_upgrade_system()->upgrade_component(0, typeid(TestStat)));
  ASSERT_EQ(registry.get_component<TestStat>(0)->get_value(), 300);
  ASSERT_EQ(registry.get_component<TestStat>(0)->get_max_value(), 350);
  ASSERT_EQ(registry.get_component<TestStat>(0)->get_current_level(), 1);
}

/// Test that a test stat that can be upgraded multiple times is upgraded correctly.
TEST_F(UpgradeSystemFixture, TestUpgradeSystemUpgradeMultipleTimes) {
  create_upgradeable_game_object();
  registry.get_component<Money>(0)->money = 1000;
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
  registry.get_component<Money>(0)->money = 1000;
  ASSERT_TRUE(get_upgrade_system()->upgrade_component(0, typeid(TestStat)));
  ASSERT_FALSE(get_upgrade_system()->upgrade_component(0, typeid(TestStat)));
}

/// Test that a test stat is not upgraded if the player does not have enough money.
TEST_F(UpgradeSystemFixture, TestUpgradeSystemUpgradeNotEnoughMoney) {
  create_game_object(100, 1);
  ASSERT_FALSE(get_upgrade_system()->upgrade_component(0, typeid(TestStat)));
}

/// Test that a test stat is not upgraded if it is not allowed.
TEST_F(UpgradeSystemFixture, TestUpgradeSystemUpgradeNonUpgradeable) {
  create_game_object(100, -1);
  ASSERT_FALSE(get_upgrade_system()->upgrade_component(0, typeid(TestStat)));
}

/// Test that an exception is thrown if a game object does not have the target component.
TEST_F(UpgradeSystemFixture, TestEffectSystemApplyStatusEffectNonexistentTargetComponent) {
  registry.create_game_object(GameObjectType::Player, cpvzero, {});
  ASSERT_THROW_MESSAGE(get_upgrade_system()->upgrade_component(0, typeid(TestStat)), RegistryError,
                       "The component `TestStat` for the game object ID `0` is not registered with the registry.")
}

/// Test that an exception is thrown if an invalid game object ID is provided.
TEST_F(UpgradeSystemFixture, TestUpgradeSystemUpgradeInvalidGameObjectId) {
  ASSERT_THROW_MESSAGE((get_upgrade_system()->upgrade_component(-1, typeid(TestStat))), RegistryError,
                       "The component `TestStat` for the game object ID `-1` is not registered with the registry.")
}
