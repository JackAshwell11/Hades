// Local headers
#include "game_objects/stats.hpp"
#include "game_objects/systems/effects.hpp"
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
/// Implements the fixture for the EffectSystem tests.
class EffectSystemFixture : public testing::Test {
 protected:
  /// The registry that manages the game objects, components, and systems.
  Registry registry{};

  /// The increase function for an effect.
  ActionFunction increase_function{[](int level) { return 5 + std::pow(level, 2); }};

  /// The duration function for an effect.
  ActionFunction duration_function{[](int level) { return 10 * level; }};

  /// The interval function for an effect.
  ActionFunction interval_function{[](int level) { return std::pow(2, level); }};

  /// The data for a status effect.
  StatusEffectData status_effect_data{StatusEffectType::TEMP, increase_function, duration_function, interval_function};

  /// Set up the fixture for the tests.
  void SetUp() override { registry.add_system<EffectSystem>(); }

  /// Create a game object for the instant effect tests.
  void create_instant_game_object() { registry.create_game_object({std::make_shared<TestStat>(100, -1)}); }

  /// Create a game object for the status effect tests.
  void create_status_game_object() {
    registry.create_game_object({std::make_shared<TestStat>(200, -1), std::make_shared<StatusEffects>()});
  }

  /// Get the effect system from the registry.
  ///
  /// @return The effect system.
  auto get_effect_system() -> std::shared_ptr<EffectSystem> { return registry.get_system<EffectSystem>(); }
};

// ----- TESTS ----------------------------------
/// Test that a status effect is updated correctly with a small delta time.
TEST_F(EffectSystemFixture, TestEffectSystemUpdateSmallDeltaTime) {
  create_status_game_object();
  auto test_stat{registry.get_component<TestStat>(0)};
  test_stat->set_value(100);
  get_effect_system()->apply_status_effect(0, typeid(TestStat), status_effect_data, 1);
  ASSERT_EQ(test_stat->get_value(), 106);
  get_effect_system()->update(5);
  ASSERT_EQ(test_stat->get_value(), 112);
  ASSERT_EQ(registry.get_component<StatusEffects>(0)->applied_effects.at(StatusEffectType::TEMP).time_counter, 5);
}

/// Test that a status effect is updated correctly with a large delta time.
TEST_F(EffectSystemFixture, TestEffectSystemUpdateLargeDeltaTime) {
  create_status_game_object();
  auto test_stat{registry.get_component<TestStat>(0)};
  test_stat->set_value(100);
  get_effect_system()->apply_status_effect(0, typeid(TestStat), status_effect_data, 1);
  ASSERT_EQ(test_stat->get_value(), 106);
  get_effect_system()->update(20);
  ASSERT_EQ(test_stat->get_value(), 106);
  ASSERT_FALSE(registry.get_component<StatusEffects>(0)->applied_effects.contains(StatusEffectType::TEMP));
}

/// Test that a status effect is updated correctly after multiple updates.
TEST_F(EffectSystemFixture, TestEffectSystemUpdateMultipleDeltaTimes) {
  create_status_game_object();
  auto test_stat{registry.get_component<TestStat>(0)};
  test_stat->set_value(100);
  get_effect_system()->apply_status_effect(0, typeid(TestStat), status_effect_data, 1);
  auto &applied_effects{registry.get_component<StatusEffects>(0)->applied_effects};
  get_effect_system()->update(1);
  ASSERT_EQ(test_stat->get_value(), 106);
  ASSERT_EQ(applied_effects.at(StatusEffectType::TEMP).time_counter, 1);
  get_effect_system()->update(5);
  ASSERT_EQ(test_stat->get_value(), 112);
  ASSERT_EQ(applied_effects.at(StatusEffectType::TEMP).time_counter, 6);
  get_effect_system()->update(1);
  ASSERT_EQ(test_stat->get_value(), 118);
  ASSERT_EQ(applied_effects.at(StatusEffectType::TEMP).time_counter, 7);
  get_effect_system()->update(1);
  ASSERT_EQ(test_stat->get_value(), 124);
  ASSERT_EQ(applied_effects.at(StatusEffectType::TEMP).time_counter, 8);
  get_effect_system()->update(2);
  ASSERT_EQ(test_stat->get_value(), 124);
  ASSERT_FALSE(applied_effects.contains(StatusEffectType::TEMP));
}

/// Test that a status effect is not updated if one does not exist.
TEST_F(EffectSystemFixture, TestEffectSystemUpdateNoStatusEffect) {
  create_status_game_object();
  auto test_stat{registry.get_component<TestStat>(0)};
  test_stat->set_value(100);
  get_effect_system()->update(5);
  ASSERT_EQ(test_stat->get_value(), 100);
}

/// Test that an instant effect is applied correctly if the value equals the maximum.
TEST_F(EffectSystemFixture, TestEffectSystemApplyInstantEffectValueEqualMax) {
  create_instant_game_object();
  get_effect_system()->apply_instant_effect(0, typeid(TestStat), increase_function, 0);
  ASSERT_EQ(registry.get_component<TestStat>(0)->get_value(), 100);
}

/// Test that an instant effect is applied correctly if the value is lower than the maximum.
TEST_F(EffectSystemFixture, TestEffectSystemApplyInstantEffectValueLowerMax) {
  create_instant_game_object();
  auto component{registry.get_component<TestStat>(0)};
  component->set_value(40);
  get_effect_system()->apply_instant_effect(0, typeid(TestStat), increase_function, 0);
  ASSERT_EQ(component->get_value(), 45);
}

/// Test that an instant effect is not applied if the target component is not initialised.
TEST_F(EffectSystemFixture, TestEffectSystemApplyInstantEffectNonexistentTargetComponent) {
  registry.create_game_object({});
  ASSERT_THROW_MESSAGE(
      get_effect_system()->apply_instant_effect(0, typeid(TestStat), increase_function, 1), RegistryError,
      "The game object `0` is not registered with the registry or does not have the required component.")
}

/// Test that an exception is raised if an invalid game object ID is provided.
TEST_F(EffectSystemFixture, TestEffectSystemApplyInstantEffectInvalidGameObjectId){ASSERT_THROW_MESSAGE(
    get_effect_system()->apply_instant_effect(-1, typeid(TestStat), increase_function, 1), RegistryError,
    "The game object `-1` is not registered with the registry or does not have the required component.")}

/// Test that a status effect is applied correctly if no status effect is currently applied.
TEST_F(EffectSystemFixture, TestEffectSystemApplyStatusEffectNoAppliedEffect) {
  create_status_game_object();
  ASSERT_TRUE(get_effect_system()->apply_status_effect(0, typeid(TestStat), status_effect_data, 1));
  ASSERT_EQ(registry.get_component<TestStat>(0)->get_value(), 200);
  auto applied_status_effect{registry.get_component<StatusEffects>(0)->applied_effects.at(StatusEffectType::TEMP)};
  ASSERT_EQ(applied_status_effect.value, 6);
  ASSERT_EQ(applied_status_effect.duration, 10);
  ASSERT_EQ(applied_status_effect.interval, 2);
  ASSERT_EQ(applied_status_effect.time_counter, 0);
}

/// Test that a status effect is applied correctly if the value is lower than the max.
TEST_F(EffectSystemFixture, TestEffectSystemApplyStatusEffectValueLowerMax) {
  create_status_game_object();
  auto component{registry.get_component<TestStat>(0)};
  component->set_value(150);
  ASSERT_TRUE(get_effect_system()->apply_status_effect(0, typeid(TestStat), status_effect_data, 1));
  ASSERT_EQ(component->get_value(), 156);
}

/// Test that a status effect is not applied if a status effect is already applied.
TEST_F(EffectSystemFixture, TestEffectSystemApplyStatusEffectExistingStatusEffect) {
  create_status_game_object();
  ASSERT_TRUE(get_effect_system()->apply_status_effect(0, typeid(TestStat), status_effect_data, 1));
  ASSERT_TRUE(registry.get_component<StatusEffects>(0)->applied_effects.contains(StatusEffectType::TEMP));
  ASSERT_FALSE(get_effect_system()->apply_status_effect(0, typeid(TestStat), status_effect_data, 1));
}

/// Test that multiple status effects are applied correctly.
TEST_F(EffectSystemFixture, TestEffectSystemApplyStatusEffectMultipleStatusEffects) {
  create_status_game_object();
  ASSERT_TRUE(get_effect_system()->apply_status_effect(0, typeid(TestStat), status_effect_data, 1));
  const StatusEffectData status_effect_data_two{StatusEffectType::TEMP2, increase_function, duration_function,
                                                interval_function};
  ASSERT_TRUE(get_effect_system()->apply_status_effect(0, typeid(TestStat), status_effect_data_two, 1));
  auto status_effects{registry.get_component<StatusEffects>(0)->applied_effects};
  ASSERT_TRUE(status_effects.contains(StatusEffectType::TEMP));
  ASSERT_TRUE(status_effects.contains(StatusEffectType::TEMP2));
}

/// Test that a status effect is not applied if the target component is not initialised.
TEST_F(EffectSystemFixture, TestEffectSystemApplyStatusEffectNonexistentTargetComponent) {
  registry.create_game_object({});
  ASSERT_THROW_MESSAGE(
      get_effect_system()->apply_status_effect(0, typeid(TestStat), status_effect_data, 1), RegistryError,
      "The game object `0` is not registered with the registry or does not have the required component.")
}

/// Test that an exception is raised if an invalid game object ID is provided.
TEST_F(EffectSystemFixture, TestEffectSystemApplyStatusEffectInvalidGameObjectId) {
  ASSERT_THROW_MESSAGE(
      get_effect_system()->apply_status_effect(-1, typeid(TestStat), status_effect_data, 1), RegistryError,
      "The game object `-1` is not registered with the registry or does not have the required component.")
}
