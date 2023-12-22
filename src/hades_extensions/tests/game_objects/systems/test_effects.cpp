// Local headers
#include "game_objects/stats.hpp"
#include "game_objects/systems/effects.hpp"
#include "macros.hpp"

// ----- STRUCTURES -----------------------------
/// Represents a test stat useful for testing.
struct TestStat final : Stat {
  /// Initialise the object.
  ///
  /// @param value - The initial and maximum value of the test stat.
  /// @param maximum_level - The maximum level of the test stat.
  TestStat(const double value, const int maximum_level) : Stat(value, maximum_level) {}
};

/// Represents an extra test stat useful for testing.
struct TestStat2 final : Stat {
  /// Initialise the object.
  ///
  /// @param value - The initial and maximum value of the test stat.
  /// @param maximum_level - The maximum level of the test stat.
  TestStat2(const double value, const int maximum_level) : Stat(value, maximum_level) {}
};

// ----- FIXTURES ------------------------------
/// Implements the fixture for the EffectSystem tests.
class EffectSystemFixture : public testing::Test {
 protected:
  /// The registry that manages the game objects, components, and systems.
  Registry registry{};

  /// The increase function for an effect.
  const ActionFunction increase_function{[](const int level) { return 5 + std::pow(level, 2); }};

  /// The duration function for an effect.
  const ActionFunction duration_function{[](const int level) { return 10 * level; }};

  /// The interval function for an effect.
  const ActionFunction interval_function{[](const int level) { return std::pow(2, level); }};

  /// The data for a status effect.
  const StatusEffectData status_effect_data{StatusEffectType::TEMP, increase_function, duration_function,
                                            interval_function};

  /// Set up the fixture for the tests.
  void SetUp() override {
    registry.add_system<EffectSystem>();
    registry.create_game_object(
        {std::make_shared<TestStat>(200, -1), std::make_shared<TestStat2>(100, -1), std::make_shared<StatusEffects>()});
  }

  /// Create a game object to hold the instant and status effects.
  void create_effect_applier(const bool instant = false, const bool status = false) {
    const auto effect_applier{std::make_shared<EffectApplier>(std::unordered_map<std::type_index, ActionFunction>{},
                                                              std::unordered_map<std::type_index, StatusEffectData>{})};
    if (instant) {
      effect_applier->instant_effects.emplace(typeid(TestStat), increase_function);
    }
    if (status) {
      effect_applier->status_effects.emplace(typeid(TestStat), status_effect_data);
    }
    registry.create_game_object({effect_applier});
  }

  /// Get the effect system from the registry.
  ///
  /// @return The effect system.
  [[nodiscard]] auto get_effect_system() const -> std::shared_ptr<EffectSystem> {
    return registry.get_system<EffectSystem>();
  }
};

// ----- TESTS ----------------------------------
/// Test that a status effect is updated correctly with a small delta time.
TEST_F(EffectSystemFixture, TestEffectSystemUpdateSmallDeltaTime) {
  create_effect_applier(false, true);
  const auto test_stat{registry.get_component<TestStat>(0)};
  test_stat->set_value(100);
  ASSERT_TRUE(get_effect_system()->apply_effects(1, 0));
  ASSERT_EQ(test_stat->get_value(), 106);
  get_effect_system()->update(5);
  ASSERT_EQ(test_stat->get_value(), 112);
  ASSERT_EQ(registry.get_component<StatusEffects>(0)->applied_effects.at(StatusEffectType::TEMP).time_counter, 5);
}

/// Test that a status effect is updated correctly with a large delta time.
TEST_F(EffectSystemFixture, TestEffectSystemUpdateLargeDeltaTime) {
  create_effect_applier(false, true);
  const auto test_stat{registry.get_component<TestStat>(0)};
  test_stat->set_value(100);
  ASSERT_TRUE(get_effect_system()->apply_effects(1, 0));
  ASSERT_EQ(test_stat->get_value(), 106);
  get_effect_system()->update(20);
  ASSERT_EQ(test_stat->get_value(), 106);
  ASSERT_FALSE(registry.get_component<StatusEffects>(0)->applied_effects.contains(StatusEffectType::TEMP));
}

/// Test that a status effect is updated correctly after multiple updates.
TEST_F(EffectSystemFixture, TestEffectSystemUpdateMultipleDeltaTimes) {
  create_effect_applier(false, true);
  const auto test_stat{registry.get_component<TestStat>(0)};
  test_stat->set_value(100);
  ASSERT_TRUE(get_effect_system()->apply_effects(1, 0));
  const auto &applied_effects{registry.get_component<StatusEffects>(0)->applied_effects};
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
  create_effect_applier(false, false);
  const auto test_stat{registry.get_component<TestStat>(0)};
  test_stat->set_value(100);
  get_effect_system()->update(5);
  ASSERT_EQ(test_stat->get_value(), 100);
}

/// Test that an instant effect is applied correctly if the value equals the maximum.
TEST_F(EffectSystemFixture, TestEffectSystemApplyEffectsInstantValueEqualMax) {
  create_effect_applier(true, false);
  ASSERT_FALSE(get_effect_system()->apply_effects(1, 0));
  ASSERT_EQ(registry.get_component<TestStat>(0)->get_value(), 200);
}

/// Test that an instant effect is applied correctly if the value is lower than the maximum.
TEST_F(EffectSystemFixture, TestEffectSystemApplyEffectsInstantValueLowerMax) {
  create_effect_applier(true, false);
  const auto component{registry.get_component<TestStat>(0)};
  component->set_value(150);
  ASSERT_TRUE(get_effect_system()->apply_effects(1, 0));
  ASSERT_EQ(component->get_value(), 156);
}

/// Test that a status effect is applied correctly if no status effect is currently applied.
TEST_F(EffectSystemFixture, TestEffectSystemApplyEffectsStatusNoAppliedEffect) {
  create_effect_applier(false, true);
  ASSERT_TRUE(get_effect_system()->apply_effects(1, 0));
  ASSERT_EQ(registry.get_component<TestStat>(0)->get_value(), 200);
  const auto applied_status_effect{
      registry.get_component<StatusEffects>(0)->applied_effects.at(StatusEffectType::TEMP)};
  ASSERT_EQ(applied_status_effect.value, 6);
  ASSERT_EQ(applied_status_effect.duration, 10);
  ASSERT_EQ(applied_status_effect.interval, 2);
  ASSERT_EQ(applied_status_effect.time_counter, 0);
}

/// Test that a status effect is applied correctly if the value is lower than the max.
TEST_F(EffectSystemFixture, TestEffectSystemApplyEffectsStatusValueLowerMax) {
  create_effect_applier(false, true);
  const auto component{registry.get_component<TestStat>(0)};
  component->set_value(150);
  ASSERT_TRUE(get_effect_system()->apply_effects(1, 0));
  ASSERT_EQ(component->get_value(), 156);
}

/// Test that a status effect is not applied if a status effect is already applied.
TEST_F(EffectSystemFixture, TestEffectSystemApplyEffectsStatusExistingStatusEffect) {
  create_effect_applier(false, true);
  ASSERT_TRUE(get_effect_system()->apply_effects(1, 0));
  ASSERT_TRUE(registry.get_component<StatusEffects>(0)->applied_effects.contains(StatusEffectType::TEMP));
  ASSERT_FALSE(get_effect_system()->apply_effects(1, 0));
}

/// Test that multiple status effects are applied correctly.
TEST_F(EffectSystemFixture, TestEffectSystemApplyEffectsStatusMultipleStatusEffects) {
  create_effect_applier(false, true);
  registry.get_component<EffectApplier>(1)->status_effects.emplace(
      typeid(TestStat2),
      StatusEffectData{StatusEffectType::TEMP2, increase_function, duration_function, interval_function});
  ASSERT_TRUE(get_effect_system()->apply_effects(1, 0));
  ASSERT_FALSE(get_effect_system()->apply_effects(1, 0));
  const auto status_effects{registry.get_component<StatusEffects>(0)->applied_effects};
  ASSERT_TRUE(status_effects.contains(StatusEffectType::TEMP));
  ASSERT_TRUE(status_effects.contains(StatusEffectType::TEMP2));
}

/// Test that applying effects throws an exception if the target component is not initialised.
TEST_F(EffectSystemFixture, TestEffectSystemApplyEffectsNonexistentTargetComponent) {
  create_effect_applier(false, false);
  registry.create_game_object({});
  ASSERT_THROW_MESSAGE(
      get_effect_system()->apply_effects(1, 2), RegistryError,
      "The game object `2` is not registered with the registry or does not have the required component.")
}

/// Test that applying effects throws an exception if an invalid source game object ID is provided.
TEST_F(EffectSystemFixture, TestEffectSystemApplyEffectsInvalidSourceGameObjectId){ASSERT_THROW_MESSAGE(
    get_effect_system()->apply_effects(-1, 0), RegistryError,
    "The game object `-1` is not registered with the registry or does not have the required component.")}

/// Test that applying effects throws an exception if an invalid target game object ID is provided.
TEST_F(EffectSystemFixture, TestEffectSystemApplyEffectsInvalidTargetGameObjectId) {
  create_effect_applier(false, false);
  ASSERT_THROW_MESSAGE(
      get_effect_system()->apply_effects(1, -1), RegistryError,
      "The game object `-1` is not registered with the registry or does not have the required component.")
}

/// Test that applying effects throws an exception if the source game object does not have an effect applier component.
TEST_F(EffectSystemFixture, TestEffectSystemApplyEffectsNonexistentEffectApplier) {
  registry.create_game_object({});
  ASSERT_THROW_MESSAGE(
      get_effect_system()->apply_effects(1, 0), RegistryError,
      "The game object `1` is not registered with the registry or does not have the required component.")
}

/// Test that a status effect throws an exception if the target game object does not have a status effects component.
TEST_F(EffectSystemFixture, TestEffectSystemApplyEffectsNonexistentStatusEffects) {
  create_effect_applier(false, false);
  registry.create_game_object({std::make_shared<TestStat>(50, -1)});
  ASSERT_THROW_MESSAGE(
      get_effect_system()->apply_effects(1, 2), RegistryError,
      "The game object `2` is not registered with the registry or does not have the required component.")
}
