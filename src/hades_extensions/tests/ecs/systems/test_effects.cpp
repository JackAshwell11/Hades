// Local headers
#include "ecs/registry.hpp"
#include "ecs/stats.hpp"
#include "ecs/systems/effects.hpp"
#include "macros.hpp"

/// Represents a test stat useful for testing.
struct TestStat final : Stat {
  /// Initialise the object.
  ///
  /// @param value - The initial and maximum value of the test stat.
  /// @param maximum_level - The maximum level of the test stat.
  TestStat(const double value, const int maximum_level) : Stat(value, maximum_level) {}
};

/// Implements the fixture for the EffectSystem tests.
class EffectSystemFixture : public testing::Test {
 protected:
  /// A random generator for use in testing.
  std::mt19937 random_generator;

  /// The registry that manages the game objects, components, and systems.
  Registry registry{random_generator};

  /// Set up the fixture for the tests.
  void SetUp() override {
    registry.add_system<EffectSystem>();
    registry.create_game_object(GameObjectType::Player, cpvzero,
                                {std::make_shared<StatusEffects>(), std::make_shared<TestStat>(200, -1)});
  }

  /// Create a game object to hold the instant and status effects.
  ///
  /// @param instant - Whether to create an instant effect or not.
  /// @param status - Whether to create a status effect or not.
  void create_effect_applier(const bool instant = false, const bool status = false) {
    const auto effect_applier{std::make_shared<EffectApplier>()};
    if (instant) {
      effect_applier->add_instant_effect(5, typeid(TestStat));
    }
    if (status) {
      effect_applier->add_status_effect(StatusEffectType::Regeneration, 5, 10, 2, typeid(TestStat));
    }
    registry.create_game_object(GameObjectType::Enemy, cpvzero, {effect_applier});
  }

  /// Get the effect system from the registry.
  ///
  /// @return The effect system.
  [[nodiscard]] auto get_effect_system() const -> std::shared_ptr<EffectSystem> {
    return registry.get_system<EffectSystem>();
  }
};

/// Test that a status effect is updated correctly with a small delta time.
TEST_F(EffectSystemFixture, TestEffectSystemUpdateSmallDeltaTime) {
  create_effect_applier(false, true);
  const auto test_stat{registry.get_component<TestStat>(0)};
  test_stat->set_value(100);
  ASSERT_TRUE(get_effect_system()->apply_effects(1, 0));
  ASSERT_EQ(test_stat->get_value(), 105);
  get_effect_system()->update(2);
  ASSERT_EQ(test_stat->get_value(), 110);
  ASSERT_EQ(registry.get_component<StatusEffects>(0)->active_effects[0]->time_elapsed, 2);
}

/// Test that a status effect is updated correctly with a large delta time.
TEST_F(EffectSystemFixture, TestEffectSystemUpdateLargeDeltaTime) {
  create_effect_applier(false, true);
  const auto test_stat{registry.get_component<TestStat>(0)};
  test_stat->set_value(100);
  ASSERT_TRUE(get_effect_system()->apply_effects(1, 0));
  ASSERT_EQ(test_stat->get_value(), 105);
  get_effect_system()->update(20);
  ASSERT_EQ(test_stat->get_value(), 130);
  ASSERT_TRUE(registry.get_component<StatusEffects>(0)->active_effects.empty());
}

/// Test that a status effect is updated correctly after multiple updates.
TEST_F(EffectSystemFixture, TestEffectSystemUpdateMultipleDeltaTimes) {
  create_effect_applier(false, true);
  const auto test_stat{registry.get_component<TestStat>(0)};
  test_stat->set_value(100);
  ASSERT_TRUE(get_effect_system()->apply_effects(1, 0));
  ASSERT_EQ(test_stat->get_value(), 105);
  const auto &active_effects{registry.get_component<StatusEffects>(0)->active_effects};
  get_effect_system()->update(1);
  ASSERT_EQ(test_stat->get_value(), 105);
  ASSERT_EQ(active_effects[0]->time_elapsed, 1);
  get_effect_system()->update(5);
  ASSERT_EQ(test_stat->get_value(), 120);
  ASSERT_EQ(active_effects[0]->time_elapsed, 6);
  get_effect_system()->update(2);
  ASSERT_EQ(test_stat->get_value(), 125);
  ASSERT_EQ(active_effects[0]->time_elapsed, 8);
  get_effect_system()->update(3);
  ASSERT_EQ(test_stat->get_value(), 130);
  ASSERT_TRUE(active_effects.empty());
}

/// Test that multiple status effects are updated correctly.
TEST_F(EffectSystemFixture, TestEffectSystemUpdateMultipleStatusEffects) {
  create_effect_applier(false, true);
  const auto component{registry.get_component<TestStat>(0)};
  component->set_value(100);
  registry.get_component<EffectApplier>(1)->add_status_effect(StatusEffectType::Regeneration, 1, 1, 1,
                                                              typeid(TestStat));
  ASSERT_TRUE(get_effect_system()->apply_effects(1, 0));
  ASSERT_EQ(component->get_value(), 106);
  ASSERT_EQ(registry.get_component<StatusEffects>(0)->active_effects.size(), 2);
  get_effect_system()->update(2);
  ASSERT_EQ(component->get_value(), 112);
  ASSERT_EQ(registry.get_component<StatusEffects>(0)->active_effects.size(), 1);
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
  ASSERT_EQ(component->get_value(), 155);
}

/// Test that a status effect is applied correctly if the value equals the maximum.
TEST_F(EffectSystemFixture, TestEffectSystemApplyEffectsStatusValueEqualMax) {
  create_effect_applier(false, true);
  ASSERT_FALSE(get_effect_system()->apply_effects(1, 0));
  ASSERT_EQ(registry.get_component<TestStat>(0)->get_value(), 200);
}

/// Test that a status effect is applied correctly if the value is lower than the max.
TEST_F(EffectSystemFixture, TestEffectSystemApplyEffectsStatusValueLowerMax) {
  create_effect_applier(false, true);
  const auto component{registry.get_component<TestStat>(0)};
  component->set_value(150);
  ASSERT_TRUE(get_effect_system()->apply_effects(1, 0));
  ASSERT_EQ(component->get_value(), 155);
}

/// Test that multiple status effects are applied correctly.
TEST_F(EffectSystemFixture, TestEffectSystemApplyEffectsStatusMultipleStatusEffects) {
  create_effect_applier(false, true);
  registry.get_component<TestStat>(0)->set_value(100);
  registry.get_component<EffectApplier>(1)->add_status_effect(StatusEffectType::Regeneration, 1, 1, 1,
                                                              typeid(TestStat));
  ASSERT_TRUE(get_effect_system()->apply_effects(1, 0));
  ASSERT_EQ(registry.get_component<StatusEffects>(0)->active_effects.size(), 2);
}

/// Test that multiple instant and status effects are applied correctly.
TEST_F(EffectSystemFixture, TestEffectSystemApplyEffectsStatusMultipleInstantAndStatusEffects) {
  create_effect_applier(true, true);
  const auto component{registry.get_component<TestStat>(0)};
  component->set_value(100);
  registry.get_component<EffectApplier>(1)->add_status_effect(StatusEffectType::Regeneration, 1, 1, 1,
                                                              typeid(TestStat));
  ASSERT_TRUE(get_effect_system()->apply_effects(1, 0));
  ASSERT_EQ(registry.get_component<StatusEffects>(0)->active_effects.size(), 2);
  ASSERT_EQ(component->get_value(), 111);
}

/// Test that an exception is thrown if the game object does not have the target component.
TEST_F(EffectSystemFixture, TestEffectSystemApplyEffectsNonexistentTargetComponent) {
  create_effect_applier(false, false);
  registry.create_game_object(GameObjectType::Player, cpvzero, {});
  ASSERT_THROW_MESSAGE(get_effect_system()->apply_effects(1, 2), RegistryError,
                       "The component `StatusEffects` for the game object ID `2` is not registered with the registry.")
}

/// Test that an exception is thrown if an invalid source game object ID is provided.
TEST_F(EffectSystemFixture, TestEffectSystemApplyEffectsInvalidSourceGameObjectId){ASSERT_THROW_MESSAGE(
    get_effect_system()->apply_effects(-1, 0), RegistryError,
    "The component `EffectApplier` for the game object ID `-1` is not registered with the registry.")}

/// Test that an exception is thrown if an invalid target game object ID is provided.
TEST_F(EffectSystemFixture, TestEffectSystemApplyEffectsInvalidTargetGameObjectId) {
  create_effect_applier(false, false);
  ASSERT_THROW_MESSAGE(get_effect_system()->apply_effects(1, -1), RegistryError,
                       "The component `StatusEffects` for the game object ID `-1` is not registered with the registry.")
}

/// Test that an exception is thrown if the source game object does not have an effect applier component.
TEST_F(EffectSystemFixture, TestEffectSystemApplyEffectsNonexistentEffectApplier) {
  registry.create_game_object(GameObjectType::Player, cpvzero, {});
  ASSERT_THROW_MESSAGE(get_effect_system()->apply_effects(1, 0), RegistryError,
                       "The component `EffectApplier` for the game object ID `1` is not registered with the registry.")
}

/// Test that an exception is thrown if the source game object does not have a status effects component.
TEST_F(EffectSystemFixture, TestEffectSystemApplyEffectsNonexistentStatusEffects) {
  create_effect_applier(false, false);
  registry.create_game_object(GameObjectType::Player, cpvzero, {std::make_shared<TestStat>(50, -1)});
  ASSERT_THROW_MESSAGE(get_effect_system()->apply_effects(1, 2), RegistryError,
                       "The component `StatusEffects` for the game object ID `2` is not registered with the registry.")
}
