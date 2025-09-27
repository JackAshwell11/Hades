// External headers
#include <nlohmann/json.hpp>

// Local headers
#include "ecs/registry.hpp"
#include "ecs/stats.hpp"
#include "ecs/systems/effects.hpp"
#include "events.hpp"
#include "macros.hpp"

/// Implements the fixture for the EffectSystem tests.
class EffectSystemFixture : public testing::Test {
 protected:
  /// The configuration for the effect applier.
  struct EffectApplierConfig {
    /// Whether the instant effect is enabled or not.
    bool instant = false;

    /// Whether the status effect is enabled or not.
    bool status = false;

    /// Whether the negative effect is enabled or not.
    bool negative = false;
  };

  /// The registry that manages the game objects, components, and systems.
  Registry registry;

  /// Set up the fixture for the tests.
  void SetUp() override {
    registry.add_system<EffectSystem>();
    const auto game_object_id{registry.create_game_object(GameObjectType::Player)};
    registry.add_component<Health>(game_object_id, 200, -1);
    registry.add_component<StatusEffects>(game_object_id);
  }

  /// Tear down the fixture after the tests.
  void TearDown() override { clear_listeners(); }

  /// Create a game object to hold the instant and status effects.
  ///
  /// @param config - The configuration for the effect applier.
  void create_effect_applier(const EffectApplierConfig& config) {
    const auto game_object_id{registry.create_game_object(GameObjectType::Enemy)};
    registry.add_component<EffectApplier>(game_object_id);
    const auto effect_applier{registry.get_component<EffectApplier>(game_object_id)};
    if (config.instant) {
      effect_applier->add_instant_effect(EffectType::Regeneration, 5);
    }
    if (config.status) {
      effect_applier->add_status_effect(EffectType::Regeneration, 5, 10, 2);
    }
    if (config.negative) {
      effect_applier->add_instant_effect(EffectType::Poison, -5);
    }
  }

  /// Get the effect system from the registry.
  ///
  /// @return The effect system.
  [[nodiscard]] auto get_effect_system() const -> std::shared_ptr<EffectSystem> {
    return registry.get_system<EffectSystem>();
  }
};

/// Test that the status effects component is reset correctly.
TEST_F(EffectSystemFixture, TestStatusEffectsReset) {
  create_effect_applier({.status = true});
  registry.get_component<Health>(0)->set_value(100);
  (void)get_effect_system()->apply_effects(1, 0);
  const auto status_effects{registry.get_component<StatusEffects>(0)};
  ASSERT_FALSE(status_effects->active_effects.empty());
  status_effects->reset();
  ASSERT_TRUE(status_effects->active_effects.empty());
}

/// Test that the status effects component is serialised to a file correctly.
TEST_F(EffectSystemFixture, TestStatusEffectsToFile) {
  create_effect_applier({.status = true});
  registry.get_component<Health>(0)->set_value(100);
  (void)get_effect_system()->apply_effects(1, 0);
  get_effect_system()->update(5);
  nlohmann::json json;
  registry.get_component<StatusEffects>(0)->to_file(json);
  ASSERT_EQ(json, nlohmann::json::parse(R"({
    "active_effects": {
      "0": {
        "value": 5.0,
        "duration": 10.0,
        "interval": 2.0,
        "time_elapsed": 5.0,
        "interval_accumulator": 1.0
      }
    }
  })"));
}

/// Test that the status effects component is deserialised from a file correctly.
TEST_F(EffectSystemFixture, TestStatusEffectsFromFile) {
  const nlohmann::json json(nlohmann::json::parse(R"({
    "active_effects": {
      "0": {
        "value": 5.0,
        "duration": 10.0,
        "interval": 2.0,
        "time_elapsed": 5.0,
        "interval_accumulator": 1.0
      }, "1": {
        "value": -5.0,
        "duration": 10.0,
        "interval": 2.0,
        "time_elapsed": 3.0,
        "interval_accumulator": 0.5
      }
    }
  })"));
  const auto status_effects{registry.get_component<StatusEffects>(0)};
  status_effects->from_file(json);
  ASSERT_EQ(status_effects->active_effects.size(), 2);
  const auto regeneration_effect{status_effects->active_effects.at(EffectType::Regeneration)};
  ASSERT_EQ(regeneration_effect.value, 5.0);
  ASSERT_EQ(regeneration_effect.duration, 10.0);
  ASSERT_EQ(regeneration_effect.interval, 2.0);
  ASSERT_EQ(regeneration_effect.time_elapsed, 5.0);
  ASSERT_EQ(regeneration_effect.interval_accumulator, 1.0);
  const auto poison_effect{status_effects->active_effects.at(EffectType::Poison)};
  ASSERT_EQ(poison_effect.value, -5.0);
  ASSERT_EQ(poison_effect.duration, 10.0);
  ASSERT_EQ(poison_effect.interval, 2.0);
  ASSERT_EQ(poison_effect.time_elapsed, 3.0);
  ASSERT_EQ(poison_effect.interval_accumulator, 0.5);
}

/// Test that a status effect is updated correctly with a small delta time.
TEST_F(EffectSystemFixture, TestEffectSystemUpdateSmallDeltaTime) {
  create_effect_applier({.status = true});
  const auto health{registry.get_component<Health>(0)};
  health->set_value(100);
  ASSERT_TRUE(get_effect_system()->apply_effects(1, 0));
  ASSERT_EQ(health->get_value(), 105);
  get_effect_system()->update(2);
  ASSERT_EQ(health->get_value(), 110);
  ASSERT_EQ(registry.get_component<StatusEffects>(0)->active_effects.at(EffectType::Regeneration).time_elapsed, 2);
}

/// Test that a status effect is updated correctly with a large delta time.
TEST_F(EffectSystemFixture, TestEffectSystemUpdateLargeDeltaTime) {
  create_effect_applier({.status = true});
  const auto health{registry.get_component<Health>(0)};
  health->set_value(100);
  ASSERT_TRUE(get_effect_system()->apply_effects(1, 0));
  ASSERT_EQ(health->get_value(), 105);
  get_effect_system()->update(20);
  ASSERT_EQ(health->get_value(), 130);
  ASSERT_TRUE(registry.get_component<StatusEffects>(0)->active_effects.empty());
}

/// Test that a status effect is updated correctly after multiple updates.
TEST_F(EffectSystemFixture, TestEffectSystemUpdateMultipleDeltaTimes) {
  create_effect_applier({.status = true});
  const auto health{registry.get_component<Health>(0)};
  health->set_value(100);
  ASSERT_TRUE(get_effect_system()->apply_effects(1, 0));
  ASSERT_EQ(health->get_value(), 105);
  const auto& active_effects{registry.get_component<StatusEffects>(0)->active_effects};
  get_effect_system()->update(1);
  ASSERT_EQ(health->get_value(), 105);
  ASSERT_EQ(active_effects.at(EffectType::Regeneration).time_elapsed, 1);
  get_effect_system()->update(5);
  ASSERT_EQ(health->get_value(), 120);
  ASSERT_EQ(active_effects.at(EffectType::Regeneration).time_elapsed, 6);
  get_effect_system()->update(2);
  ASSERT_EQ(health->get_value(), 125);
  ASSERT_EQ(active_effects.at(EffectType::Regeneration).time_elapsed, 8);
  get_effect_system()->update(3);
  ASSERT_EQ(health->get_value(), 130);
  ASSERT_TRUE(active_effects.empty());
}

/// Test that multiple status effects are updated correctly.
TEST_F(EffectSystemFixture, TestEffectSystemUpdateMultipleStatusEffects) {
  create_effect_applier({.status = true});
  const auto health{registry.get_component<Health>(0)};
  health->set_value(100);
  registry.get_component<EffectApplier>(1)->add_status_effect(EffectType::Poison, -1, 1, 1);
  ASSERT_TRUE(get_effect_system()->apply_effects(1, 0));
  ASSERT_EQ(health->get_value(), 104);
  ASSERT_EQ(registry.get_component<StatusEffects>(0)->active_effects.size(), 2);
  get_effect_system()->update(2);
  ASSERT_EQ(health->get_value(), 108);
  ASSERT_EQ(registry.get_component<StatusEffects>(0)->active_effects.size(), 1);
}

/// Test that a status effect is not updated if one does not exist.
TEST_F(EffectSystemFixture, TestEffectSystemUpdateNoStatusEffect) {
  create_effect_applier({});
  const auto health{registry.get_component<Health>(0)};
  health->set_value(100);
  get_effect_system()->update(5);
  ASSERT_EQ(health->get_value(), 100);
}

/// Test that the status effect update callback is called correctly.
TEST_F(EffectSystemFixture, TestEffectSystemUpdateStatusEffectCallback) {
  std::unordered_map<EffectType, double> callback_args;
  auto status_effect_update_callback{
      [&callback_args](std::unordered_map<EffectType, double> effects) { callback_args = std::move(effects); }};
  add_callback<EventType::StatusEffectUpdate>(status_effect_update_callback);
  create_effect_applier({.status = true});
  const auto health{registry.get_component<Health>(0)};
  health->set_value(100);
  ASSERT_TRUE(get_effect_system()->apply_effects(1, 0));
  get_effect_system()->update(5);
  ASSERT_EQ(callback_args.size(), 1);
  ASSERT_EQ(callback_args.at(EffectType::Regeneration), 5);
}

/// Test that an instant effect is applied correctly if the value equals the maximum.
TEST_F(EffectSystemFixture, TestEffectSystemApplyEffectsInstantValueEqualMax) {
  create_effect_applier({.instant = true});
  ASSERT_FALSE(get_effect_system()->apply_effects(1, 0));
  ASSERT_EQ(registry.get_component<Health>(0)->get_value(), 200);
}

/// Test that an instant effect is applied correctly if the value is lower than the maximum.
TEST_F(EffectSystemFixture, TestEffectSystemApplyEffectsInstantValueLowerMax) {
  create_effect_applier({.instant = true});
  const auto health{registry.get_component<Health>(0)};
  health->set_value(150);
  ASSERT_TRUE(get_effect_system()->apply_effects(1, 0));
  ASSERT_EQ(health->get_value(), 155);
}

/// Test that a status effect is applied correctly if the value equals the maximum.
TEST_F(EffectSystemFixture, TestEffectSystemApplyEffectsStatusValueEqualMax) {
  create_effect_applier({.status = true});
  ASSERT_FALSE(get_effect_system()->apply_effects(1, 0));
  ASSERT_EQ(registry.get_component<Health>(0)->get_value(), 200);
}

/// Test that a status effect is applied correctly if the value is lower than the max.
TEST_F(EffectSystemFixture, TestEffectSystemApplyEffectsStatusValueLowerMax) {
  create_effect_applier({.status = true});
  const auto health{registry.get_component<Health>(0)};
  health->set_value(150);
  ASSERT_TRUE(get_effect_system()->apply_effects(1, 0));
  ASSERT_EQ(health->get_value(), 155);
}

/// Test that a negative instant effect is applied correctly if the value equal 0.
TEST_F(EffectSystemFixture, TestEffectSystemApplyEffectsNegativeValueEqualZero) {
  create_effect_applier({.negative = true});
  const auto health{registry.get_component<Health>(0)};
  health->set_value(0);
  ASSERT_FALSE(get_effect_system()->apply_effects(1, 0));
  ASSERT_EQ(health->get_value(), 0);
}

/// Test that a negative instant effect is applied correctly if the value is greater than 0.
TEST_F(EffectSystemFixture, TestEffectSystemApplyEffectsNegativeValueGreaterZero) {
  create_effect_applier({.negative = true});
  ASSERT_TRUE(get_effect_system()->apply_effects(1, 0));
  ASSERT_EQ(registry.get_component<Health>(0)->get_value(), 195);
}

/// Test that multiple different status effects are applied correctly.
TEST_F(EffectSystemFixture, TestEffectSystemApplyEffectsStatusMultipleDifferentStatusEffects) {
  create_effect_applier({.status = true});
  const auto health{registry.get_component<Health>(0)};
  health->set_value(100);
  registry.get_component<EffectApplier>(1)->add_status_effect(EffectType::Poison, -1, 1, 1);
  ASSERT_TRUE(get_effect_system()->apply_effects(1, 0));
  ASSERT_EQ(health->get_value(), 104);
  ASSERT_EQ(registry.get_component<StatusEffects>(0)->active_effects.size(), 2);
}

/// Test that multiple identical status effects are applied correctly.
TEST_F(EffectSystemFixture, TestEffectSystemApplyEffectsStatusMultipleIdenticalStatusEffects) {
  create_effect_applier({.status = true});
  registry.get_component<Health>(0)->set_value(100);
  registry.get_component<EffectApplier>(1)->add_status_effect(EffectType::Regeneration, 1, 1, 1);
  ASSERT_TRUE(get_effect_system()->apply_effects(1, 0));
  ASSERT_EQ(registry.get_component<Health>(0)->get_value(), 105);
  ASSERT_EQ(registry.get_component<StatusEffects>(0)->active_effects.at(EffectType::Regeneration).duration, 11);
  ASSERT_EQ(registry.get_component<StatusEffects>(0)->active_effects.size(), 1);
}

/// Test that multiple instant and status effects are applied correctly.
TEST_F(EffectSystemFixture, TestEffectSystemApplyEffectsStatusMultipleInstantAndStatusEffects) {
  create_effect_applier({.instant = true, .status = true});
  const auto health{registry.get_component<Health>(0)};
  health->set_value(100);
  registry.get_component<EffectApplier>(1)->add_status_effect(EffectType::Poison, -1, 1, 1);
  ASSERT_TRUE(get_effect_system()->apply_effects(1, 0));
  ASSERT_EQ(registry.get_component<StatusEffects>(0)->active_effects.size(), 2);
  ASSERT_EQ(health->get_value(), 109);
}

/// Test that the status effect update callback is called correctly.
TEST_F(EffectSystemFixture, TestEffectSystemApplyEffectsStatusEffectCallback) {
  std::unordered_map<EffectType, double> callback_args;
  auto status_effect_update_callback{
      [&callback_args](std::unordered_map<EffectType, double> effects) { callback_args = std::move(effects); }};
  add_callback<EventType::StatusEffectUpdate>(status_effect_update_callback);
  create_effect_applier({.status = true});
  const auto health{registry.get_component<Health>(0)};
  health->set_value(100);
  ASSERT_TRUE(get_effect_system()->apply_effects(1, 0));
  ASSERT_EQ(callback_args.size(), 1);
  ASSERT_EQ(callback_args.at(EffectType::Regeneration), 10);
}

/// Test that an exception is thrown if the game object does not have the target component.
TEST_F(EffectSystemFixture, TestEffectSystemApplyEffectsNonexistentTargetComponent) {
  create_effect_applier({});
  registry.create_game_object(GameObjectType::Player);
  ASSERT_THROW_MESSAGE(get_effect_system()->apply_effects(1, 2), RegistryError,
                       "The component `StatusEffects` for the game object ID `2` is not registered with the registry.")
}

/// Test that an exception is thrown if an invalid source game object ID is provided.
TEST_F(EffectSystemFixture, TestEffectSystemApplyEffectsInvalidSourceGameObjectId){ASSERT_THROW_MESSAGE(
    get_effect_system()->apply_effects(-1, 0), RegistryError,
    "The component `EffectApplier` for the game object ID `-1` is not registered with the registry.")}

/// Test that an exception is thrown if an invalid target game object ID is provided.
TEST_F(EffectSystemFixture, TestEffectSystemApplyEffectsInvalidTargetGameObjectId) {
  create_effect_applier({});
  ASSERT_THROW_MESSAGE(get_effect_system()->apply_effects(1, -1), RegistryError,
                       "The component `StatusEffects` for the game object ID `-1` is not registered with the registry.")
}

/// Test that an exception is thrown if the source game object does not have an effect applier component.
TEST_F(EffectSystemFixture, TestEffectSystemApplyEffectsNonexistentEffectApplier) {
  registry.create_game_object(GameObjectType::Player);
  ASSERT_THROW_MESSAGE(get_effect_system()->apply_effects(1, 0), RegistryError,
                       "The component `EffectApplier` for the game object ID `1` is not registered with the registry.")
}

/// Test that an exception is thrown if the source game object does not have a status effects component.
TEST_F(EffectSystemFixture, TestEffectSystemApplyEffectsNonexistentStatusEffects) {
  create_effect_applier({});
  const auto game_object_id{registry.create_game_object(GameObjectType::Player)};
  registry.add_component<Health>(game_object_id, -1, -1);
  ASSERT_THROW_MESSAGE(get_effect_system()->apply_effects(1, 2), RegistryError,
                       "The component `StatusEffects` for the game object ID `2` is not registered with the registry.")
}
