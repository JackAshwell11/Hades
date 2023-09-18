// External includes
#include "gtest/gtest.h"

// Custom includes
#include "macros.hpp"
#include "game_objects/systems/attributes.hpp"

// ----- CLASSES ------------------------------
/// Represents a full game object attribute useful for testing.
class FullGameObjectAttribute : public GameObjectAttributeBase {
 public:
  /// Initialise the object.
  ///
  /// @param initial_value - The initial value of the view distance attribute.
  /// @param level_limit - The level limit of the view distance attribute.
  FullGameObjectAttribute(double initial_value, int level_limit) : GameObjectAttributeBase(initial_value,
                                                                                           level_limit) {}
};

/// Represents an empty game object attribute useful for testing.
class EmptyGameObjectAttribute : public GameObjectAttributeBase {
 public:
  /// Initialise the object.
  ///
  /// @param initial_value - The initial value of the view distance attribute.
  /// @param level_limit - The level limit of the view distance attribute.
  EmptyGameObjectAttribute(double initial_value, int level_limit) : GameObjectAttributeBase(initial_value,
                                                                                            level_limit) {}

  /// Get if the game object attribute can have instant effects or not.
  ///
  /// @return Whether the game object attribute can have instant effects or not.
  [[nodiscard]] bool has_instant_effect() const final { return false; }

  /// Get if the game object attribute has a maximum value or not.
  ///
  /// @return Whether the game object attribute has a maximum value or not.
  [[nodiscard]] bool has_maximum() const final { return false; }

  /// Get if the game object attribute can have status effects or not.
  ///
  /// @return Whether the game object attribute can have status effects or not.
  [[nodiscard]] bool has_status_effect() const final { return false; }

  /// Get if the game object attribute can be upgraded or not.
  ///
  /// @return Whether the game object attribute can be upgraded or not.
  [[nodiscard]] bool is_upgradable() const final { return false; }
};

// ----- FIXTURES ------------------------------
/// Implements the GameObjectAttribute fixture for the game_objects/systems/attacks.hpp tests.
class GameObjectAttributeFixture : public testing::Test {
 protected:
  /// The registry that manages the game objects, components, and systems.
  Registry registry{};

  /// Set up the fixture for the tests.
  void SetUp() override {
    registry.add_system(std::make_shared<GameObjectAttributeSystem>(registry));
  }

  /// Create two game object attributes for use in testing.
  void create_game_object_attributes() {
    std::vector<std::unique_ptr<ComponentBase>> components;
    components.push_back(std::make_unique<FullGameObjectAttribute>(150, 3));
    components.push_back(std::make_unique<EmptyGameObjectAttribute>(100, 5));
    registry.create_game_object(false, std::move(components));
  }

  /// Get the game object attribute system from the registry.
  ///
  /// @return The game object attribute system.
  std::shared_ptr<GameObjectAttributeSystem> get_game_object_attribute_system() {
    return registry.find_system<GameObjectAttributeSystem>();
  }
};

// ----- TESTS ----------------------------------
/// Test that a game object is updated correctly with a small delta time.
TEST_F(GameObjectAttributeFixture, TestGameObjectAttributeSystemUpdateSmallDeltaTime) {
  create_game_object_attributes();
  std::tuple<std::function<double(int)>, std::function<double(int)>>
      status_effect = std::make_tuple([](int level) { return level * 2; }, [](int level) { return level + 100; });
  get_game_object_attribute_system()->apply_status_effect<FullGameObjectAttribute>(0, status_effect, 2);
  get_game_object_attribute_system()->update(5);
  auto full_game_object_attribute = registry.get_component<FullGameObjectAttribute>(0);
  ASSERT_EQ(full_game_object_attribute->value(), 154);
  ASSERT_EQ(full_game_object_attribute->max_value(), 154);
  ASSERT_EQ(full_game_object_attribute->applied_status_effect->time_counter, 5);
}

/// Test that a status effect is removed with a large delta time.
TEST_F(GameObjectAttributeFixture, TestGameObjectAttributeSystemUpdateLargeDeltaTime) {
  create_game_object_attributes();
  std::tuple<std::function<double(int)>, std::function<double(int)>>
      status_effect = std::make_tuple([](int level) { return std::pow(level, 2); }, [](int level) { return 20; });
  get_game_object_attribute_system()->apply_status_effect<FullGameObjectAttribute>(0, status_effect, 2);
  get_game_object_attribute_system()->update(30);
  auto full_game_object_attribute = registry.get_component<FullGameObjectAttribute>(0);
  ASSERT_EQ(full_game_object_attribute->value(), 150);
  ASSERT_EQ(full_game_object_attribute->max_value(), 150);
  ASSERT_FALSE(full_game_object_attribute->applied_status_effect.has_value());
}

/// Test that a status effect is removed after multiple updates.
TEST_F(GameObjectAttributeFixture, TestGameObjectAttributeSystemUpdateMultipleDeltaTimes) {
  create_game_object_attributes();
  std::tuple<std::function<double(int)>, std::function<double(int)>> status_effect =
      std::make_tuple([](int level) { return 50 + level / 2; }, [](int level) { return std::pow(level, 3) + 50; });
  get_game_object_attribute_system()->apply_status_effect<FullGameObjectAttribute>(0, status_effect, 3);
  get_game_object_attribute_system()->update(40);
  auto full_game_object_attribute = registry.get_component<FullGameObjectAttribute>(0);
  ASSERT_EQ(full_game_object_attribute->value(), 201.5);
  ASSERT_EQ(full_game_object_attribute->max_value(), 201.5);
  ASSERT_EQ(full_game_object_attribute->applied_status_effect->time_counter, 40);
  get_game_object_attribute_system()->update(40);
  ASSERT_EQ(full_game_object_attribute->value(), 150);
  ASSERT_EQ(full_game_object_attribute->max_value(), 150);
  ASSERT_FALSE(full_game_object_attribute->applied_status_effect.has_value());
}

/// Test that a status effect is removed when value is less than the original.
TEST_F(GameObjectAttributeFixture, TestGameObjectAttributeSystemUpdateLessThanOriginal) {
  create_game_object_attributes();
  std::tuple<std::function<double(int)>, std::function<double(int)>>
      status_effect = std::make_tuple([](int level) { return 100; }, [](int level) { return 20; });
  get_game_object_attribute_system()->apply_status_effect<FullGameObjectAttribute>(0, status_effect, 4);
  auto full_game_object_attribute = registry.get_component<FullGameObjectAttribute>(0);
  full_game_object_attribute->value(full_game_object_attribute->value() - 200);
  get_game_object_attribute_system()->update(30);
  ASSERT_EQ(full_game_object_attribute->value(), 50);
  ASSERT_EQ(full_game_object_attribute->max_value(), 150);
  ASSERT_FALSE(full_game_object_attribute->applied_status_effect.has_value());
}

/// Test that a game object is updated if a status effect does not exist.
TEST_F(GameObjectAttributeFixture, TestGameObjectAttributeSystemUpdateNoStatusEffect) {
  create_game_object_attributes();
  auto full_game_object_attribute = registry.get_component<FullGameObjectAttribute>(0);
  ASSERT_FALSE(full_game_object_attribute->applied_status_effect.has_value());
  get_game_object_attribute_system()->update(5);
  ASSERT_FALSE(full_game_object_attribute->applied_status_effect.has_value());
}

/// Test that updating a game object fails on an empty game object attribute.
TEST_F(GameObjectAttributeFixture, TestGameObjectAttributeSystemUpdateEmptyGameObjectAttribute) {
  create_game_object_attributes();
  auto empty_game_object_attribute = registry.get_component<EmptyGameObjectAttribute>(0);
  ASSERT_FALSE(empty_game_object_attribute->applied_status_effect.has_value());
  get_game_object_attribute_system()->update(5);
  ASSERT_FALSE(empty_game_object_attribute->applied_status_effect.has_value());
}

// TODO: Reformat this (need to get clang-format ready for this)

/// Test that an instant effect is not applied if the value is equal to the max.
TEST_F(GameObjectAttributeFixture, TestGameObjectAttributeSystemApplyInstantEffectEqual) {
  create_game_object_attributes();
  ASSERT_FALSE(get_game_object_attribute_system()->apply_instant_effect<FullGameObjectAttribute>(0,
                                                                                                 [](int level) { return 50; },
                                                                                                 3));
  ASSERT_EQ(registry.get_component<FullGameObjectAttribute>(0)->value(), 150);
}

/// Test that an instant effect is applied if the value is lower than the max.
TEST_F(GameObjectAttributeFixture, TestGameObjectAttributeSystemApplyInstantEffectLower) {
  create_game_object_attributes();
  auto full_game_object_attribute = registry.get_component<FullGameObjectAttribute>(0);
  full_game_object_attribute->value(full_game_object_attribute->value() - 50);
  ASSERT_TRUE(get_game_object_attribute_system()->apply_instant_effect<FullGameObjectAttribute>(0, [](int level) {
                                                                                                  return 10 * level;
                                                                                                },
                                                                                                2));
  ASSERT_EQ(full_game_object_attribute->value(), 120);
}

/// Test that an empty game object attribute fails when applied an instant effect.
TEST_F(GameObjectAttributeFixture, TestGameObjectAttributeSystemApplyInstantEffectEmptyGameObjectAttribute) {
  create_game_object_attributes();
  ASSERT_FALSE(get_game_object_attribute_system()->apply_instant_effect<EmptyGameObjectAttribute>(0,
                                                                                                  [](int level) { return 0; },
                                                                                                  3));
}

/// Test that an exception is raised if an invalid game object ID is provided.
TEST_F(GameObjectAttributeFixture, TestGameObjectAttributeSystemApplyInstantEffectInvalidGameObjectId) {
  create_game_object_attributes();
  ASSERT_THROW_MESSAGE(get_game_object_attribute_system()->apply_instant_effect<FullGameObjectAttribute>(-1,
                                                                                                         [](int level) { return 0; },
                                                                                                         3),
                       RegistryException,
                       "The game object `-1` is not registered with the registry.")
}

/// Test that a status effect is applied if no status effect is currently applied.
TEST_F(GameObjectAttributeFixture, TestGameObjectAttributeSystemApplyStatusEffectNoAppliedEffect) {
  create_game_object_attributes();
  std::tuple<std::function<double(int)>, std::function<double(int)>> status_effect =
      std::make_tuple([](int level) { return 150 + std::pow(level, 3); }, [](int level) { return 20 + 10 * level; });
  ASSERT_TRUE(get_game_object_attribute_system()->apply_status_effect<FullGameObjectAttribute>(0, status_effect, 2));
  auto full_game_object_attribute = registry.get_component<FullGameObjectAttribute>(0);
  ASSERT_EQ(full_game_object_attribute->value(), 308);
  ASSERT_EQ(full_game_object_attribute->max_value(), 308);
  auto applied_status_effect = full_game_object_attribute->applied_status_effect.value();
  ASSERT_EQ(applied_status_effect.value, 158);
  ASSERT_EQ(applied_status_effect.duration, 40);
  ASSERT_EQ(applied_status_effect.original_value, 150);
  ASSERT_EQ(applied_status_effect.original_max_value, 150);
  ASSERT_EQ(applied_status_effect.time_counter, 0);
}

/// Test that a status effect is applied if the value is lower than the max.
TEST_F(GameObjectAttributeFixture, TestGameObjectAttributeSystemApplyStatusEffectValueLowerMax) {
  create_game_object_attributes();
  auto full_game_object_attribute = registry.get_component<FullGameObjectAttribute>(0);
  full_game_object_attribute->value(full_game_object_attribute->value() - 20);
  std::tuple<std::function<double(int)>, std::function<double(int)>> status_effect =
      std::make_tuple([](int level) { return 20 * level; }, [](int level) { return 10 - std::pow(level, 2); });
  ASSERT_TRUE(get_game_object_attribute_system()->apply_status_effect<FullGameObjectAttribute>(0, status_effect, 3));
  ASSERT_TRUE(full_game_object_attribute->applied_status_effect.has_value());
}

/// Test that a status effect is not applied if a status effect is already applied.
TEST_F(GameObjectAttributeFixture, TestGameObjectAttributeSystemApplyStatusEffectExistingStatusEffect) {
  create_game_object_attributes();
  std::tuple<std::function<double(int)>, std::function<double(int)>>
      status_effect = std::make_tuple([](int level) { return 0; }, [](int level) { return 0; });
  get_game_object_attribute_system()->apply_status_effect<FullGameObjectAttribute>(0, status_effect, 3);
  ASSERT_TRUE(registry.get_component<FullGameObjectAttribute>(0)->applied_status_effect.has_value());
  ASSERT_FALSE(get_game_object_attribute_system()->apply_status_effect<FullGameObjectAttribute>(0, status_effect, 2));
}

/// Test that an empty game object attribute fails when applied a status effect.
TEST_F(GameObjectAttributeFixture, TestGameObjectAttributeSystemApplyStatusEffectEmptyGameObjectAttribute) {
  create_game_object_attributes();
  std::tuple<std::function<double(int)>, std::function<double(int)>>
      status_effect = std::make_tuple([](int level) { return 0; }, [](int level) { return 0; });
  ASSERT_FALSE(get_game_object_attribute_system()->apply_status_effect<EmptyGameObjectAttribute>(0, status_effect, 3));
}

/// Test that an exception is raised if an invalid game object ID is provided.
TEST_F(GameObjectAttributeFixture, TestGameObjectAttributeSystemApplyStatusEffectInvalidGameObjectId) {
  create_game_object_attributes();
  std::tuple<std::function<double(int)>, std::function<double(int)>>
      status_effect = std::make_tuple([](int level) { return 0; }, [](int level) { return 0; });
  ASSERT_THROW_MESSAGE(get_game_object_attribute_system()->apply_status_effect<FullGameObjectAttribute>(-1,
                                                                                                        status_effect,
                                                                                                        3),
                       RegistryException,
                       "The game object `-1` is not registered with the registry.")
}
