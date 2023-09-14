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
/// Implements the ArmourRegen fixture for the game_objects/systems/movements.hpp tests.
class ArmourRegenFixture : public testing::Test {
 protected:
  /// The registry that manages the game objects, components, and systems.
  Registry registry{};

  /// Set up the fixture for the tests.
  void SetUp() override {
    std::vector<std::unique_ptr<ComponentBase>> components;
    components.push_back(std::make_unique<Armour>(50, 4));
    components.push_back(std::make_unique<ArmourRegenCooldown>(4, 5));
    components.push_back(std::make_unique<ArmourRegen>());
    registry.create_game_object(false, std::move(components));
    registry.add_system<ArmourRegenSystem>(std::make_unique<ArmourRegenSystem>());
  }

  /// Get the armour regen system from the registry.
  ///
  /// @return The armour regen system.
  ArmourRegenSystem *get_armour_regen_system() {
    return registry.find_system<ArmourRegenSystem>();
  }
};

/// Implements the GameObjectAttribute fixture for the game_objects/systems/attacks.hpp tests.
class GameObjectAttributeFixture : public testing::Test {
 protected:
  /// The registry that manages the game objects, components, and systems.
  Registry registry{};

  /// Set up the fixture for the tests.
  void SetUp() override {
    registry.add_system<GameObjectAttributeSystem>(std::make_unique<GameObjectAttributeSystem>());
  }

  /// Create two game object attributes for use in testing.
  void create_game_object_attributes() {
    std::vector<std::unique_ptr<ComponentBase>> components;
    components.push_back(std::make_unique<FullGameObjectAttribute>(150, 3));
    components.push_back(std::make_unique<EmptyGameObjectAttribute>(100, 5));
    registry.create_game_object(false, std::move(components));
  }

  /// Create health and armour attributes for use in testing.
  void create_health_and_armour_attributes() {
    std::vector<std::unique_ptr<ComponentBase>> components;
    components.push_back(std::make_unique<Health>(300, 4));
    components.push_back(std::make_unique<Armour>(100, 6));
    registry.create_game_object(false, std::move(components));
  }

  /// Get the game object attribute system from the registry.
  ///
  /// @return The game object attribute system.
  GameObjectAttributeSystem *get_game_object_attribute_system() {
    return registry.find_system<GameObjectAttributeSystem>();
  }
};

// ----- TESTS ----------------------------------
/// Test that the armour regen component is updated correctly when armour is full.
TEST_F(ArmourRegenFixture, TestArmourRegenSystemUpdateFullArmour) {
  get_armour_regen_system()->update(registry, 5);
  ASSERT_EQ(registry.get_component<Armour>(0)->value(), 50);
  ASSERT_EQ(registry.get_component<ArmourRegen>(0)->time_since_armour_regen, 0);
}

/// Test that the armour regen component is updated with a small delta time.
TEST_F(ArmourRegenFixture, TestArmourRegenSystemUpdateSmallDeltaTime) {
  auto *armour = registry.get_component<Armour>(0);
  armour->value(armour->value() - 10);
  get_armour_regen_system()->update(registry, 2);
  ASSERT_EQ(armour->value(), 40);
  ASSERT_EQ(registry.get_component<ArmourRegen>(0)->time_since_armour_regen, 2);
}

/// Test that the armour regen component is updated with a large delta time.
TEST_F(ArmourRegenFixture, TestArmourRegenSystemUpdateLargeDeltaTime) {
  auto *armour = registry.get_component<Armour>(0);
  armour->value(armour->value() - 10);
  get_armour_regen_system()->update(registry, 6);
  ASSERT_EQ(armour->value(), 41);
  ASSERT_EQ(registry.get_component<ArmourRegen>(0)->time_since_armour_regen, 0);
}

/// Test that the armour regen component is updated multiple times correctly.
TEST_F(ArmourRegenFixture, TestArmourRegenSystemUpdateMultipleUpdates) {
  auto *armour = registry.get_component<Armour>(0);
  armour->value(armour->value() - 10);
  get_armour_regen_system()->update(registry, 1);
  ASSERT_EQ(armour->value(), 40);
  ASSERT_EQ(registry.get_component<ArmourRegen>(0)->time_since_armour_regen, 1);
  get_armour_regen_system()->update(registry, 2);
  ASSERT_EQ(armour->value(), 40);
  ASSERT_EQ(registry.get_component<ArmourRegen>(0)->time_since_armour_regen, 3);
}

// TODO: These fail because the template types are hardcoded and the test attributes dont match to any

/// Test that a game object is updated correctly with a small delta time.
TEST_F(GameObjectAttributeFixture, TestGameObjectAttributeSystemUpdateSmallDeltaTime) {
  create_game_object_attributes();
  std::tuple<std::function<double(int)>, std::function<double(int)>>
      status_effect = std::make_tuple([](int level) { return level * 2; }, [](int level) { return level + 100; });
  GameObjectAttributeSystem::apply_status_effect<FullGameObjectAttribute>(registry, 0, status_effect, 2);
  get_game_object_attribute_system()->update(registry, 5);
  auto *full_game_object_attribute = registry.get_component<FullGameObjectAttribute>(0);
  ASSERT_EQ(full_game_object_attribute->value(), 154);
  ASSERT_EQ(full_game_object_attribute->max_value(), 154);
  ASSERT_EQ(full_game_object_attribute->applied_status_effect->time_counter, 5);
}

/// Test that a status effect is removed with a large delta time.
TEST_F(GameObjectAttributeFixture, TestGameObjectAttributeSystemUpdateLargeDeltaTime) {
  create_game_object_attributes();
  std::tuple<std::function<double(int)>, std::function<double(int)>>
      status_effect = std::make_tuple([](int level) { return std::pow(level, 2); }, [](int level) { return 20; });
  GameObjectAttributeSystem::apply_status_effect<FullGameObjectAttribute>(registry, 0, status_effect, 2);
  get_game_object_attribute_system()->update(registry, 30);
  auto *full_game_object_attribute = registry.get_component<FullGameObjectAttribute>(0);
  ASSERT_EQ(full_game_object_attribute->value(), 150);
  ASSERT_EQ(full_game_object_attribute->max_value(), 150);
  ASSERT_FALSE(full_game_object_attribute->applied_status_effect.has_value());
}

/// Test that a status effect is removed after multiple updates.
TEST_F(GameObjectAttributeFixture, TestGameObjectAttributeSystemUpdateMultipleDeltatimes) {
  create_game_object_attributes();
  std::tuple<std::function<double(int)>, std::function<double(int)>> status_effect =
      std::make_tuple([](int level) { return 50 + level / 2; }, [](int level) { return std::pow(level, 3) + 50; });
  GameObjectAttributeSystem::apply_status_effect<FullGameObjectAttribute>(registry, 0, status_effect, 3);
  get_game_object_attribute_system()->update(registry, 40);
  auto *full_game_object_attribute = registry.get_component<FullGameObjectAttribute>(0);
  ASSERT_EQ(full_game_object_attribute->value(), 201.5);
  ASSERT_EQ(full_game_object_attribute->max_value(), 201.5);
  ASSERT_EQ(full_game_object_attribute->applied_status_effect->time_counter, 40);
  get_game_object_attribute_system()->update(registry, 40);
  ASSERT_EQ(full_game_object_attribute->value(), 150);
  ASSERT_EQ(full_game_object_attribute->max_value(), 150);
  ASSERT_FALSE(full_game_object_attribute->applied_status_effect.has_value());
}

/// Test that a status effect is removed when value is less than the original.
TEST_F(GameObjectAttributeFixture, TestGameObjectAttributeSystemUpdateLessThanOriginal) {
  create_game_object_attributes();
  std::tuple<std::function<double(int)>, std::function<double(int)>>
      status_effect = std::make_tuple([](int level) { return 100; }, [](int level) { return 20; });
  GameObjectAttributeSystem::apply_status_effect<FullGameObjectAttribute>(registry, 0, status_effect, 4);
  auto *full_game_object_attribute = registry.get_component<FullGameObjectAttribute>(0);
  full_game_object_attribute->value(full_game_object_attribute->value() - 200);
  get_game_object_attribute_system()->update(registry, 30);
  ASSERT_EQ(full_game_object_attribute->value(), 50);
  ASSERT_EQ(full_game_object_attribute->max_value(), 150);
  ASSERT_FALSE(full_game_object_attribute->applied_status_effect.has_value());
}

/// Test that a game object is updated if a status effect does not exist.
TEST_F(GameObjectAttributeFixture, TestGameObjectAttributeSystemUpdateNoStatusEffect) {
  create_game_object_attributes();
  auto *full_game_object_attribute = registry.get_component<FullGameObjectAttribute>(0);
  ASSERT_FALSE(full_game_object_attribute->applied_status_effect.has_value());
  get_game_object_attribute_system()->update(registry, 5);
  ASSERT_FALSE(full_game_object_attribute->applied_status_effect.has_value());
}

/// Test that updating a game object fails on an empty game object attribute.
TEST_F(GameObjectAttributeFixture, TestGameObjectAttributeSystemUpdateEmptyGameObjectAttribute) {
  create_game_object_attributes();
  auto *empty_game_object_attribute = registry.get_component<EmptyGameObjectAttribute>(0);
  ASSERT_FALSE(empty_game_object_attribute->applied_status_effect.has_value());
  get_game_object_attribute_system()->update(registry, 5);
  ASSERT_FALSE(empty_game_object_attribute->applied_status_effect.has_value());
}

/// Test that a full game object attribute is upgraded correctly.
TEST_F(GameObjectAttributeFixture, TestGameObjectAttributeSystemUpgradeValueEqualMax) {
  create_game_object_attributes();
  GameObjectAttributeSystem::upgrade<FullGameObjectAttribute>(registry, 0, [](int level) { return 150 * (level + 1); });
  auto *full_game_object_attribute = registry.get_component<FullGameObjectAttribute>(0);
  ASSERT_EQ(full_game_object_attribute->value(), 300);
  ASSERT_EQ(full_game_object_attribute->max_value(), 300);
  ASSERT_EQ(full_game_object_attribute->current_level(), 1);
}

/// Test that a full game object attribute is upgraded if value is lower than max.
TEST_F(GameObjectAttributeFixture, TestGameObjectAttributeSystemUpgradeValueLowerMax) {
  create_game_object_attributes();
  auto *full_game_object_attribute = registry.get_component<FullGameObjectAttribute>(0);
  full_game_object_attribute->value(full_game_object_attribute->value() - 50);
  GameObjectAttributeSystem::upgrade<FullGameObjectAttribute>(registry,
                                                              0,
                                                              [](int level) { return 150 + std::pow(level, 2); });
  ASSERT_EQ(full_game_object_attribute->value(), 101);
  ASSERT_EQ(full_game_object_attribute->max_value(), 151);
  ASSERT_EQ(full_game_object_attribute->current_level(), 1);
}

/// Test that a full game object attribute is not upgraded if level limit is reached.
TEST_F(GameObjectAttributeFixture, TestGameObjectAttributeSystemUpgradeMaxLimit) {
  create_game_object_attributes();
  GameObjectAttributeSystem::upgrade<FullGameObjectAttribute>(registry, 0, [](int level) { return 0; });
  GameObjectAttributeSystem::upgrade<FullGameObjectAttribute>(registry, 0, [](int level) { return 0; });
  GameObjectAttributeSystem::upgrade<FullGameObjectAttribute>(registry, 0, [](int level) { return 0; });
  ASSERT_EQ(registry.get_component<FullGameObjectAttribute>(0)->current_level(), 3);
  ASSERT_FALSE(GameObjectAttributeSystem::upgrade<FullGameObjectAttribute>(registry, 0, [](int level) { return 0; }));
}

/// Test that an empty game object attribute fails when upgraded.
TEST_F(GameObjectAttributeFixture, TestGameObjectAttributeSystemUpgradeEmptyGameObjectAttribute) {
  create_game_object_attributes();
  ASSERT_FALSE(GameObjectAttributeSystem::upgrade<EmptyGameObjectAttribute>(registry, 0, [](int level) { return 0; }));
}

// TODO: Reformat this (need to get clang-format ready for this)

/// Test that an exception is raised if an invalid game object ID is provided.
TEST_F(GameObjectAttributeFixture, TestGameObjectAttributeSystemUpgradeInvalidGameObjectId) {
  create_game_object_attributes();
  ASSERT_THROW_MESSAGE(GameObjectAttributeSystem::upgrade<FullGameObjectAttribute>(registry,
                                                                                   -1,
                                                                                   [](int level) { return 0; }),
                       RegistryException,
                       "The game object `-1` is not registered with the registry.")
}

/// Test that an instant effect is not applied if the value is equal to the max.
TEST_F(GameObjectAttributeFixture, TestGameObjectAttributeSystemApplyInstantEffectEqual) {
  create_game_object_attributes();
  ASSERT_FALSE(GameObjectAttributeSystem::apply_instant_effect<FullGameObjectAttribute>(registry,
                                                                                        0,
                                                                                        [](int level) { return 50; },
                                                                                        3));
  ASSERT_EQ(registry.get_component<FullGameObjectAttribute>(0)->value(), 150);
}

/// Test that an instant effect is applied if the value is lower than the max.
TEST_F(GameObjectAttributeFixture, TestGameObjectAttributeSystemApplyInstantEffectLower) {
  create_game_object_attributes();
  auto *full_game_object_attribute = registry.get_component<FullGameObjectAttribute>(0);
  full_game_object_attribute->value(full_game_object_attribute->value() - 50);
  ASSERT_TRUE(GameObjectAttributeSystem::apply_instant_effect<FullGameObjectAttribute>(registry,
                                                                                       0,
                                                                                       [](int level) {
                                                                                         return 10 * level;
                                                                                       },
                                                                                       2));
  ASSERT_EQ(full_game_object_attribute->value(), 120);
}

/// Test that an empty game object attribute fails when applied an instant effect.
TEST_F(GameObjectAttributeFixture, TestGameObjectAttributeSystemApplyInstantEffectEmptyGameObjectAttribute) {
  create_game_object_attributes();
  ASSERT_FALSE(GameObjectAttributeSystem::apply_instant_effect<EmptyGameObjectAttribute>(registry,
                                                                                         0,
                                                                                         [](int level) { return 0; },
                                                                                         3));
}

/// Test that an exception is raised if an invalid game object ID is provided.
TEST_F(GameObjectAttributeFixture, TestGameObjectAttributeSystemApplyInstantEffectInvalidGameObjectId) {
  create_game_object_attributes();
  ASSERT_THROW_MESSAGE(GameObjectAttributeSystem::apply_instant_effect<FullGameObjectAttribute>(registry,
                                                                                                -1,
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
  ASSERT_TRUE(GameObjectAttributeSystem::apply_status_effect<FullGameObjectAttribute>(registry, 0, status_effect, 2));
  auto *full_game_object_attribute = registry.get_component<FullGameObjectAttribute>(0);
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
  auto *full_game_object_attribute = registry.get_component<FullGameObjectAttribute>(0);
  full_game_object_attribute->value(full_game_object_attribute->value() - 20);
  std::tuple<std::function<double(int)>, std::function<double(int)>> status_effect =
      std::make_tuple([](int level) { return 20 * level; }, [](int level) { return 10 - std::pow(level, 2); });
  ASSERT_TRUE(GameObjectAttributeSystem::apply_status_effect<FullGameObjectAttribute>(registry, 0, status_effect, 3));
  ASSERT_TRUE(full_game_object_attribute->applied_status_effect.has_value());
}

/// Test that a status effect is not applied if a status effect is already applied.
TEST_F(GameObjectAttributeFixture, TestGameObjectAttributeSystemApplyStatusEffectExistingStatusEffect) {
  create_game_object_attributes();
  std::tuple<std::function<double(int)>, std::function<double(int)>>
      status_effect = std::make_tuple([](int level) { return 0; }, [](int level) { return 0; });
  GameObjectAttributeSystem::apply_status_effect<FullGameObjectAttribute>(registry, 0, status_effect, 3);
  ASSERT_TRUE(registry.get_component<FullGameObjectAttribute>(0)->applied_status_effect.has_value());
  ASSERT_FALSE(GameObjectAttributeSystem::apply_status_effect<FullGameObjectAttribute>(registry, 0, status_effect, 2));
}

/// Test that an empty game object attribute fails when applied a status effect.
TEST_F(GameObjectAttributeFixture, TestGameObjectAttributeSystemApplyStatusEffectEmptyGameObjectAttribute) {
  create_game_object_attributes();
  std::tuple<std::function<double(int)>, std::function<double(int)>>
      status_effect = std::make_tuple([](int level) { return 0; }, [](int level) { return 0; });
  ASSERT_FALSE(GameObjectAttributeSystem::apply_status_effect<EmptyGameObjectAttribute>(registry, 0, status_effect, 3));
}

/// Test that an exception is raised if an invalid game object ID is provided.
TEST_F(GameObjectAttributeFixture, TestGameObjectAttributeSystemApplyStatusEffectInvalidGameObjectId) {
  create_game_object_attributes();
  std::tuple<std::function<double(int)>, std::function<double(int)>>
      status_effect = std::make_tuple([](int level) { return 0; }, [](int level) { return 0; });
  ASSERT_THROW_MESSAGE(GameObjectAttributeSystem::apply_status_effect<FullGameObjectAttribute>(registry,
                                                                                               -1,
                                                                                               status_effect,
                                                                                               3),
                       RegistryException,
                       "The game object `-1` is not registered with the registry.")
}

/// Test that damage is dealt when health and armour are lower than damage.
TEST_F(GameObjectAttributeFixture, TestGameObjectAttributeSystemDealDamageLowHealthArmour) {
  create_health_and_armour_attributes();
  GameObjectAttributeSystem::deal_damage(registry, 0, 350);
  ASSERT_EQ(registry.get_component<Health>(0)->value(), 50);
  ASSERT_EQ(registry.get_component<Armour>(0)->value(), 0);
}

/// Test that no damage is dealt when armour is larger than damage.
TEST_F(GameObjectAttributeFixture, TestGameObjectAttributeSystemDealDamageLargeArmour) {
  create_health_and_armour_attributes();
  GameObjectAttributeSystem::deal_damage(registry, 0, 50);
  ASSERT_EQ(registry.get_component<Health>(0)->value(), 300);
  ASSERT_EQ(registry.get_component<Armour>(0)->value(), 50);
}

/// Test that no damage is dealt when damage is zero.
TEST_F(GameObjectAttributeFixture, TestGameObjectAttributeSystemDealDamageZeroDamage) {
  create_health_and_armour_attributes();
  GameObjectAttributeSystem::deal_damage(registry, 0, 0);
  ASSERT_EQ(registry.get_component<Health>(0)->value(), 300);
  ASSERT_EQ(registry.get_component<Armour>(0)->value(), 100);
}

/// Test that damage is dealt when armour is zero.
TEST_F(GameObjectAttributeFixture, TestGameObjectAttributeSystemDealDamageZeroArmour) {
  create_health_and_armour_attributes();
  auto *armour = registry.get_component<Armour>(0);
  armour->value(0);
  GameObjectAttributeSystem::deal_damage(registry, 0, 100);
  ASSERT_EQ(registry.get_component<Health>(0)->value(), 200);
  ASSERT_EQ(armour->value(), 0);
}

/// Test that damage is dealt when health is zero.
TEST_F(GameObjectAttributeFixture, TestGameObjectAttributeSystemDealDamageZeroHealth) {
  create_health_and_armour_attributes();
  auto *health = registry.get_component<Health>(0);
  health->value(0);
  GameObjectAttributeSystem::deal_damage(registry, 0, 50);
  ASSERT_EQ(health->value(), 0);
  ASSERT_EQ(registry.get_component<Armour>(0)->value(), 50);
}

/// Test that no damage is dealt when the attributes are not initialised.
TEST_F(GameObjectAttributeFixture, TestGameObjectAttributeSystemDealDamageNonexistentAttributes) {
  registry.create_game_object(false, {});
  ASSERT_THROW_MESSAGE(GameObjectAttributeSystem::deal_damage(registry, 0, 100),
                       RegistryException,
                       "The game object `0` is not registered with the registry.")
}

/// Test that an exception is raised if an invalid game object ID is provided.
TEST_F(GameObjectAttributeFixture, TestGameObjectAttributeSystemDealDamageInvalidGameObjectId) {
  ASSERT_THROW_MESSAGE(GameObjectAttributeSystem::deal_damage(registry, -1, 100),
                       RegistryException,
                       "The game object `-1` is not registered with the registry.")
}
