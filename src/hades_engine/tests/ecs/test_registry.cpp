// Local headers
#include "ecs/registry.hpp"
#include "ecs/stats.hpp"
#include "ecs/systems/attacks.hpp"
#include "events.hpp"
#include "macros.hpp"

/// Represents a game object component useful for testing.
struct TestGameObjectComponentOne final : ComponentBase {};

/// Represents a game object component with data useful for testing.
struct TestGameObjectComponentTwo final : ComponentBase {
  /// A test list of integers.
  std::vector<int> test_list;

  /// Initialise the object.
  ///
  /// @param test_list - The list to be used for testing.
  explicit TestGameObjectComponentTwo(const std::vector<int>& test_list) : test_list(test_list) {}
};

/// Represents a test system useful for testing.
struct TestSystem final : SystemBase {
  /// Whether the system has been called or not.
  mutable bool called{false};

  /// Initialise the system.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  explicit TestSystem(Registry* registry) : SystemBase(registry) {}

  /// Update the system.
  void update(double /*delta_time*/) const override { called = true; }
};

/// Implements the fixture for the ecs/registry.hpp tests.
class RegistryFixture : public testing::Test {
 protected:
  /// The registry that manages the game objects, components, and systems.
  Registry registry;

  /// Tear down the fixture after the tests.
  void TearDown() override { clear_listeners(); }
};

/// Test that a game object with no components works correctly.
TEST_F(RegistryFixture, TestRegistryGameObjectNoComponents) {
  ASSERT_EQ(registry.create_game_object(GameObjectType::Player), 0);
  ASSERT_TRUE(registry.has_game_object(0));
  ASSERT_FALSE(registry.has_component<TestGameObjectComponentOne>(0));
  ASSERT_FALSE(registry.has_component<TestGameObjectComponentTwo>(0));
  ASSERT_EQ(registry.get_game_object_type(0), GameObjectType::Player);
  ASSERT_EQ(registry.get_game_object_ids(GameObjectType::Player).size(), 1);
  ASSERT_EQ(std::ranges::distance(registry.get_game_object_components<TestGameObjectComponentOne>()), 0);
  ASSERT_EQ(std::ranges::distance(registry.get_game_object_components<TestGameObjectComponentTwo>()), 0);
  ASSERT_EQ(std::ranges::distance(
                registry.get_game_object_components<TestGameObjectComponentOne, TestGameObjectComponentTwo>()),
            0);
}

/// Test that a game object with a single component works correctly.
TEST_F(RegistryFixture, TestRegistryGameObjectSingleComponent) {
  registry.create_game_object(GameObjectType::Player);
  registry.add_component<TestGameObjectComponentOne>(0);
  ASSERT_TRUE(registry.has_game_object(0));
  ASSERT_TRUE(registry.has_component<TestGameObjectComponentOne>(0));
  ASSERT_FALSE(registry.has_component<TestGameObjectComponentTwo>(0));
  ASSERT_NE(registry.get_component<TestGameObjectComponentOne>(0), nullptr);
  ASSERT_EQ(registry.get_game_object_type(0), GameObjectType::Player);
  ASSERT_EQ(registry.get_game_object_ids(GameObjectType::Player).size(), 1);
  ASSERT_EQ(std::ranges::distance(registry.get_game_object_components<TestGameObjectComponentOne>()), 1);
  ASSERT_EQ(std::ranges::distance(registry.get_game_object_components<TestGameObjectComponentTwo>()), 0);
  ASSERT_EQ(std::ranges::distance(
                registry.get_game_object_components<TestGameObjectComponentOne, TestGameObjectComponentTwo>()),
            0);
}

/// Test that a game object with multiple components works correctly.
TEST_F(RegistryFixture, TestRegistryGameObjectMultipleComponents) {
  registry.create_game_object(GameObjectType::Player);
  registry.add_component<TestGameObjectComponentOne>(0);
  registry.add_component<TestGameObjectComponentTwo>(0, std::vector{10});
  ASSERT_TRUE(registry.has_game_object(0));
  ASSERT_TRUE(registry.has_component<TestGameObjectComponentOne>(0));
  ASSERT_TRUE(registry.has_component<TestGameObjectComponentTwo>(0));
  ASSERT_NE(registry.get_component<TestGameObjectComponentOne>(0), nullptr);
  ASSERT_NE(registry.get_component<TestGameObjectComponentTwo>(0), nullptr);
  ASSERT_EQ(registry.get_game_object_type(0), GameObjectType::Player);
  ASSERT_EQ(registry.get_game_object_ids(GameObjectType::Player).size(), 1);
  ASSERT_EQ(std::ranges::distance(registry.get_game_object_components<TestGameObjectComponentOne>()), 1);
  ASSERT_EQ(std::ranges::distance(registry.get_game_object_components<TestGameObjectComponentTwo>()), 1);
  ASSERT_EQ(std::ranges::distance(
                registry.get_game_object_components<TestGameObjectComponentOne, TestGameObjectComponentTwo>()),
            1);
}

/// Test that a game object with two identical components only adds the first one.
TEST_F(RegistryFixture, TestRegistryGameObjectDuplicateComponent) {
  registry.create_game_object(GameObjectType::Player);
  registry.add_component<TestGameObjectComponentTwo>(0, std::vector({10}));
  registry.add_component<TestGameObjectComponentTwo>(0, std::vector({20}));
  ASSERT_EQ(registry.get_component<TestGameObjectComponentTwo>(0)->test_list[0], 20);
}

/// Test that a game object with a kinematic component works correctly.
TEST_F(RegistryFixture, TestRegistryGameObjectKinematicComponent) {
  registry.create_game_object(GameObjectType::Player);
  registry.add_component<KinematicComponent>(0, cpvzero);
  ASSERT_TRUE(cpSpaceContainsBody(registry.get_space(), *registry.get_component<KinematicComponent>(0)->body));
  ASSERT_TRUE(cpSpaceContainsShape(registry.get_space(), *registry.get_component<KinematicComponent>(0)->shape));
  ASSERT_EQ(cpShapeGetBody(*registry.get_component<KinematicComponent>(0)->shape),
            *registry.get_component<KinematicComponent>(0)->body);
  ASSERT_EQ(cpBodyGetPosition(*registry.get_component<KinematicComponent>(0)->body), cpvzero);
  ASSERT_EQ(cpShapeGetCollisionType(*registry.get_component<KinematicComponent>(0)->shape),
            static_cast<cpCollisionType>(GameObjectType::Player));
  const auto [group, categories, mask]{cpShapeGetFilter(*registry.get_component<KinematicComponent>(0)->shape)};
  ASSERT_EQ(group, CP_NO_GROUP);
  ASSERT_EQ(categories, static_cast<cpBitmask>(GameObjectType::Player));
  ASSERT_EQ(mask, CP_ALL_CATEGORIES);
}

/// Test that multiple game objects work correctly.
TEST_F(RegistryFixture, TestRegistryMultipleGameObjects) {
  registry.create_game_object(GameObjectType::Player);
  registry.add_component<TestGameObjectComponentOne>(0);
  registry.create_game_object(GameObjectType::Enemy);
  registry.add_component<TestGameObjectComponentTwo>(1, std::vector({10}));
  ASSERT_TRUE(registry.has_game_object(0));
  ASSERT_TRUE(registry.has_game_object(1));
  ASSERT_TRUE(registry.has_component<TestGameObjectComponentOne>(0));
  ASSERT_TRUE(registry.has_component<TestGameObjectComponentTwo>(1));
  ASSERT_EQ(registry.get_game_object_type(0), GameObjectType::Player);
  ASSERT_EQ(registry.get_game_object_type(1), GameObjectType::Enemy);
  ASSERT_EQ(std::ranges::distance(registry.get_game_object_components<TestGameObjectComponentOne>()), 1);
  ASSERT_EQ(std::ranges::distance(registry.get_game_object_components<TestGameObjectComponentTwo>()), 1);
  ASSERT_EQ(std::ranges::distance(
                registry.get_game_object_components<TestGameObjectComponentOne, TestGameObjectComponentTwo>()),
            0);
}

/// Test that reusing a game object ID works correctly.
TEST_F(RegistryFixture, TestRegistryReuseGameObjectID) {
  registry.create_game_object(GameObjectType::Player);
  registry.create_game_object(GameObjectType::Player);
  registry.create_game_object(GameObjectType::Player);
  ASSERT_TRUE(registry.has_game_object(0));
  ASSERT_TRUE(registry.has_game_object(1));
  ASSERT_TRUE(registry.has_game_object(2));
  registry.delete_game_object(1);
  ASSERT_FALSE(registry.has_game_object(1));
  const auto new_id{registry.create_game_object(GameObjectType::Player)};
  ASSERT_EQ(new_id, 1);
  ASSERT_TRUE(registry.has_game_object(1));
}

/// Test that deleting a game object works correctly.
TEST_F(RegistryFixture, TestRegistryDeleteGameObject) {
  registry.create_game_object(GameObjectType::Player);
  registry.add_component<TestGameObjectComponentOne>(0);
  registry.add_component<TestGameObjectComponentTwo>(0, std::vector({10}));
  registry.delete_game_object(0);
  ASSERT_FALSE(registry.has_game_object(0));
  ASSERT_FALSE(registry.has_component<TestGameObjectComponentOne>(0));
  ASSERT_FALSE(registry.has_component<TestGameObjectComponentTwo>(0));
}

/// Test that deleting a game object does not affect other game objects.
TEST_F(RegistryFixture, TestRegistryDeleteGameObjectNoEffect) {
  registry.create_game_object(GameObjectType::Player);
  registry.add_component<TestGameObjectComponentOne>(0);
  registry.add_component<TestGameObjectComponentTwo>(0, std::vector({10}));
  registry.create_game_object(GameObjectType::Player);
  registry.add_component<TestGameObjectComponentOne>(1);
  registry.add_component<TestGameObjectComponentTwo>(1, std::vector({20}));
  registry.delete_game_object(0);
  ASSERT_TRUE(registry.has_game_object(1));
  ASSERT_TRUE(registry.has_component<TestGameObjectComponentOne>(1));
  ASSERT_TRUE(registry.has_component<TestGameObjectComponentTwo>(1));
}

/// Test that deleting a game object with a kinematic component works correctly.
TEST_F(RegistryFixture, TestRegistryDeleteGameObjectKinematicComponent) {
  registry.create_game_object(GameObjectType::Player);
  registry.add_component<KinematicComponent>(0, cpvzero);
  auto* body{*registry.get_component<KinematicComponent>(0)->body};
  auto* shape{*registry.get_component<KinematicComponent>(0)->shape};
  registry.delete_game_object(0);
  ASSERT_FALSE(registry.has_game_object(0));
  ASSERT_EQ(std::ranges::distance(registry.get_game_object_components<KinematicComponent>()), 0);
  ASSERT_FALSE(cpSpaceContainsBody(registry.get_space(), body));
  ASSERT_FALSE(cpSpaceContainsShape(registry.get_space(), shape));
}

/// Test that clearing the registry without any registered game objects works correctly.
TEST_F(RegistryFixture, TestRegistryClearNoGameObjects) {
  registry.clear_game_objects();
  ASSERT_EQ(registry.get_game_object_ids(GameObjectType::Player).size(), 0);
}

/// Test that clearing the registry with non-preserved game objects works correctly.
TEST_F(RegistryFixture, TestRegistryClearWithGameObjects) {
  registry.create_game_object(GameObjectType::Player);
  registry.create_game_object(GameObjectType::Player);
  registry.clear_game_objects();
  ASSERT_EQ(registry.get_game_object_ids(GameObjectType::Player).size(), 0);
}

/// Test that clearing the registry with preserved game objects works correctly.
TEST_F(RegistryFixture, TestRegistryClearWithPreservedGameObjects) {
  registry.create_game_object(GameObjectType::Player);
  registry.create_game_object(GameObjectType::Player);
  registry.clear_game_objects({0});
  ASSERT_EQ(registry.get_game_object_ids(GameObjectType::Player).size(), 1);
  ASSERT_TRUE(registry.has_game_object(0));
  ASSERT_FALSE(registry.has_game_object(1));
}

/// Test that the game object creation callback is called correctly.
TEST_F(RegistryFixture, TestRegistryGameObjectCreationCallback) {
  auto called{-1};
  add_callback<EventType::GameObjectCreation>(
      [&called](const GameObjectID event, const GameObjectType) { called = event; });
  ASSERT_EQ(registry.create_game_object(GameObjectType::Player), 0);
  ASSERT_EQ(called, 0);
}

/// Test that the game object death callback is called correctly.
TEST_F(RegistryFixture, TestRegistryGameObjectDeathCallback) {
  auto called{-1};
  add_callback<EventType::GameObjectDeath>([&called](const GameObjectID event) { called = event; });
  ASSERT_EQ(registry.create_game_object(GameObjectType::Player), 0);
  ASSERT_EQ(called, -1);
  registry.delete_game_object(0);
  ASSERT_EQ(called, 0);
}

/// Test that deleting an unregistered game object raises an error.
TEST_F(RegistryFixture, TestRegistryGameObjectDeleteUnregistered){
    ASSERT_THROW_MESSAGE(registry.delete_game_object(0), RegistryError,
                         "The game object ID `0` is not registered with the registry.")}

/// Test that getting the type of an unregistered game object raises an error.
TEST_F(RegistryFixture, TestRegistryGameObjectGetTypeUnregistered){
    ASSERT_THROW_MESSAGE(registry.get_game_object_type(0), RegistryError,
                         "The game object ID `0` is not registered with the registry.")}

/// Test that a system is added to the registry correctly.
TEST_F(RegistryFixture, TestRegistryAddGetSystemRegistered) {
  registry.add_system<TestSystem>();
  ASSERT_NE(registry.get_system<TestSystem>(), nullptr);
}

/// Test that an exception is thrown when adding the same system twice.
TEST_F(RegistryFixture, TestRegistryAddSystemAlreadyRegistered) {
  registry.add_system<TestSystem>();
  ASSERT_NO_THROW(registry.add_system<TestSystem>());
  ASSERT_NE(registry.get_system<TestSystem>(), nullptr);
}

/// Test that an exception is thrown when a system is not registered.
TEST_F(RegistryFixture, TestRegistryGetSystemNotRegistered){
    ASSERT_THROW_MESSAGE(registry.get_system<TestSystem>(), RegistryError,
                         "The `TestSystem` is not registered with the registry.")}

/// Test that a system is updated correctly.
TEST_F(RegistryFixture, TestRegistrySystemUpdate) {
  registry.create_game_object(GameObjectType::Player);
  registry.add_component<TestGameObjectComponentTwo>(0, std::vector({10}));
  registry.add_system<TestSystem>();
  registry.update(0);
  ASSERT_TRUE(registry.get_system<TestSystem>()->called);
}

/// Test a collision between a player and an enemy bullet.
TEST_F(RegistryFixture, TestRegistryPlayerEnemyBulletCollision) {
  // Add the required systems and the player and bullet to the registry
  registry.add_system<PhysicsSystem>();
  registry.add_system<DamageSystem>();
  const auto player_id{registry.create_game_object(GameObjectType::Player)};
  registry.add_component<Armour>(0, 0);
  registry.add_component<Health>(0, 100);
  registry.add_component<KinematicComponent>(0, cpvzero);
  registry.get_system<PhysicsSystem>()->add_bullet({{.x = -32, .y = 0}, {.x = 16, .y = 0}}, 50, GameObjectType::Enemy);

  // Test that the collision is handled correctly
  registry.update(1);
  ASSERT_EQ(registry.get_component<Health>(player_id)->get_value(), 50);
  ASSERT_FALSE(registry.has_game_object(1));
}

/// Test a collision between an enemy and a player bullet is handled correctly.
TEST_F(RegistryFixture, TestRegistryEnemyPlayerBulletCollision) {
  // Add the required systems and the enemy and bullet to the registry
  registry.add_system<PhysicsSystem>();
  registry.add_system<DamageSystem>();
  const auto enemy_id{registry.create_game_object(GameObjectType::Enemy)};
  registry.add_component<Armour>(0, 0);
  registry.add_component<Health>(0, 100);
  registry.add_component<KinematicComponent>(0, cpvzero);
  registry.get_system<PhysicsSystem>()->add_bullet({{.x = -32, .y = 0}, {.x = 16, .y = 0}}, 50, GameObjectType::Player);

  // Test that the collision is handled correctly
  registry.update(1);
  ASSERT_EQ(registry.get_component<Health>(enemy_id)->get_value(), 50);
  ASSERT_FALSE(registry.has_game_object(1));
}

/// Test a collision between a player and a player bullet is handled correctly.
TEST_F(RegistryFixture, TestRegistryPlayerPlayerBulletCollision) {
  // Add the required systems and the player and bullet to the registry
  registry.add_system<PhysicsSystem>();
  registry.add_system<DamageSystem>();
  const auto player_id{registry.create_game_object(GameObjectType::Player)};
  registry.add_component<Armour>(0, 0);
  registry.add_component<Health>(0, 100);
  registry.add_component<KinematicComponent>(0, cpvzero);
  registry.get_system<PhysicsSystem>()->add_bullet({{.x = -32, .y = 0}, {.x = 16, .y = 0}}, 50, GameObjectType::Player);

  // Test that the collision is handled correctly (the player should not be damaged)
  for (int i{0}; i < 5; i++) {
    registry.update(1);
    ASSERT_EQ(registry.get_component<Health>(player_id)->get_value(), 100);
    ASSERT_EQ(cpBodyGetPosition(*registry.get_component<KinematicComponent>(1)->body), cpv(16 * (i - 1), 0));
  }
}

/// Test a collision between an enemy and an enemy bullet is handled correctly.
TEST_F(RegistryFixture, TestRegistryEnemyEnemyBulletCollision) {
  // Add the required systems and the enemy and bullet to the registry
  registry.add_system<PhysicsSystem>();
  registry.add_system<DamageSystem>();
  const auto enemy_id{registry.create_game_object(GameObjectType::Enemy)};
  registry.add_component<Armour>(0, 0);
  registry.add_component<Health>(0, 100);
  registry.add_component<KinematicComponent>(0, cpvzero);
  registry.get_system<PhysicsSystem>()->add_bullet({{.x = -32, .y = 0}, {.x = 16, .y = 0}}, 50, GameObjectType::Enemy);

  // Test that the collision is handled correctly (the enemy should not be damaged)
  for (int i{0}; i < 5; i++) {
    registry.update(1);
    ASSERT_EQ(registry.get_component<Health>(enemy_id)->get_value(), 100);
    ASSERT_EQ(cpBodyGetPosition(*registry.get_component<KinematicComponent>(1)->body), cpv(16 * (i - 1), 0));
  }
}

/// Test a collision between a wall and a bullet is handled correctly.
TEST_F(RegistryFixture, TestRegistryWallBulletCollision) {
  // Add the required systems and the enemy and bullet to the registry
  registry.add_system<PhysicsSystem>();
  registry.add_system<DamageSystem>();
  registry.create_game_object(GameObjectType::Wall);
  registry.add_component<KinematicComponent>(0, cpvzero, true);
  registry.get_system<PhysicsSystem>()->add_bullet({{.x = -32, .y = 0}, {.x = 16, .y = 0}}, 50, GameObjectType::Player);

  // Test that the collision is handled correctly (the bullet should be deleted)
  registry.update(2);
  ASSERT_FALSE(registry.has_game_object(1));
}
