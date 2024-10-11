// Local headers
#include "ecs/registry.hpp"
#include "ecs/stats.hpp"
#include "ecs/systems/attacks.hpp"
#include "ecs/systems/physics.hpp"
#include "macros.hpp"

namespace {
/// Throw a RegistryError with given values.
///
/// @tparam Ts - The types of the values.
/// @param vals - The values to be used for testing.
/// @throws RegistryError - Always for testing.
template <typename... Ts>
void throw_registry_error(const Ts... vals) {
  throw RegistryError(vals...);
}
}  // namespace

/// Represents a game object component useful for testing.
struct TestGameObjectComponentOne final : ComponentBase {};

/// Represents a game object component with data useful for testing.
struct TestGameObjectComponentTwo final : ComponentBase {
  /// A test list of integers.
  std::vector<int> test_list;

  /// Initialise the object.
  ///
  /// @param test_list - The list to be used for testing.
  explicit TestGameObjectComponentTwo(const std::vector<int> &test_list) : test_list(test_list) {}
};

/// Represents a test system useful for testing.
struct TestSystem final : SystemBase {
  /// Whether the system has been called or not.
  mutable bool called{false};

  /// Initialise the system.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  explicit TestSystem(Registry *registry) : SystemBase(registry) {}

  /// Update the system.
  void update(double /*delta_time*/) const override { called = true; }
};

/// Implements the fixture for the ecs/registry.hpp tests.
class RegistryFixture : public testing::Test {
 protected:
  /// The registry that manages the game objects, components, and systems.
  Registry registry;
};

/// Test that RegistryError is thrown correctly when given a message.
TEST(Tests, TestThrowRegistryErrorNonEmptyMessage){
    ASSERT_THROW_MESSAGE(throw_registry_error("test", std::type_index(typeid(TestSystem))), RegistryError,
                         "The test `TestSystem` is not registered with the registry.")}

/// Test that RegistryError is thrown correctly when given an extra message.
TEST(Tests, TestThrowRegistryErrorNonEmptyValues){
    ASSERT_THROW_MESSAGE(throw_registry_error("test", std::type_index(typeid(TestSystem)), "test"), RegistryError,
                         "The test `TestSystem` test.")}

/// Test that RegistryError is thrown correctly when given multiple empty values.
TEST(Tests, TestThrowRegistryErrorEmptyValues){
    ASSERT_THROW_MESSAGE(throw_registry_error("", std::type_index(typeid(TestSystem)), ""), RegistryError,
                         "The  `TestSystem` .")}

/// Test that RegistryError is thrown correctly when given a game object ID and a type_index.
TEST(Tests, TestThrowRegistryErrorGameObjectIDTypeIndex){
    ASSERT_THROW_MESSAGE(throw_registry_error(0, std::type_index(typeid(TestSystem))), RegistryError,
                         "The component `TestSystem` for the game object ID `0` is not registered with the registry.")}

/// Test that a valid position is converted correctly.
TEST(Tests, TestGridPosToPixelPositivePosition) {
  ASSERT_EQ(grid_pos_to_pixel({.x = 100, .y = 100}), cpv(6432, 6432));
}

/// Test that a zero position is converted correctly.
TEST(Tests, TestGridPosToPixelZeroPosition) { ASSERT_EQ(grid_pos_to_pixel(cpvzero), cpv(32, 32)); }

/// Test that a negative x position raises an error.
TEST(Tests, TestGridPosToPixelNegativeXPosition){
    ASSERT_THROW_MESSAGE(grid_pos_to_pixel({-100, 100}), std::invalid_argument, "The position cannot be negative.")}

/// Test that a negative y position raises an error.
TEST(Tests, TestGridPosToPixelNegativeYPosition){
    ASSERT_THROW_MESSAGE(grid_pos_to_pixel({100, -100}), std::invalid_argument, "The position cannot be negative.")}

/// Test that a negative x and y position raises an error.
TEST(Tests, TestGridPosToPixelNegativeXYPosition){
    ASSERT_THROW_MESSAGE(grid_pos_to_pixel({-100, -100}), std::invalid_argument, "The position cannot be negative.")}

/// Test that a game object with no components is added to the registry correctly.
TEST_F(RegistryFixture, TestRegistryEmptyGameObject) {
  // Create the callback for the game object death event
  int called{-1};
  registry.add_callback(EventType::GameObjectDeath, [&called](const auto event) { called = event; });

  // Test that creating the game object works correctly
  ASSERT_EQ(registry.create_game_object(GameObjectType::Player, cpvzero, {}), 0);
  ASSERT_TRUE(registry.has_game_object(0));
  ASSERT_FALSE(registry.has_component(0, typeid(TestGameObjectComponentOne)));
  ASSERT_FALSE(registry.has_component(0, typeid(TestGameObjectComponentTwo)));
  ASSERT_EQ(registry.get_game_object_type(0), GameObjectType::Player);
  ASSERT_EQ(std::ranges::distance(registry.find_components<TestGameObjectComponentOne>()), 0);
  ASSERT_EQ(std::ranges::distance(registry.find_components<TestGameObjectComponentTwo>()), 0);

  // Test that deleting the game object works correctly
  registry.delete_game_object(0);
  ASSERT_EQ(called, 0);
  ASSERT_THROW_MESSAGE(registry.delete_game_object(0), RegistryError,
                       "The game object `0` is not registered with the registry.")
}

/// Test that multiple components are added to the registry correctly.
TEST_F(RegistryFixture, TestRegistryGameObjectComponents) {
  // Test that creating the game object works correctly
  registry.create_game_object(GameObjectType::Player, cpvzero,
                              {std::make_shared<TestGameObjectComponentOne>(),
                               std::make_shared<TestGameObjectComponentTwo>(std::vector({10}))});
  ASSERT_TRUE(registry.has_game_object(0));
  ASSERT_NE(registry.get_component<TestGameObjectComponentOne>(0), nullptr);
  ASSERT_NE(registry.get_component(0, typeid(TestGameObjectComponentTwo)), nullptr);
  ASSERT_EQ(registry.get_game_object_type(0), GameObjectType::Player);
  ASSERT_EQ(std::ranges::distance(registry.find_components<TestGameObjectComponentOne>()), 1);
  ASSERT_EQ(std::ranges::distance(registry.find_components<TestGameObjectComponentTwo>()), 1);
  ASSERT_EQ(std::ranges::distance(registry.find_components<TestGameObjectComponentOne, TestGameObjectComponentTwo>()),
            1);

  // Test that deleting the game object works correctly
  registry.delete_game_object(0);
  ASSERT_FALSE(registry.has_game_object(0));
  ASSERT_THROW_MESSAGE(
      registry.get_component<TestGameObjectComponentOne>(0), RegistryError,
      "The component `TestGameObjectComponentOne` for the game object ID `0` is not registered with the registry.")
  ASSERT_THROW_MESSAGE(
      (registry.get_component(0, typeid(TestGameObjectComponentTwo))), RegistryError,
      "The component `TestGameObjectComponentTwo` for the game object ID `0` is not registered with the registry.")
  ASSERT_THROW_MESSAGE(registry.get_game_object_type(0), RegistryError,
                       "The game object `0` is not registered with the registry.")
  ASSERT_EQ(std::ranges::distance(registry.find_components<TestGameObjectComponentOne>()), 0);
  ASSERT_EQ(std::ranges::distance(registry.find_components<TestGameObjectComponentTwo>()), 0);
  ASSERT_EQ(std::ranges::distance(registry.find_components<TestGameObjectComponentOne, TestGameObjectComponentTwo>()),
            0);
}

/// Test that multiple game objects are added to the registry correctly.
TEST_F(RegistryFixture, TestRegistryMultipleGameObjects) {
  // Test that creating two game objects works correctly
  ASSERT_EQ(
      registry.create_game_object(GameObjectType::Player, cpvzero, {std::make_shared<TestGameObjectComponentOne>()}),
      0);
  ASSERT_EQ(registry.create_game_object(GameObjectType::Player, cpvzero,
                                        {std::make_shared<TestGameObjectComponentOne>(),
                                         std::make_shared<TestGameObjectComponentTwo>(std::vector({10}))}),
            1);
  ASSERT_TRUE(registry.has_game_object(0));
  ASSERT_EQ(std::ranges::distance(registry.find_components<TestGameObjectComponentOne>()), 2);
  ASSERT_EQ(std::ranges::distance(registry.find_components<TestGameObjectComponentTwo>()), 1);
  ASSERT_EQ(std::ranges::distance(registry.find_components<TestGameObjectComponentOne, TestGameObjectComponentTwo>()),
            1);

  // Test that deleting the first game object works correctly
  registry.delete_game_object(0);
  ASSERT_FALSE(registry.has_game_object(0));
  ASSERT_EQ(std::ranges::distance(registry.find_components<TestGameObjectComponentOne>()), 1);
  ASSERT_EQ(std::ranges::distance(registry.find_components<TestGameObjectComponentTwo>()), 1);
  ASSERT_EQ(std::ranges::distance(registry.find_components<TestGameObjectComponentOne, TestGameObjectComponentTwo>()),
            1);
}

/// Test that a game object with a kinematic component is added to the registry correctly.
TEST_F(RegistryFixture, TestRegistryGameObjectKinematicComponent) {
  // Test that creating the game object works correctly
  registry.create_game_object(GameObjectType::Player, cpvzero,
                              {std::make_shared<KinematicComponent>(std::vector<cpVect>{})});
  ASSERT_TRUE(registry.has_game_object(0));
  ASSERT_NE(registry.get_component<KinematicComponent>(0), nullptr);
  ASSERT_NE(registry.get_component(0, typeid(KinematicComponent)), nullptr);
  ASSERT_EQ(std::ranges::distance(registry.find_components<KinematicComponent>()), 1);

  // Test that the body and shape are added to each other and the space correctly
  ASSERT_TRUE(cpSpaceContainsBody(registry.get_space(), *registry.get_component<KinematicComponent>(0)->body));
  ASSERT_TRUE(cpSpaceContainsShape(registry.get_space(), *registry.get_component<KinematicComponent>(0)->shape));
  ASSERT_EQ(cpShapeGetBody(*registry.get_component<KinematicComponent>(0)->shape),
            *registry.get_component<KinematicComponent>(0)->body);
  ASSERT_EQ(cpBodyGetPosition(*registry.get_component<KinematicComponent>(0)->body), cpv(32, 32));
  ASSERT_EQ(cpShapeGetCollisionType(*registry.get_component<KinematicComponent>(0)->shape),
            static_cast<cpCollisionType>(GameObjectType::Player));
  const auto [group, categories, mask]{cpShapeGetFilter(*registry.get_component<KinematicComponent>(0)->shape)};
  ASSERT_EQ(group, CP_NO_GROUP);
  ASSERT_EQ(categories, static_cast<cpBitmask>(GameObjectType::Player));
  ASSERT_EQ(mask, CP_ALL_CATEGORIES);

  // Test that deleting the game object works correctly
  auto *body{*registry.get_component<KinematicComponent>(0)->body};
  auto *shape{*registry.get_component<KinematicComponent>(0)->shape};
  registry.delete_game_object(0);
  ASSERT_FALSE(registry.has_game_object(0));
  ASSERT_EQ(std::ranges::distance(registry.find_components<KinematicComponent>()), 0);
  ASSERT_FALSE(cpSpaceContainsBody(registry.get_space(), body));
  ASSERT_FALSE(cpSpaceContainsShape(registry.get_space(), shape));
}

/// Test that a game object with duplicate components is added to the registry correctly.
TEST_F(RegistryFixture, TestRegistryGameObjectDuplicateComponents) {
  // Test that creating a game object with two of the same components only adds the first one
  registry.create_game_object(GameObjectType::Player, cpvzero,
                              {std::make_shared<TestGameObjectComponentTwo>(std::vector({10})),
                               std::make_shared<TestGameObjectComponentTwo>(std::vector({20}))});
  ASSERT_EQ(registry.get_component<TestGameObjectComponentTwo>(0)->test_list[0], 10);
}

/// Test that passing the same component to multiple game objects works correctly.
TEST_F(RegistryFixture, TestRegistryGameObjectSameComponent) {
  const auto component_one{std::make_shared<TestGameObjectComponentTwo>(std::vector({10}))};
  registry.create_game_object(GameObjectType::Player, cpvzero, {component_one});
  registry.create_game_object(GameObjectType::Player, cpvzero, {component_one});
  registry.get_component<TestGameObjectComponentTwo>(0)->test_list[0] = 20;
  ASSERT_EQ(registry.get_component<TestGameObjectComponentTwo>(0)->test_list[0], 20);
  ASSERT_EQ(registry.get_component<TestGameObjectComponentTwo>(1)->test_list[0], 20);
}

/// Test that an exception is thrown if a system is not registered.
TEST_F(RegistryFixture,
       TestRegistryZeroSystems){ASSERT_THROW_MESSAGE(registry.get_system<TestSystem>(), RegistryError,
                                                     "The system `TestSystem` is not registered with the registry.")}

/// Test that a system is updated correctly.
TEST_F(RegistryFixture, TestRegistrySystemUpdate) {
  // Test that the system is added correctly
  registry.create_game_object(GameObjectType::Player, cpvzero,
                              {std::make_shared<TestGameObjectComponentTwo>(std::vector({10}))});
  registry.add_system<TestSystem>();
  ASSERT_THROW_MESSAGE(registry.add_system<TestSystem>(), RegistryError,
                       "The system `TestSystem` is already registered with the registry.")
  ASSERT_NE(registry.get_system<TestSystem>(), nullptr);

  // Test that the system is updated correctly
  registry.update(0);
  ASSERT_TRUE(registry.get_system<TestSystem>()->called);
}

/// Test that an exception is thrown if you add the same system twice.
TEST_F(RegistryFixture, TestRegistryDuplicateSystem) {
  registry.add_system<TestSystem>();
  ASSERT_THROW_MESSAGE(registry.add_system<TestSystem>(), RegistryError,
                       "The system `TestSystem` is already registered with the registry.")
}

/// Test that an exception is thrown if you get a system that is not registered.
TEST_F(RegistryFixture, TestRegistryGetSystemNotRegistered){
    ASSERT_THROW_MESSAGE(registry.get_system<TestSystem>(), RegistryError,
                         "The system `TestSystem` is not registered with the registry.")}

/// Test a collision between a player and an enemy bullet.
TEST_F(RegistryFixture, TestRegistryPlayerEnemyBulletCollision) {
  // Add the required systems and the player and bullet to the registry
  registry.add_system<PhysicsSystem>();
  registry.add_system<DamageSystem>();
  const auto player_id{registry.create_game_object(GameObjectType::Player, cpvzero,
                                                   {std::make_shared<Armour>(0, 0), std::make_shared<Health>(200, 0),
                                                    std::make_shared<KinematicComponent>(std::vector<cpVect>{})})};
  registry.get_system<PhysicsSystem>()->add_bullet({{.x = -32, .y = 0}, {.x = 16, .y = 0}}, 20, GameObjectType::Enemy);

  // Test that the collision is handled correctly
  ASSERT_EQ(registry.get_component<Health>(player_id)->get_value(), 200);
  ASSERT_EQ(cpBodyGetPosition(*registry.get_component<KinematicComponent>(1)->body), cpv(-32, 0));
  registry.get_system<PhysicsSystem>()->update(1);
  ASSERT_EQ(registry.get_component<Health>(player_id)->get_value(), 200);
  ASSERT_EQ(cpBodyGetPosition(*registry.get_component<KinematicComponent>(1)->body), cpv(-16, 0));
  registry.get_system<PhysicsSystem>()->update(1);
  ASSERT_EQ(registry.get_component<Health>(player_id)->get_value(), 180);
  ASSERT_THROW_MESSAGE(
      registry.get_component<KinematicComponent>(1), RegistryError,
      "The component `KinematicComponent` for the game object ID `1` is not registered with the registry.")
}

/// Test a collision between a player and a player bullet.
TEST_F(RegistryFixture, TestRegistryPlayerPlayerBulletCollision) {
  // Add the required systems and the player and bullet to the registry
  registry.add_system<PhysicsSystem>();
  registry.add_system<DamageSystem>();
  const auto player_id{registry.create_game_object(GameObjectType::Player, cpvzero,
                                                   {std::make_shared<Armour>(0, 0), std::make_shared<Health>(200, 0),
                                                    std::make_shared<KinematicComponent>(std::vector<cpVect>{})})};
  registry.get_system<PhysicsSystem>()->add_bullet({{.x = -32, .y = 0}, {.x = 16, .y = 0}}, 20, GameObjectType::Player);

  // Test that the collision is handled correctly
  for (int i{0}; i < 10; i++) {
    ASSERT_EQ(registry.get_component<Health>(player_id)->get_value(), 200);
    ASSERT_EQ(cpBodyGetPosition(*registry.get_component<KinematicComponent>(1)->body), cpv(-32 + (i * 16), 0));
    registry.get_system<PhysicsSystem>()->update(1);
  }
}

/// Test a collision between an enemy and a player bullet.
TEST_F(RegistryFixture, TestRegistryEnemyPlayerBulletCollision) {
  // Add the required systems and the enemy and bullet to the registry
  registry.add_system<PhysicsSystem>();
  registry.add_system<DamageSystem>();
  const auto enemy_id{registry.create_game_object(GameObjectType::Enemy, cpvzero,
                                                  {std::make_shared<Armour>(0, 0), std::make_shared<Health>(100, 0),
                                                   std::make_shared<KinematicComponent>(std::vector<cpVect>{})})};
  registry.get_system<PhysicsSystem>()->add_bullet({{.x = -32, .y = 0}, {.x = 16, .y = 0}}, 50, GameObjectType::Player);

  // Test that the collision is handled correctly
  ASSERT_EQ(registry.get_component<Health>(enemy_id)->get_value(), 100);
  ASSERT_EQ(cpBodyGetPosition(*registry.get_component<KinematicComponent>(1)->body), cpv(-32, 0));
  registry.get_system<PhysicsSystem>()->update(1);
  ASSERT_EQ(registry.get_component<Health>(enemy_id)->get_value(), 100);
  ASSERT_EQ(cpBodyGetPosition(*registry.get_component<KinematicComponent>(1)->body), cpv(-16, 0));
  registry.get_system<PhysicsSystem>()->update(1);
  ASSERT_EQ(registry.get_component<Health>(enemy_id)->get_value(), 50);
  ASSERT_THROW_MESSAGE(
      registry.get_component<KinematicComponent>(1), RegistryError,
      "The component `KinematicComponent` for the game object ID `1` is not registered with the registry.")
}

/// Test a collision between an enemy and an enemy bullet.
TEST_F(RegistryFixture, TestRegistryEnemyEnemyBulletCollision) {
  // Add the required systems and the enemy and bullet to the registry
  registry.add_system<PhysicsSystem>();
  registry.add_system<DamageSystem>();
  const auto enemy_id{registry.create_game_object(GameObjectType::Enemy, cpvzero,
                                                  {std::make_shared<Armour>(0, 0), std::make_shared<Health>(100, 0),
                                                   std::make_shared<KinematicComponent>(std::vector<cpVect>{})})};
  registry.get_system<PhysicsSystem>()->add_bullet({{.x = -32, .y = 0}, {.x = 16, .y = 0}}, 50, GameObjectType::Enemy);

  // Test that the collision is handled correctly
  for (int i = 0; i < 10; i++) {
    ASSERT_EQ(registry.get_component<Health>(enemy_id)->get_value(), 100);
    ASSERT_EQ(cpBodyGetPosition(*registry.get_component<KinematicComponent>(1)->body), cpv(-32 + (i * 16), 0));
    registry.get_system<PhysicsSystem>()->update(1);
  }
}

/// Test a collision between a wall and a bullet.
TEST_F(RegistryFixture, TestRegistryWallBulletCollision) {
  // Add the required systems and the enemy and bullet to the registry
  registry.add_system<PhysicsSystem>();
  registry.add_system<DamageSystem>();
  const auto wall_id{
      registry.create_game_object(GameObjectType::Wall, cpvzero, {std::make_shared<KinematicComponent>(true)})};
  registry.get_system<PhysicsSystem>()->add_bullet({{.x = -32, .y = 0}, {.x = 16, .y = 0}}, 30, GameObjectType::Player);

  // Test that the collision is handled correctly
  ASSERT_EQ(cpBodyGetPosition(*registry.get_component<KinematicComponent>(wall_id)->body), cpv(32, 32));
  ASSERT_EQ(cpBodyGetPosition(*registry.get_component<KinematicComponent>(1)->body), cpv(-32, 0));
  registry.get_system<PhysicsSystem>()->update(1);
  ASSERT_EQ(cpBodyGetPosition(*registry.get_component<KinematicComponent>(wall_id)->body), cpv(32, 32));
  ASSERT_THROW_MESSAGE(
      registry.get_component<KinematicComponent>(1), RegistryError,
      "The component `KinematicComponent` for the game object ID `1` is not registered with the registry.")
}

/// Test that an event is not notified if there are no callbacks added to the registry.
TEST_F(RegistryFixture, TestRegistryNotifyCallbacksNoCallbacksAdded) {
  constexpr bool called{false};
  registry.notify_callbacks(EventType::GameObjectDeath, 0);
  ASSERT_FALSE(called);
}

/// Test that an event is not notified if there are no callbacks listening for that event.
TEST_F(RegistryFixture, TestRegistryNotifyCallbacksNoCallbacksListening) {
  auto called{-1};
  registry.add_callback(EventType::GameObjectCreation, [&called](const auto event) { called = event; });
  registry.notify_callbacks(EventType::GameObjectDeath, 0);
  ASSERT_EQ(called, -1);
}

/// Test that an event is notified correctly if there is a callback listening for that event.
TEST_F(RegistryFixture, TestRegistryNotifyCallbacksListeningCallback) {
  auto called{-1};
  registry.add_callback(EventType::GameObjectDeath, [&called](const auto event) { called = event; });
  registry.notify_callbacks(EventType::GameObjectDeath, 0);
  ASSERT_EQ(called, 0);
}
