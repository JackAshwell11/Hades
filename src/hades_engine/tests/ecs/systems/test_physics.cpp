// Local headers
#include "ecs/registry.hpp"
#include "ecs/systems/movements.hpp"
#include "ecs/systems/physics.hpp"
#include "events.hpp"
#include "factories.hpp"
#include "macros.hpp"

/// Implements the fixture for the PhysicsSystem tests.
class PhysicsSystemFixture : public testing::Test {
 protected:
  /// The registry that manages the game objects, components, and systems.
  Registry registry;

  /// Set up the fixture for the tests.
  void SetUp() override {
    const auto game_object_id{registry.create_game_object(GameObjectType::Player)};
    registry.add_component<KinematicComponent>(
        game_object_id, cpvzero, std::vector<cpVect>{{.x = 0.0, .y = 1.0}, {.x = 1.0, .y = 2.0}, {.x = 2.0, .y = 0.0}});
    registry.add_system<PhysicsSystem>();
  }

  /// Tear down the fixture after the tests.
  void TearDown() override { clear_listeners(); }

  /// Create objects at the specified positions.
  ///
  /// @param type The type of the objects to create.
  /// @param positions The positions to create the objects.
  void create_objects(const GameObjectType type, const std::vector<cpVect>& positions) {
    for (const auto& position : positions) {
      const auto object_id{registry.create_game_object(type)};
      registry.add_component<KinematicComponent>(object_id, position, type == GameObjectType::Wall);
    }
    registry.get_system<PhysicsSystem>()->update(1);
  }

  /// Get the physics system from the registry.
  ///
  /// @return The physics system.
  [[nodiscard]] auto get_physics_system() const -> std::shared_ptr<PhysicsSystem> {
    return registry.get_system<PhysicsSystem>();
  }
};

/// Test that providing an invalid number of vertices throws an exception.
TEST_F(PhysicsSystemFixture, TestKinematicComponentInvalidVertices) {
  ASSERT_THROW_MESSAGE(KinematicComponent(cpvzero, std::vector<cpVect>{{0.0, 1.0}}), std::invalid_argument,
                       "The shape must have at least 3 vertices.");
}

/// Test that providing a position sets the body position correctly.
TEST_F(PhysicsSystemFixture, TestKinematicComponentPosition) {
  const KinematicComponent kinematic_component{{.x = 10, .y = 20}};
  ASSERT_EQ(cpBodyGetPosition(*kinematic_component.body), cpv(10, 20));
}

/// Test updating the physics system with a game object that has no velocity and no force.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemUpdateNoVelocityNoForce) {
  get_physics_system()->update(1.0);
  ASSERT_EQ(cpBodyGetPosition(*registry.get_component<KinematicComponent>(0)->body), cpvzero);
  ASSERT_EQ(cpBodyGetVelocity(*registry.get_component<KinematicComponent>(0)->body), cpvzero);
  ASSERT_EQ(cpBodyGetForce(*registry.get_component<KinematicComponent>(0)->body), cpvzero);
}

/// Test updating the physics system with a game object that has no velocity and a force.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemUpdateNoVelocityValidForce) {
  cpBodySetForce(*registry.get_component<KinematicComponent>(0)->body, {.x = 10, .y = 0});
  get_physics_system()->update(1.0);
  ASSERT_EQ(cpBodyGetPosition(*registry.get_component<KinematicComponent>(0)->body), cpvzero);
  ASSERT_EQ(cpBodyGetVelocity(*registry.get_component<KinematicComponent>(0)->body), cpv(10, 0));
  ASSERT_EQ(cpBodyGetForce(*registry.get_component<KinematicComponent>(0)->body), cpvzero);
}

/// Test updating the physics system with a game object that has velocity and no force.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemUpdateValidVelocityNoForce) {
  cpBodySetVelocity(*registry.get_component<KinematicComponent>(0)->body, {.x = 100, .y = 0});
  get_physics_system()->update(1.0);
  ASSERT_EQ(cpBodyGetPosition(*registry.get_component<KinematicComponent>(0)->body), cpv(100, 0));
  ASSERT_EQ(cpBodyGetVelocity(*registry.get_component<KinematicComponent>(0)->body), cpv(0.01, 0));
  ASSERT_EQ(cpBodyGetForce(*registry.get_component<KinematicComponent>(0)->body), cpvzero);
}

/// Test updating the physics system with a game object that has velocity and a force.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemUpdateValidVelocityValidForce) {
  cpBodySetVelocity(*registry.get_component<KinematicComponent>(0)->body, {.x = 100, .y = 0});
  cpBodySetForce(*registry.get_component<KinematicComponent>(0)->body, {.x = 10, .y = 0});
  get_physics_system()->update(1.0);
  ASSERT_EQ(cpBodyGetPosition(*registry.get_component<KinematicComponent>(0)->body), cpv(100, 0));
  ASSERT_EQ(cpBodyGetVelocity(*registry.get_component<KinematicComponent>(0)->body), cpv(10.01, 0));
  ASSERT_EQ(cpBodyGetForce(*registry.get_component<KinematicComponent>(0)->body), cpvzero);
}

/// Test that updating the physics system with a player game object calls the correct callbacks.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemUpdatePlayerCallbacks) {
  std::vector<std::pair<double, double>> changed_positions;
  const auto position_changed{[&changed_positions](const GameObjectID, const std::pair<double, double>& position) {
    changed_positions.push_back(position);
  }};
  add_callback<EventType::PositionChanged>(position_changed);
  cpBodySetVelocity(*registry.get_component<KinematicComponent>(0)->body, {.x = 100, .y = 0});
  get_physics_system()->update(5.0);
  const std::vector<std::pair<double, double>> expected_positions{{500, 0}};
  ASSERT_EQ(changed_positions, expected_positions);
}

/// Test that updating the physics system with enemy game objects calls the correct callbacks.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemUpdateEnemyCallbacks) {
  registry.delete_game_object(0);
  create_objects(GameObjectType::Enemy, {{.x = 0, .y = 0}, {.x = 100, .y = 0}, {.x = 200, .y = 0}});
  cpBodySetVelocity(*registry.get_component<KinematicComponent>(1)->body, {.x = 100, .y = 0});
  cpBodySetVelocity(*registry.get_component<KinematicComponent>(2)->body, {.x = 50, .y = 0});
  cpBodySetVelocity(*registry.get_component<KinematicComponent>(0)->body, {.x = 25, .y = 0});
  std::vector<std::pair<double, double>> changed_positions;
  const auto position_changed{[&changed_positions](const GameObjectID, const std::pair<double, double>& position) {
    changed_positions.push_back(position);
  }};
  add_callback<EventType::PositionChanged>(position_changed);
  get_physics_system()->update(5.0);
  const std::vector<std::pair<double, double>> expected_positions{{125, 0}, {600, 0}, {450, 0}};
  ASSERT_EQ(changed_positions, expected_positions);
}

/// Tests that updating the physics systems with bullet game objects calls the correct callbacks.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemUpdateBulletCallbacks) {
  registry.delete_game_object(0);
  create_objects(GameObjectType::Bullet, {{.x = 0, .y = 200}, {.x = 0, .y = 300}, {.x = 0, .y = 400}});
  cpBodySetVelocity(*registry.get_component<KinematicComponent>(1)->body, {.x = 0, .y = 150});
  cpBodySetVelocity(*registry.get_component<KinematicComponent>(2)->body, {.x = 0, .y = 100});
  cpBodySetVelocity(*registry.get_component<KinematicComponent>(0)->body, {.x = 0, .y = 50});
  std::vector<std::pair<double, double>> changed_positions;
  const auto position_changed{[&changed_positions](const GameObjectID, const std::pair<double, double>& position) {
    changed_positions.push_back(position);
  }};
  add_callback<EventType::PositionChanged>(position_changed);
  get_physics_system()->update(5.0);
  const std::vector<std::pair<double, double>> expected_positions{{0, 450}, {0, 1050}, {0, 900}};
  ASSERT_EQ(changed_positions, expected_positions);
}

/// Test that adding a zero force to a game object without a force works correctly.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemAddForceZeroForceNoForce) {
  get_physics_system()->add_force(0, cpvzero);
  ASSERT_EQ(cpBodyGetForce(*registry.get_component<KinematicComponent>(0)->body), cpvzero);
}

/// Test that adding a positive force to a game object without a force works correctly.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemAddForcePositiveForceNoForce) {
  get_physics_system()->add_force(0, {10, 0});
  ASSERT_EQ(cpBodyGetForce(*registry.get_component<KinematicComponent>(0)->body), cpv(5000, 0));
}

/// Test that adding a negative force to a game object without a force works correctly.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemAddForceNegativeForceNoForce) {
  get_physics_system()->add_force(0, {-10, 0});
  ASSERT_EQ(cpBodyGetForce(*registry.get_component<KinematicComponent>(0)->body), cpv(-5000, 0));
}

/// Test that adding a positive infinite force to a game object without a force works correctly.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemAddForcePositiveInfiniteForceNoForce) {
  get_physics_system()->add_force(0,
                                  {std::numeric_limits<cpFloat>::infinity(), std::numeric_limits<cpFloat>::infinity()});
  ASSERT_TRUE(std::isnan(cpBodyGetForce(*registry.get_component<KinematicComponent>(0)->body).x));
  ASSERT_TRUE(std::isnan(cpBodyGetForce(*registry.get_component<KinematicComponent>(0)->body).y));
}

/// Test that adding a positive force to a game object with a force works correctly.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemAddForcePositiveForceValidForce) {
  cpBodySetForce(*registry.get_component<KinematicComponent>(0)->body, {.x = 10, .y = 0});
  get_physics_system()->add_force(0, {10, 0});
  ASSERT_EQ(cpBodyGetForce(*registry.get_component<KinematicComponent>(0)->body), cpv(5010, 0));
}

/// Test that adding a force to a static wall body doesn't change its position.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemAddForceStaticWall) {
  const auto wall_id{registry.create_game_object(GameObjectType::Wall)};
  registry.add_component<KinematicComponent>(wall_id, cpvzero, true);
  get_physics_system()->add_force(wall_id, {10, 0});
  get_physics_system()->update(1);
  ASSERT_EQ(cpBodyGetPosition(*registry.get_component<KinematicComponent>(wall_id)->body), cpvzero);
}

/// Test that an exception is thrown if a game object does not have a kinematic component.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemAddForceNonexistentKinematicComponent) {
  registry.create_game_object(GameObjectType::Player);
  ASSERT_THROW_MESSAGE(
      get_physics_system()->add_force(1, {0, 0}), RegistryError,
      "The component `KinematicComponent` for the game object ID `1` is not registered with the registry.");
}

/// Test that an exception is thrown if an invalid game object ID is provided.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemAddForceInvalidGameObjectId) {
  ASSERT_THROW_MESSAGE(get_physics_system()->add_force(-1, {0, 0}), RegistryError,
                       "The game object ID `-1` is not registered with the registry.");
}

/// Test that adding a bullet with a zero position and velocity works correctly.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemAddBulletZero) {
  get_physics_system()->add_bullet({cpvzero, cpvzero}, 50, GameObjectType::Player);
  const auto* body{*registry.get_component<KinematicComponent>(1)->body};
  ASSERT_EQ(cpBodyGetPosition(body), cpvzero);
  ASSERT_EQ(cpBodyGetForce(body), cpvzero);
}

/// Test that adding a bullet with a non-zero position works correctly.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemAddBulletNonZeroPosition) {
  get_physics_system()->add_bullet({{.x = 100, .y = 0}, cpvzero}, 100, GameObjectType::Player);
  const auto* body{*registry.get_component<KinematicComponent>(1)->body};
  ASSERT_EQ(cpBodyGetPosition(body), cpv(100, 0));
  ASSERT_EQ(cpBodyGetForce(body), cpvzero);
}

/// Test that adding a bullet with a non-zero velocity works correctly.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemAddBulletNonZeroVelocity) {
  get_physics_system()->add_bullet({cpvzero, {.x = 200, .y = 150}}, 10, GameObjectType::Player);
  const auto* body{*registry.get_component<KinematicComponent>(1)->body};
  ASSERT_EQ(cpBodyGetPosition(body), cpvzero);
  ASSERT_EQ(cpBodyGetVelocity(body), cpv(200, 150));
}

/// Test that getting the nearest item doesn't work if the game object is far away.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemGetNearestItemFarAway) {
  create_objects(GameObjectType::HealthPotion, {{.x = 100, .y = 100}});
  ASSERT_EQ(get_physics_system()->get_nearest_item(0), -1);
}

/// Test that getting the nearest item doesn't work if the game object is next to the item.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemGetNearestItemNextTo) {
  // If a shape is at the maximum distance, it is not accepted, its distance must be less
  create_objects(GameObjectType::HealthPotion, {{.x = 64, .y = 0}});
  ASSERT_EQ(get_physics_system()->get_nearest_item(0), -1);
}

/// Test that getting the nearest item works if the game object is close enough to the item.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemGetNearestItemWithinMinimum) {
  create_objects(GameObjectType::HealthPotion, {{.x = 50, .y = 32}});
  ASSERT_EQ(get_physics_system()->get_nearest_item(0), 1);
}

/// Test that getting the nearest item works if the game object is on top of the item.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemGetNearestItemOnTopOf) {
  create_objects(GameObjectType::HealthPotion, {{.x = 0, .y = 0}});
  ASSERT_EQ(get_physics_system()->get_nearest_item(0), 1);
}

/// Test that getting the nearest item works if the game object is touching the item and there are multiple items.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemGetNearestItemMultipleItems) {
  create_objects(GameObjectType::HealthPotion, {{.x = 48, .y = 32}, {.x = 32, .y = 32}});
  ASSERT_EQ(get_physics_system()->get_nearest_item(0), 2);
}

/// Test that no raycasts hit when getting the distances to the walls when there are no walls.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemGetWallDistancesNoWalls) {
  const std::vector expected_result(8, cpv(MAX_WALL_DISTANCE, MAX_WALL_DISTANCE));
  ASSERT_EQ(get_physics_system()->get_wall_distances({0, 0}), expected_result);
}

/// Test that only the bottom three raycasts hit when getting the distances to the walls when there is one wall game
/// object.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemGetWallDistancesOneWallGameObject) {
  create_objects(GameObjectType::Wall, {{.x = 96, .y = 32}});
  const auto result{get_physics_system()->get_wall_distances({.x = 96, .y = 96})};
  const std::vector expected_result{cpv(MAX_WALL_DISTANCE, MAX_WALL_DISTANCE),
                                    cpv(MAX_WALL_DISTANCE, MAX_WALL_DISTANCE),
                                    cpv(96, 64),
                                    cpv(MAX_WALL_DISTANCE, MAX_WALL_DISTANCE),
                                    cpv(MAX_WALL_DISTANCE, MAX_WALL_DISTANCE),
                                    cpv(112, 64),
                                    cpv(MAX_WALL_DISTANCE, MAX_WALL_DISTANCE),
                                    cpv(80, 64)};
  for (auto i{0}; std::cmp_less(i, result.size()); i++) {
    ASSERT_DOUBLE_EQ(result[i].x, expected_result[i].x);
    ASSERT_DOUBLE_EQ(result[i].y, expected_result[i].y);
  }
}

/// Test that all raycasts hit when getting the distances to the walls when there are four wall game objects.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemGetWallDistancesFourWallGameObjects) {
  create_objects(GameObjectType::Wall,
                 {{.x = 96, .y = 32}, {.x = 32, .y = 96}, {.x = 160, .y = 96}, {.x = 96, .y = 160}});
  const auto result{get_physics_system()->get_wall_distances({96, 96})};
  const std::vector expected_result{cpv(96, 128),  cpv(128, 96), cpv(96, 64),  cpv(64, 96),
                                    cpv(112, 128), cpv(112, 64), cpv(80, 128), cpv(80, 64)};
  for (auto i{0}; std::cmp_less(i, result.size()); i++) {
    ASSERT_DOUBLE_EQ(result[i].x, expected_result[i].x);
    ASSERT_DOUBLE_EQ(result[i].y, expected_result[i].y);
  }
}

/// Test that all raycasts hit when getting the distances to the walls when there are eight wall game objects.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemGetWallDistancesEightWallGameObjects) {
  create_objects(GameObjectType::Wall, {{.x = 32, .y = 32},
                                        {.x = 96, .y = 32},
                                        {.x = 160, .y = 32},
                                        {.x = 32, .y = 96},
                                        {.x = 160, .y = 96},
                                        {.x = 32, .y = 160},
                                        {.x = 96, .y = 160},
                                        {.x = 160, .y = 160}});
  const auto result{get_physics_system()->get_wall_distances({96, 96})};
  const std::vector expected_result{cpv(96, 128),  cpv(128, 96), cpv(96, 64),  cpv(64, 96),
                                    cpv(112, 128), cpv(112, 64), cpv(80, 128), cpv(80, 64)};
  for (auto i{0}; std::cmp_less(i, result.size()); i++) {
    ASSERT_DOUBLE_EQ(result[i].x, expected_result[i].x);
    ASSERT_DOUBLE_EQ(result[i].y, expected_result[i].y);
  }
}
