// Local headers
#include "ecs/systems/movements.hpp"
#include "ecs/systems/physics.hpp"
#include "macros.hpp"

/// Implements the fixture for the PhysicsSystem tests.
class PhysicsSystemFixture : public testing::Test {
 protected:
  /// A random generator for use in testing.
  std::mt19937 random_generator;

  /// The registry that manages the game objects, components, and systems.
  Registry registry{random_generator};

  /// Set up the fixture for the tests.
  void SetUp() override {
    registry.create_game_object(
        GameObjectType::Player, cpvzero,
        {std::make_shared<KinematicComponent>(std::vector<cpVect>{{0.0, 1.0}, {1.0, 2.0}, {2.0, 0.0}}),
         std::make_shared<MovementForce>(100, -1)});
    registry.add_system<PhysicsSystem>();
  }

  /// Create objects at the specified positions.
  ///
  /// @param type The type of the objects to create.
  /// @param positions The positions to create the objects.
  /// @param override Whether to override the game objects' positions.
  void create_objects(const GameObjectType type, const std::vector<cpVect> &positions, const bool override = true) {
    for (const auto &position : positions) {
      const auto object_id{registry.create_game_object(
          type, position, {std::make_shared<KinematicComponent>(type == GameObjectType::Wall)})};
      if (override) {
        cpBodySetPosition(*registry.get_component<KinematicComponent>(object_id)->body, position);
      }
    }
  }

  /// Get the physics system from the registry.
  ///
  /// @return The physics system.
  [[nodiscard]] auto get_physics_system() const -> std::shared_ptr<PhysicsSystem> {
    return registry.get_system<PhysicsSystem>();
  }
};

/// Test updating the physics system with a game object that has no velocity and no force.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemUpdateNoVelocityNoForce) {
  get_physics_system()->update(1.0);
  ASSERT_EQ(cpBodyGetPosition(*registry.get_component<KinematicComponent>(0)->body), cpv(32, 32));
  ASSERT_EQ(cpBodyGetVelocity(*registry.get_component<KinematicComponent>(0)->body), cpvzero);
  ASSERT_EQ(cpBodyGetForce(*registry.get_component<KinematicComponent>(0)->body), cpvzero);
}

/// Test updating the physics system with a game object that has no velocity and a force.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemUpdateNoVelocityValidForce) {
  cpBodySetForce(*registry.get_component<KinematicComponent>(0)->body, {.x = 10, .y = 0});
  get_physics_system()->update(1.0);
  ASSERT_EQ(cpBodyGetPosition(*registry.get_component<KinematicComponent>(0)->body), cpv(32, 32));
  ASSERT_EQ(cpBodyGetVelocity(*registry.get_component<KinematicComponent>(0)->body), cpv(10, 0));
  ASSERT_EQ(cpBodyGetForce(*registry.get_component<KinematicComponent>(0)->body), cpvzero);
}

/// Test updating the physics system with a game object that has velocity and no force.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemUpdateValidVelocityNoForce) {
  cpBodySetVelocity(*registry.get_component<KinematicComponent>(0)->body, {.x = 100, .y = 0});
  get_physics_system()->update(1.0);
  ASSERT_EQ(cpBodyGetPosition(*registry.get_component<KinematicComponent>(0)->body), cpv(132, 32));
  ASSERT_EQ(cpBodyGetVelocity(*registry.get_component<KinematicComponent>(0)->body), cpv(0.01, 0));
  ASSERT_EQ(cpBodyGetForce(*registry.get_component<KinematicComponent>(0)->body), cpvzero);
}

/// Test updating the physics system with a game object that has velocity and a force.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemUpdateValidVelocityValidForce) {
  cpBodySetVelocity(*registry.get_component<KinematicComponent>(0)->body, {.x = 100, .y = 0});
  cpBodySetForce(*registry.get_component<KinematicComponent>(0)->body, {.x = 10, .y = 0});
  get_physics_system()->update(1.0);
  ASSERT_EQ(cpBodyGetPosition(*registry.get_component<KinematicComponent>(0)->body), cpv(132, 32));
  ASSERT_EQ(cpBodyGetVelocity(*registry.get_component<KinematicComponent>(0)->body), cpv(10.01, 0));
  ASSERT_EQ(cpBodyGetForce(*registry.get_component<KinematicComponent>(0)->body), cpvzero);
}

/// Test that adding a zero force to a game object without a force works correctly.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemAddForceZeroForceNoForce) {
  get_physics_system()->add_force(0, cpvzero);
  ASSERT_EQ(cpBodyGetForce(*registry.get_component<KinematicComponent>(0)->body), cpvzero);
}

/// Test that adding a positive force to a game object without a force works correctly.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemAddForcePositiveForceNoForce) {
  get_physics_system()->add_force(0, {10, 0});
  ASSERT_EQ(cpBodyGetForce(*registry.get_component<KinematicComponent>(0)->body), cpv(100, 0));
}

/// Test that adding a negative force to a game object without a force works correctly.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemAddForceNegativeForceNoForce) {
  get_physics_system()->add_force(0, {-10, 0});
  ASSERT_EQ(cpBodyGetForce(*registry.get_component<KinematicComponent>(0)->body), cpv(-100, 0));
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
  ASSERT_EQ(cpBodyGetForce(*registry.get_component<KinematicComponent>(0)->body), cpv(110, 0));
}

/// Test that adding a force to a static wall body doesn't change its position.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemAddForceStaticWall) {
  // Walls should never have a MovementForce component, but if they do, their position should not change
  const auto wall_id{registry.create_game_object(
      GameObjectType::Wall, cpvzero,
      {std::make_shared<KinematicComponent>(true), std::make_shared<MovementForce>(100, -1)})};
  get_physics_system()->add_force(wall_id, {10, 0});
  get_physics_system()->update(1);
  ASSERT_EQ(cpBodyGetPosition(*registry.get_component<KinematicComponent>(wall_id)->body), cpv(32, 32));
}

/// Test that an exception is thrown if a game object does not have a kinematic component.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemAddForceNonexistentKinematicComponent) {
  registry.create_game_object(GameObjectType::Player, cpvzero, {std::make_shared<MovementForce>(100, -1)});
  ASSERT_THROW_MESSAGE(
      get_physics_system()->add_force(1, {0, 0}), RegistryError,
      "The component `KinematicComponent` for the game object ID `1` is not registered with the registry.");
}

/// Test that an exception is thrown if a game object does not have a movement force component.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemAddForceNonexistentMovementForceComponent) {
  registry.create_game_object(GameObjectType::Player, cpvzero, {std::make_shared<KinematicComponent>()});
  ASSERT_THROW_MESSAGE(get_physics_system()->add_force(1, {0, 0}), RegistryError,
                       "The component `MovementForce` for the game object ID `1` is not registered with the registry.");
}

/// Test that an exception is thrown if an invalid game object ID is provided.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemAddForceInvalidGameObjectId) {
  ASSERT_THROW_MESSAGE(
      get_physics_system()->add_force(-1, {0, 0}), RegistryError,
      "The component `MovementForce` for the game object ID `-1` is not registered with the registry.");
}

/// Test that adding a bullet with a zero position and velocity works correctly.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemAddBulletZero) {
  get_physics_system()->add_bullet({cpvzero, cpvzero}, 50, GameObjectType::Player);
  const auto *body{*registry.get_component<KinematicComponent>(1)->body};
  ASSERT_EQ(cpBodyGetPosition(body), cpvzero);
  ASSERT_EQ(cpBodyGetForce(body), cpvzero);
}

/// Test that adding a bullet with a non-zero position works correctly.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemAddBulletNonZeroPosition) {
  get_physics_system()->add_bullet({{.x = 100, .y = 0}, cpvzero}, 100, GameObjectType::Player);
  const auto *body{*registry.get_component<KinematicComponent>(1)->body};
  ASSERT_EQ(cpBodyGetPosition(body), cpv(100, 0));
  ASSERT_EQ(cpBodyGetForce(body), cpvzero);
}

/// Test that adding a bullet with a non-zero velocity works correctly.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemAddBulletNonZeroVelocity) {
  get_physics_system()->add_bullet({cpvzero, {.x = 200, .y = 150}}, 10, GameObjectType::Player);
  const auto *body{*registry.get_component<KinematicComponent>(1)->body};
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
  create_objects(GameObjectType::HealthPotion, {{.x = 65, .y = 32}});
  ASSERT_EQ(get_physics_system()->get_nearest_item(0), -1);
}

/// Test that getting the nearest item works if the game object is touching the item.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemGetNearestItemTouching) {
  create_objects(GameObjectType::HealthPotion, {{.x = 64, .y = 32}});
  ASSERT_EQ(get_physics_system()->get_nearest_item(0), 1);
}

/// Test that getting the nearest item works if the game object is on top of the item.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemGetNearestItemOnTopOf) {
  create_objects(GameObjectType::HealthPotion, {{.x = 32, .y = 32}});
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

/// Test that only the bottom three raycasts hit when getting the distances to the walls when there is one wall tile.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemGetWallDistancesOneWallTile) {
  create_objects(GameObjectType::Wall, {cpv(1, 0)}, false);
  const auto result{get_physics_system()->get_wall_distances({96, 96})};
  const std::vector expected_result{cpv(MAX_WALL_DISTANCE, MAX_WALL_DISTANCE),
                                    cpv(MAX_WALL_DISTANCE, MAX_WALL_DISTANCE),
                                    cpv(96, 64),
                                    cpv(MAX_WALL_DISTANCE, MAX_WALL_DISTANCE),
                                    cpv(MAX_WALL_DISTANCE, MAX_WALL_DISTANCE),
                                    cpv(112, 64),
                                    cpv(MAX_WALL_DISTANCE, MAX_WALL_DISTANCE),
                                    cpv(80, 64)};
  for (auto i{0}; i < static_cast<int>(result.size()); i++) {
    ASSERT_DOUBLE_EQ(result[i].x, expected_result[i].x);
    ASSERT_DOUBLE_EQ(result[i].y, expected_result[i].y);
  }
}

/// Test that all raycasts hit when getting the distances to the walls when there are four wall tiles.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemGetWallDistancesFourWallTiles) {
  create_objects(GameObjectType::Wall, {cpv(1, 0), cpv(0, 1), cpv(2, 1), cpv(1, 2)}, false);
  const auto result{get_physics_system()->get_wall_distances({96, 96})};
  const std::vector expected_result{cpv(96, 128),  cpv(128, 96), cpv(96, 64),  cpv(64, 96),
                                    cpv(112, 128), cpv(112, 64), cpv(80, 128), cpv(80, 64)};
  for (auto i{0}; i < static_cast<int>(result.size()); i++) {
    ASSERT_DOUBLE_EQ(result[i].x, expected_result[i].x);
    ASSERT_DOUBLE_EQ(result[i].y, expected_result[i].y);
  }
}

/// Test that all raycasts hit when getting the distances to the walls when there are eight wall tiles.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemGetWallDistancesEightWallTiles) {
  create_objects(GameObjectType::Wall,
                 {cpv(0, 0), cpv(1, 0), cpv(2, 0), cpv(0, 1), cpv(2, 1), cpv(0, 2), cpv(1, 2), cpv(2, 2)}, false);
  const auto result{get_physics_system()->get_wall_distances({96, 96})};
  const std::vector expected_result{cpv(96, 128),  cpv(128, 96), cpv(96, 64),  cpv(64, 96),
                                    cpv(112, 128), cpv(112, 64), cpv(80, 128), cpv(80, 64)};
  for (auto i{0}; i < static_cast<int>(result.size()); i++) {
    ASSERT_DOUBLE_EQ(result[i].x, expected_result[i].x);
    ASSERT_DOUBLE_EQ(result[i].y, expected_result[i].y);
  }
}
