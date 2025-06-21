// Std headers
#include <numbers>

// Local headers
#include "ecs/registry.hpp"
#include "ecs/systems/movements.hpp"
#include "ecs/systems/physics.hpp"
#include "macros.hpp"

namespace {
/// Test if two vectors are equal within a small margin of error.
///
/// @details This function is needed because ASSERT_DOUBLE_EQ does not work great with cpVect which uses floats
/// internally.
/// @param actual - The actual vector.
/// @param expected_x - The expected x value.
/// @param expected_y - The expected y value.
void test_force_double(const cpVect &actual, const double expected_x, const double expected_y) {
  ASSERT_DOUBLE_EQ(actual.x, expected_x);
  ASSERT_DOUBLE_EQ(actual.y, expected_y);
}
}  // namespace

/// Implements the fixture for the FootprintSystem tests.
class FootprintSystemFixture : public testing::Test {
 protected:
  /// The registry that manages the game objects, components, and systems.
  Registry registry;

  /// Set up the fixture for the tests.
  void SetUp() override {
    registry.create_game_object(
        GameObjectType::Player, cpvzero,
        {std::make_shared<Footprints>(), std::make_shared<KinematicComponent>(),
         std::make_shared<FootprintInterval>(0.2, -1), std::make_shared<FootprintLimit>(10, -1)});
    registry.add_system<FootprintSystem>();
    registry.add_system<SteeringMovementSystem>();

    // Set the position of the game object to (0, 0) since grid_pos_to_pixel() sets the position to (32, 32)
    cpBodySetPosition(*registry.get_component<KinematicComponent>(0)->body, cpvzero);
  }

  /// Get the footprint system from the registry.
  ///
  /// @return The footprint system.
  [[nodiscard]] auto get_footprint_system() const -> std::shared_ptr<FootprintSystem> {
    return registry.get_system<FootprintSystem>();
  }
};

/// Implements the fixture for the KeyboardMovementSystem tests.
class KeyboardMovementSystemFixture : public testing::Test {
 protected:
  /// The registry that manages the game objects, components, and systems.
  Registry registry;

  /// Set up the fixture for the tests.
  void SetUp() override {
    registry.create_game_object(GameObjectType::Player, cpvzero,
                                {std::make_shared<MovementForce>(100, -1), std::make_shared<KeyboardMovement>(),
                                 std::make_shared<KinematicComponent>()});
    registry.add_system<KeyboardMovementSystem>();
    registry.add_system<PhysicsSystem>();
  }

  /// Get the keyboard movement system from the registry.
  ///
  /// @return The keyboard movement system.
  [[nodiscard]] auto get_keyboard_movement_system() const -> std::shared_ptr<KeyboardMovementSystem> {
    return registry.get_system<KeyboardMovementSystem>();
  }
};

/// Implements the fixture for the SteeringMovementSystem tests.
class SteeringMovementSystemFixture : public testing::Test {
 protected:
  /// The registry that manages the game objects, components, and systems.
  Registry registry;

  /// Set up the fixture for the tests.
  void SetUp() override {
    // Create the target game object and add the required systems
    registry.create_game_object(GameObjectType::Player, cpvzero,
                                {std::make_shared<FootprintInterval>(0.3, -1), std::make_shared<FootprintLimit>(10, -1),
                                 std::make_shared<Footprints>(), std::make_shared<KinematicComponent>()});
    registry.add_system<FootprintSystem>();
    registry.add_system<PhysicsSystem>();
    registry.add_system<SteeringMovementSystem>();
  }

  /// Create a steering movement component.
  ///
  /// @param steering_behaviours - The steering behaviours to initialise the component with.
  /// @return The game object ID of the created game object.
  auto create_steering_movement_component(
      const std::unordered_map<SteeringMovementState, std::vector<SteeringBehaviours>> &steering_behaviours) -> int {
    const int game_object_id{registry.create_game_object(
        GameObjectType::Player, cpvzero,
        {std::make_shared<MovementForce>(200, -1), std::make_shared<SteeringMovement>(steering_behaviours),
         std::make_shared<KinematicComponent>(), std::make_shared<ViewDistance>(2 * SPRITE_SIZE, -1)})};
    registry.get_component<SteeringMovement>(game_object_id)->target_id = 0;
    return game_object_id;
  }

  /// Get the steering movement system from the registry.
  ///
  /// @return The steering movement system.
  [[nodiscard]] auto get_steering_movement_system() const -> std::shared_ptr<SteeringMovementSystem> {
    return registry.get_system<SteeringMovementSystem>();
  }
};

/// Test that the footprint systems is updated with a small delta time.
TEST_F(FootprintSystemFixture, TestFootprintSystemUpdateSmallDeltaTime) {
  get_footprint_system()->update(0.1);
  const auto footprints{registry.get_component<Footprints>(0)};
  ASSERT_EQ(footprints->footprints, std::deque<cpVect>{});
  ASSERT_EQ(footprints->time_since_last_footprint, 0.1);
}

/// Test that the footprint system creates a footprint in an empty list.
TEST_F(FootprintSystemFixture, TestFootprintSystemUpdateLargeDeltaTimeEmptyList) {
  get_footprint_system()->update(1);
  const auto footprints{registry.get_component<Footprints>(0)};
  ASSERT_EQ(footprints->footprints, std::deque{cpvzero});
  ASSERT_EQ(footprints->time_since_last_footprint, 0);
}

/// Test that the footprint system creates a footprint in a non-empty list.
TEST_F(FootprintSystemFixture, TestFootprintSystemUpdateLargeDeltaTimeNonEmptyList) {
  const auto footprints{registry.get_component<Footprints>(0)};
  footprints->footprints = {{.x = 1, .y = 1}, {.x = 2, .y = 2}, {.x = 3, .y = 3}};
  get_footprint_system()->update(0.5);
  const std::deque<cpVect> expected_footprints{{.x = 1, .y = 1}, {.x = 2, .y = 2}, {.x = 3, .y = 3}, {.x = 0, .y = 0}};
  ASSERT_EQ(footprints->footprints, expected_footprints);
}

/// Test that the footprint system creates a footprint and removes the oldest one.
TEST_F(FootprintSystemFixture, TestFootprintSystemUpdateLargeDeltaTimeFullList) {
  const auto footprints{registry.get_component<Footprints>(0)};
  footprints->footprints = {{.x = 1, .y = 1}, {.x = 2, .y = 2}, {.x = 3, .y = 3}, {.x = 4, .y = 4}, {.x = 5, .y = 5},
                            {.x = 6, .y = 6}, {.x = 7, .y = 7}, {.x = 8, .y = 8}, {.x = 9, .y = 9}, {.x = 10, .y = 10}};
  get_footprint_system()->update(0.5);
  const std::deque<cpVect> expected_footprints{{.x = 2, .y = 2},   {.x = 3, .y = 3}, {.x = 4, .y = 4}, {.x = 5, .y = 5},
                                               {.x = 6, .y = 6},   {.x = 7, .y = 7}, {.x = 8, .y = 8}, {.x = 9, .y = 9},
                                               {.x = 10, .y = 10}, {.x = 0, .y = 0}};
  ASSERT_EQ(footprints->footprints, expected_footprints);
}

/// Test that the footprint system is updated correctly multiple times.
TEST_F(FootprintSystemFixture, TestFootprintSystemUpdateMultipleUpdates) {
  const auto footprints{registry.get_component<Footprints>(0)};
  get_footprint_system()->update(0.6);
  ASSERT_EQ(footprints->footprints, std::deque{cpvzero});
  ASSERT_EQ(footprints->time_since_last_footprint, 0);
  cpBodySetPosition(*registry.get_component<KinematicComponent>(0)->body, {.x = 1, .y = 1});
  get_footprint_system()->update(0.7);
  const std::deque<cpVect> expected_footprints{{.x = 0, .y = 0}, {.x = 1, .y = 1}};
  ASSERT_EQ(footprints->footprints, expected_footprints);
  ASSERT_EQ(footprints->time_since_last_footprint, 0);
}

/// Test that the footprint system is not updated correctly if the game object does not have the required components.
TEST_F(FootprintSystemFixture, TestFootprintSystemUpdateIncompleteComponents) {
  registry.create_game_object(GameObjectType::Player, cpvzero,
                              {std::make_shared<SteeringMovement>(
                                  std::unordered_map<SteeringMovementState, std::vector<SteeringBehaviours>>{})});
  get_footprint_system()->update(0.1);
  ASSERT_EQ(registry.get_component<Footprints>(0)->footprints, std::deque<cpVect>{});
}

/// Test that the new force is updated correctly if no keys are pressed.
TEST_F(KeyboardMovementSystemFixture, TestKeyboardMovementSystemUpdateNoKeys) {
  get_keyboard_movement_system()->update(0);
  ASSERT_EQ(cpBodyGetForce(*registry.get_component<KinematicComponent>(0)->body), cpvzero);
}

/// Test that the new force is updated correctly if the north key is pressed.
TEST_F(KeyboardMovementSystemFixture, TestKeyboardMovementSystemUpdateNorth) {
  registry.get_component<KeyboardMovement>(0)->moving_north = true;
  get_keyboard_movement_system()->update(0);
  ASSERT_EQ(cpBodyGetForce(*registry.get_component<KinematicComponent>(0)->body), cpv(0, 100));
}

/// Test that the new force is updated correctly if the south key is pressed.
TEST_F(KeyboardMovementSystemFixture, TestKeyboardMovementSystemUpdateSouth) {
  registry.get_component<KeyboardMovement>(0)->moving_south = true;
  get_keyboard_movement_system()->update(0);
  ASSERT_EQ(cpBodyGetForce(*registry.get_component<KinematicComponent>(0)->body), cpv(0, -100));
}

/// Test that the new force is updated correctly if the east key is pressed.
TEST_F(KeyboardMovementSystemFixture, TestKeyboardMovementSystemUpdateEast) {
  registry.get_component<KeyboardMovement>(0)->moving_east = true;
  get_keyboard_movement_system()->update(0);
  ASSERT_EQ(cpBodyGetForce(*registry.get_component<KinematicComponent>(0)->body), cpv(100, 0));
}

/// Test that the new force is updated correctly if the west key is pressed.
TEST_F(KeyboardMovementSystemFixture, TestKeyboardMovementSystemUpdateWest) {
  registry.get_component<KeyboardMovement>(0)->moving_west = true;
  get_keyboard_movement_system()->update(0);
  ASSERT_EQ(cpBodyGetForce(*registry.get_component<KinematicComponent>(0)->body), cpv(-100, 0));
}

/// Test that the correct force is calculated if the east and west keys are pressed.
TEST_F(KeyboardMovementSystemFixture, TestKeyboardMovementSystemUpdateEastWest) {
  const auto keyboard_movement{registry.get_component<KeyboardMovement>(0)};
  keyboard_movement->moving_east = true;
  keyboard_movement->moving_west = true;
  get_keyboard_movement_system()->update(0);
  ASSERT_EQ(cpBodyGetForce(*registry.get_component<KinematicComponent>(0)->body), cpvzero);
}

/// Test that the correct force is calculated if the north and south are pressed.
TEST_F(KeyboardMovementSystemFixture, TestKeyboardMovementSystemUpdateNorthSouth) {
  const auto keyboard_movement{registry.get_component<KeyboardMovement>(0)};
  keyboard_movement->moving_east = true;
  keyboard_movement->moving_west = true;
  get_keyboard_movement_system()->update(0);
  ASSERT_EQ(cpBodyGetForce(*registry.get_component<KinematicComponent>(0)->body), cpvzero);
}

/// Test that the correct force is calculated if north and west keys are pressed.
TEST_F(KeyboardMovementSystemFixture, TestKeyboardMovementSystemUpdateNorthWest) {
  const auto keyboard_movement{registry.get_component<KeyboardMovement>(0)};
  keyboard_movement->moving_north = true;
  keyboard_movement->moving_west = true;
  get_keyboard_movement_system()->update(0);
  ASSERT_EQ(cpBodyGetForce(*registry.get_component<KinematicComponent>(0)->body),
            cpv(-70.710678118654741, 70.710678118654741));
}

/// Test that the correct force is calculated if south and east keys are pressed.
TEST_F(KeyboardMovementSystemFixture, TestKeyboardMovementSystemUpdateSouthEast) {
  const auto keyboard_movement{registry.get_component<KeyboardMovement>(0)};
  keyboard_movement->moving_south = true;
  keyboard_movement->moving_east = true;
  get_keyboard_movement_system()->update(0);
  ASSERT_EQ(cpBodyGetForce(*registry.get_component<KinematicComponent>(0)->body),
            cpv(70.710678118654741, -70.710678118654741));
}

/// Test that the correct force is not applied if the game object does not have the required components.
TEST_F(KeyboardMovementSystemFixture, TestKeyboardMovementSystemUpdateIncompleteComponents) {
  registry.create_game_object(GameObjectType::Player, cpvzero,
                              {std::make_shared<KinematicComponent>(), std::make_shared<MovementForce>(100, -1)});
  get_keyboard_movement_system()->update(0);
  ASSERT_EQ(cpBodyGetForce(*registry.get_component<KinematicComponent>(0)->body), cpvzero);
}

/// Test that a game object is not updated if the target ID is invalid.
TEST_F(SteeringMovementSystemFixture, TestSteeringMovementSystemUpdateInvalidTargetId) {
  create_steering_movement_component({});
  registry.get_component<SteeringMovement>(1)->target_id = -1;
  get_steering_movement_system()->update(0);
  ASSERT_EQ(cpBodyGetForce(*registry.get_component<KinematicComponent>(1)->body), cpvzero);
}

/// Test that the state is correctly changed to the default state.
TEST_F(SteeringMovementSystemFixture, TestSteeringMovementSystemUpdateOutsideDistanceEmptyPathList) {
  create_steering_movement_component({});
  cpBodySetPosition(*registry.get_component<KinematicComponent>(1)->body, {.x = 500, .y = 500});
  get_steering_movement_system()->update(0);
  ASSERT_EQ(registry.get_component<SteeringMovement>(1)->movement_state, SteeringMovementState::Default);
}

/// Test that the state is correctly changed to the footprint state.
TEST_F(SteeringMovementSystemFixture, TestSteeringMovementSystemUpdateOutsideDistanceNonEmptyPathList) {
  create_steering_movement_component({});
  cpBodySetPosition(*registry.get_component<KinematicComponent>(1)->body, {.x = 500, .y = 500});
  const auto steering_movement{registry.get_component<SteeringMovement>(1)};
  steering_movement->path_list = {{.x = 300, .y = 300}, {.x = 400, .y = 400}};
  get_steering_movement_system()->update(0);
  ASSERT_EQ(steering_movement->movement_state, SteeringMovementState::Footprint);
}

/// Test that the state is correctly changed to the target stat.
TEST_F(SteeringMovementSystemFixture, TestSteeringMovementSystemUpdateWithinDistance) {
  create_steering_movement_component({});
  get_steering_movement_system()->update(0);
  ASSERT_EQ(registry.get_component<SteeringMovement>(1)->movement_state, SteeringMovementState::Target);
}

/// Test that the correct force is calculated if no behaviours are selected.
TEST_F(SteeringMovementSystemFixture, TestSteeringMovementSystemUpdateNoBehaviours) {
  create_steering_movement_component({});
  get_steering_movement_system()->update(0);
  ASSERT_EQ(cpBodyGetForce(*registry.get_component<KinematicComponent>(1)->body), cpvzero);
  ASSERT_DOUBLE_EQ(registry.get_component<KinematicComponent>(1)->rotation, 0);
}

/// Test that the correct force is calculated for the arrive behaviour.
TEST_F(SteeringMovementSystemFixture, TestSteeringMovementSystemUpdateArrive) {
  create_steering_movement_component({{SteeringMovementState::Target, {SteeringBehaviours::Arrive}}});
  cpBodySetPosition(*registry.get_component<KinematicComponent>(0)->body, cpvzero);
  cpBodySetPosition(*registry.get_component<KinematicComponent>(1)->body, {.x = 50, .y = 50});
  get_steering_movement_system()->update(0);
  test_force_double(cpBodyGetForce(*registry.get_component<KinematicComponent>(1)->body), -141.42135623730951,
                    -141.42135623730951);
  ASSERT_DOUBLE_EQ(registry.get_component<KinematicComponent>(1)->rotation, -2.3561944901923448);
}

/// Test that the correct force is calculated for the evade behaviour.
TEST_F(SteeringMovementSystemFixture, TestSteeringMovementSystemUpdateEvade) {
  create_steering_movement_component({{SteeringMovementState::Target, {SteeringBehaviours::Evade}}});
  cpBodySetPosition(*registry.get_component<KinematicComponent>(0)->body, {.x = 100, .y = 100});
  cpBodySetVelocity(*registry.get_component<KinematicComponent>(0)->body, {.x = -50, .y = 0});
  get_steering_movement_system()->update(0);
  test_force_double(cpBodyGetForce(*registry.get_component<KinematicComponent>(1)->body), -51.17851206325652,
                    -193.34104557230242);
  ASSERT_DOUBLE_EQ(registry.get_component<KinematicComponent>(1)->rotation, -1.8295672187585943);
}

/// Test that the correct force is calculated for the flee behaviour.
TEST_F(SteeringMovementSystemFixture, TestSteeringMovementSystemUpdateFlee) {
  create_steering_movement_component({{SteeringMovementState::Target, {SteeringBehaviours::Flee}}});
  cpBodySetPosition(*registry.get_component<KinematicComponent>(0)->body, {.x = 50, .y = 50});
  cpBodySetPosition(*registry.get_component<KinematicComponent>(1)->body, {.x = 100, .y = 100});
  get_steering_movement_system()->update(0);
  test_force_double(cpBodyGetForce(*registry.get_component<KinematicComponent>(1)->body), 141.42135623730951,
                    141.42135623730951);
  test_force_double(cpBodyGetForce(*registry.get_component<KinematicComponent>(1)->body), 141.42135623730951,
                    141.42135623730951);
  ASSERT_DOUBLE_EQ(registry.get_component<KinematicComponent>(1)->rotation, 0.7853981633974483);
}

/// Test that the correct force is calculated for the follow path behaviour.
TEST_F(SteeringMovementSystemFixture, TestSteeringMovementSystemUpdateFollowPath) {
  create_steering_movement_component({{SteeringMovementState::Footprint, {SteeringBehaviours::FollowPath}}});
  cpBodySetPosition(*registry.get_component<KinematicComponent>(1)->body, {.x = 200, .y = 200});
  registry.get_component<SteeringMovement>(1)->path_list = {{.x = 350, .y = 350}, {.x = 500, .y = 500}};
  get_steering_movement_system()->update(0);
  test_force_double(cpBodyGetForce(*registry.get_component<KinematicComponent>(1)->body), 141.42135623730951,
                    141.42135623730951);
  ASSERT_DOUBLE_EQ(registry.get_component<KinematicComponent>(1)->rotation, 0.7853981633974483);
}

/// Test that the correct force is calculated for the obstacle avoidance behaviour.
TEST_F(SteeringMovementSystemFixture, TestSteeringMovementSystemUpdateObstacleAvoidance) {
  create_steering_movement_component({{SteeringMovementState::Target, {SteeringBehaviours::ObstacleAvoidance}}});
  cpBodySetPosition(*registry.get_component<KinematicComponent>(1)->body, {.x = 32, .y = 96});
  cpBodySetVelocity(*registry.get_component<KinematicComponent>(1)->body, {.x = 100, .y = 100});
  registry.create_game_object(GameObjectType::Wall, {.x = 1, .y = 2}, {std::make_shared<KinematicComponent>(true)});
  get_steering_movement_system()->update(0);
  test_force_double(cpBodyGetForce(*registry.get_component<KinematicComponent>(1)->body), -141.42135623730951,
                    -141.42135623730951);
  ASSERT_DOUBLE_EQ(registry.get_component<KinematicComponent>(1)->rotation, -2.3561944901923448);
}

/// Test that the correct force is calculated for the pursue behaviour.
TEST_F(SteeringMovementSystemFixture, TestSteeringMovementSystemUpdatePursue) {
  create_steering_movement_component({{SteeringMovementState::Target, {SteeringBehaviours::Pursue}}});
  cpBodySetPosition(*registry.get_component<KinematicComponent>(0)->body, {.x = 100, .y = 100});
  cpBodySetVelocity(*registry.get_component<KinematicComponent>(0)->body, {.x = -50, .y = 0});
  get_steering_movement_system()->update(0);
  test_force_double(cpBodyGetForce(*registry.get_component<KinematicComponent>(1)->body), 51.17851206325652,
                    193.34104557230242);
  ASSERT_DOUBLE_EQ(registry.get_component<KinematicComponent>(1)->rotation, 1.312025434831199);
}

/// Test that the correct force is calculated for the seek behaviour.
TEST_F(SteeringMovementSystemFixture, TestSteeringMovementSystemUpdateSeek) {
  create_steering_movement_component({{SteeringMovementState::Target, {SteeringBehaviours::Seek}}});
  cpBodySetPosition(*registry.get_component<KinematicComponent>(0)->body, {.x = 50, .y = 50});
  cpBodySetPosition(*registry.get_component<KinematicComponent>(1)->body, {.x = 100, .y = 100});
  get_steering_movement_system()->update(0);
  test_force_double(cpBodyGetForce(*registry.get_component<KinematicComponent>(1)->body), -141.42135623730951,
                    -141.42135623730951);
  ASSERT_DOUBLE_EQ(registry.get_component<KinematicComponent>(1)->rotation, -2.3561944901923448);
}

/// Test that the correct force is calculated for the wander behaviour.
TEST_F(SteeringMovementSystemFixture, TestSteeringMovementSystemUpdateWander) {
  create_steering_movement_component({{SteeringMovementState::Target, {SteeringBehaviours::Wander}}});
  cpBodySetVelocity(*registry.get_component<KinematicComponent>(1)->body, {.x = 100, .y = -100});
  const auto before_force{cpBodyGetForce(*registry.get_component<KinematicComponent>(1)->body)};
  get_steering_movement_system()->update(0);
  const auto after_force{cpBodyGetForce(*registry.get_component<KinematicComponent>(1)->body)};
  ASSERT_EQ(round(cpvlength(after_force)), 200);
  ASSERT_NE(before_force, after_force);
  ASSERT_GE(registry.get_component<KinematicComponent>(1)->rotation, -std::numbers::pi);
  ASSERT_LE(registry.get_component<KinematicComponent>(1)->rotation, std::numbers::pi);
}

/// Test that the correct force is calculated when multiple behaviours are selected.
TEST_F(SteeringMovementSystemFixture, TestSteeringMovementSystemUpdateMultipleBehaviours) {
  create_steering_movement_component(
      {{SteeringMovementState::Footprint, {SteeringBehaviours::FollowPath, SteeringBehaviours::Seek}}});
  cpBodySetPosition(*registry.get_component<KinematicComponent>(1)->body, {.x = 300, .y = 300});
  registry.get_component<SteeringMovement>(1)->path_list = {{.x = 100, .y = 200}, {.x = -100, .y = 0}};
  get_steering_movement_system()->update(0);
  test_force_double(cpBodyGetForce(*registry.get_component<KinematicComponent>(1)->body), -162.2484370351122,
                    -116.94205693275299);
  ASSERT_DOUBLE_EQ(registry.get_component<KinematicComponent>(1)->rotation, -2.5170697673906659);
}

/// Test that the correct force is calculated when multiple states are initialised.
TEST_F(SteeringMovementSystemFixture, TestSteeringMovementSystemUpdateMultipleStates) {
  // Initialise the steering movement component with multiple states
  create_steering_movement_component({{SteeringMovementState::Target, {SteeringBehaviours::Pursue}},
                                      {SteeringMovementState::Default, {SteeringBehaviours::Seek}}});

  // Test the target state
  cpBodySetVelocity(*registry.get_component<KinematicComponent>(0)->body, {.x = -50, .y = 100});
  cpBodySetPosition(*registry.get_component<KinematicComponent>(1)->body, {.x = 100, .y = 100});
  get_steering_movement_system()->update(0);
  test_force_double(cpBodyGetForce(*registry.get_component<KinematicComponent>(1)->body), -193.02806555399468,
                    52.346594048540929);
  ASSERT_DOUBLE_EQ(registry.get_component<KinematicComponent>(1)->rotation, 2.8767753236840043);

  // Test the default state making sure to clear the force
  cpBodySetForce(*registry.get_component<KinematicComponent>(1)->body, cpvzero);
  cpBodySetPosition(*registry.get_component<KinematicComponent>(1)->body, {.x = 300, .y = 300});
  get_steering_movement_system()->update(0);
  test_force_double(cpBodyGetForce(*registry.get_component<KinematicComponent>(1)->body), -141.42135623730951,
                    -141.42135623730951);
  ASSERT_DOUBLE_EQ(registry.get_component<KinematicComponent>(1)->rotation, -2.3561944901923448);
}

/// Test that the correct force is not applied if the game object does not have the required components.
TEST_F(SteeringMovementSystemFixture, TestSteeringMovementSystemUpdateIncompleteComponents) {
  registry.create_game_object(GameObjectType::Player, cpvzero,
                              {std::make_shared<KinematicComponent>(), std::make_shared<MovementForce>(100, -1)});
  get_steering_movement_system()->update(0);
  ASSERT_EQ(cpBodyGetForce(*registry.get_component<KinematicComponent>(0)->body), cpvzero);
}

/// Test if the path list is updated if the position is within the view distance.
TEST_F(SteeringMovementSystemFixture, TestSteeringMovementSystemUpdatePathListWithinDistance) {
  create_steering_movement_component({});
  get_steering_movement_system()->update_path_list(0, {{.x = 100, .y = 100}, {.x = 300, .y = 300}});
  const std::vector<cpVect> expected_path_list{{.x = 100, .y = 100}, {.x = 300, .y = 300}};
  ASSERT_EQ(registry.get_component<SteeringMovement>(1)->path_list, expected_path_list);
}

/// Test if the path list is updated if the position is outside the view distance.
TEST_F(SteeringMovementSystemFixture, TestSteeringMovementSystemUpdatePathListOutsideDistance) {
  create_steering_movement_component({});
  get_steering_movement_system()->update_path_list(0, {{.x = 300, .y = 300}, {.x = 500, .y = 500}});
  ASSERT_EQ(registry.get_component<SteeringMovement>(1)->path_list, std::vector<cpVect>{});
}

/// Test if the path list is updated if the position is equal to the view distance.
TEST_F(SteeringMovementSystemFixture, TestSteeringMovementSystemUpdatePathListEqualDistance) {
  create_steering_movement_component({});
  get_steering_movement_system()->update_path_list(0, {{.x = 122.50966798868408, .y = 122.50966798868408}});
  ASSERT_EQ(registry.get_component<SteeringMovement>(1)->path_list,
            std::vector{cpv(122.50966798868408, 122.50966798868408)});
}

/// Test if the path list is updated if multiple footprints are within view distance.
TEST_F(SteeringMovementSystemFixture, TestSteeringMovementSystemUpdatePathListMultiplePoints) {
  create_steering_movement_component({});
  get_steering_movement_system()->update_path_list(
      0, {{.x = 100, .y = 100}, {.x = 300, .y = 300}, {.x = 50, .y = 100}, {.x = 500, .y = 500}});
  const std::vector<cpVect> expected_path_list{{.x = 50, .y = 100}, {.x = 500, .y = 500}};
  ASSERT_EQ(registry.get_component<SteeringMovement>(1)->path_list, expected_path_list);
}

/// Test if the path list is updated if the footprints list is empty.
TEST_F(SteeringMovementSystemFixture, TestSteeringMovementSystemUpdatePathListEmptyList) {
  create_steering_movement_component({});
  get_steering_movement_system()->update_path_list(0, {});
  ASSERT_EQ(registry.get_component<SteeringMovement>(1)->path_list, std::vector<cpVect>{});
}

/// Test if the path list is not updated if the target ID doesn't match.
TEST_F(SteeringMovementSystemFixture, TestSteeringMovementSystemUpdatePathListDifferentTargetId) {
  create_steering_movement_component({});
  registry.get_component<SteeringMovement>(1)->target_id = -1;
  get_steering_movement_system()->update_path_list(0, {{.x = 100, .y = 100}});
  ASSERT_EQ(registry.get_component<SteeringMovement>(1)->path_list, std::vector<cpVect>{});
}

/// Test if the path list is updated correctly if the Footprints component updates it.
TEST_F(SteeringMovementSystemFixture, TestSteeringMovementSystemUpdatePathListFootprintUpdate) {
  create_steering_movement_component({});
  registry.get_system<FootprintSystem>()->update(0.5);
  ASSERT_EQ(registry.get_component<SteeringMovement>(1)->path_list, std::vector{cpv(32, 32)});
}
