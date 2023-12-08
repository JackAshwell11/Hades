// Local headers
#include "game_objects/stats.hpp"
#include "game_objects/systems/movements.hpp"
#include "macros.hpp"

// ----- FIXTURES ------------------------------
/// Implements the fixture for the FootprintSystem tests.
class FootprintSystemFixture : public testing::Test {
 protected:
  /// The registry that manages the game objects, components, and systems.
  Registry registry{};

  /// Set up the fixture for the tests.
  void SetUp() override {
    registry.create_game_object({std::make_shared<Footprints>()}, true);
    registry.add_system<FootprintSystem>();
    registry.add_system<SteeringMovementSystem>();
  }

  /// Get the footprint system from the registry.
  ///
  /// @return The footprint system.
  [[nodiscard]] auto get_footprint_system() const -> std::shared_ptr<FootprintSystem> {
    return registry.get_system<FootprintSystem>();
  }
};

/// Implements the fixture for the KeyboardMovementSystem tests.
class KeyboardMovementFixture : public testing::Test {
 protected:
  /// The registry that manages the game objects, components, and systems.
  Registry registry{};

  /// Set up the fixture for the tests.
  void SetUp() override {
    registry.create_game_object({std::make_shared<MovementForce>(100, -1), std::make_shared<KeyboardMovement>()}, true);
    registry.add_system<KeyboardMovementSystem>();
  }

  /// Get the keyboard movement system from the registry.
  ///
  /// @return The keyboard movement system.
  [[nodiscard]] auto get_keyboard_movement_system() const -> std::shared_ptr<KeyboardMovementSystem> {
    return registry.get_system<KeyboardMovementSystem>();
  }
};

/// Implements the fixture for the SteeringMovementSystem tests.
class SteeringMovementFixture : public testing::Test {
 protected:
  /// The registry that manages the game objects, components, and systems.
  Registry registry{};

  /// Set up the fixture for the tests.
  void SetUp() override {
    // Create the target game object
    registry.create_game_object({std::make_shared<Footprints>()}, true);

    // Create the game object to follow the target and add the required systems
    create_steering_movement_component({});
    registry.add_system<FootprintSystem>();
    registry.add_system<SteeringMovementSystem>();
  }

  /// Create a steering movement component.
  ///
  /// @param steering_behaviours - The steering behaviours to initialise the component with.
  /// @return The game object ID of the created game object.
  auto create_steering_movement_component(
      const std::unordered_map<SteeringMovementState, std::vector<SteeringBehaviours>> &steering_behaviours) -> int {
    const int game_object_id{registry.create_game_object(
        {std::make_shared<MovementForce>(100, -1), std::make_shared<SteeringMovement>(steering_behaviours)}, true)};
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

// ----- FUNCTIONS -----------------------------
/// Test if two vectors are equal within a small margin of error.
///
/// @details This function is needed because ASSERT_EQ does not work great with Vec2d which uses doubles internally.
/// @param actual - The actual vector.
/// @param expected_x - The expected x value.
/// @param expected_y - The expected y value.
void test_force_double(const Vec2d &actual, const double expected_x, const double expected_y) {
  ASSERT_DOUBLE_EQ(actual.x, expected_x);
  ASSERT_DOUBLE_EQ(actual.y, expected_y);
}

// ----- TESTS ----------------------------------
/// Test that the footprint systems is updated with a small delta time.
TEST_F(FootprintSystemFixture, TestFootprintSystemUpdateSmallDeltaTime) {
  get_footprint_system()->update(0.1);
  auto footprints{registry.get_component<Footprints>(0)};
  ASSERT_EQ(footprints->footprints, std::deque<Vec2d>{});
  ASSERT_EQ(footprints->time_since_last_footprint, 0.1);
}

/// Test that the footprint system creates a footprint in an empty list.
TEST_F(FootprintSystemFixture, TestFootprintSystemUpdateLargeDeltaTimeEmptyList) {
  get_footprint_system()->update(1);
  auto footprints{registry.get_component<Footprints>(0)};
  ASSERT_EQ(footprints->footprints, std::deque<Vec2d>{Vec2d(0, 0)});
  ASSERT_EQ(footprints->time_since_last_footprint, 0);
}

/// Test that the footprint system creates a footprint in a non-empty list.
TEST_F(FootprintSystemFixture, TestFootprintSystemUpdateLargeDeltaTimeNonEmptyList) {
  auto footprints{registry.get_component<Footprints>(0)};
  footprints->footprints = {{1, 1}, {2, 2}, {3, 3}};
  get_footprint_system()->update(0.5);
  const std::deque<Vec2d> expected_footprints{{1, 1}, {2, 2}, {3, 3}, {0, 0}};
  ASSERT_EQ(footprints->footprints, expected_footprints);
}

/// Test that the footprint system creates a footprint and removes the oldest one.
TEST_F(FootprintSystemFixture, TestFootprintSystemUpdateLargeDeltaTimeFullList) {
  auto footprints{registry.get_component<Footprints>(0)};
  footprints->footprints = {{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}, {7, 7}, {8, 8}, {9, 9}, {10, 10}};
  get_footprint_system()->update(0.5);
  const std::deque<Vec2d> expected_footprints{{2, 2}, {3, 3}, {4, 4}, {5, 5},   {6, 6},
                                              {7, 7}, {8, 8}, {9, 9}, {10, 10}, {0, 0}};
  ASSERT_EQ(footprints->footprints, expected_footprints);
}

/// Test that the footprint system is updated correctly multiple times.
TEST_F(FootprintSystemFixture, TestFootprintSystemUpdateMultipleUpdates) {
  auto footprints{registry.get_component<Footprints>(0)};
  get_footprint_system()->update(0.6);
  ASSERT_EQ(footprints->footprints, std::deque<Vec2d>{Vec2d(0, 0)});
  ASSERT_EQ(footprints->time_since_last_footprint, 0);
  registry.get_kinematic_object(0)->position = {1, 1};
  get_footprint_system()->update(0.7);
  const std::deque<Vec2d> expected_footprints{{0, 0}, {1, 1}};
  ASSERT_EQ(footprints->footprints, expected_footprints);
  ASSERT_EQ(footprints->time_since_last_footprint, 0);
}

/// Test if the correct force is calculated if no keys are pressed.
TEST_F(KeyboardMovementFixture, TestKeyboardMovementSystemCalculateForceNone) {
  ASSERT_EQ(get_keyboard_movement_system()->calculate_force(0), Vec2d(0, 0));
}

/// Test if the correct force is calculated for a move north.
TEST_F(KeyboardMovementFixture, TestKeyboardMovementSystemCalculateForceNorth) {
  registry.get_component<KeyboardMovement>(0)->moving_north = true;
  ASSERT_EQ(get_keyboard_movement_system()->calculate_force(0), Vec2d(0, 100));
}

/// Test if the correct force is calculated for a move south.
TEST_F(KeyboardMovementFixture, TestKeyboardMovementSystemCalculateForceSouth) {
  registry.get_component<KeyboardMovement>(0)->moving_south = true;
  ASSERT_EQ(get_keyboard_movement_system()->calculate_force(0), Vec2d(0, -100));
}

/// Test if the correct force is calculated for a move east.
TEST_F(KeyboardMovementFixture, TestKeyboardMovementSystemCalculateForceEast) {
  registry.get_component<KeyboardMovement>(0)->moving_east = true;
  ASSERT_EQ(get_keyboard_movement_system()->calculate_force(0), Vec2d(100, 0));
}

/// Test if the correct force is calculated for a move west.
TEST_F(KeyboardMovementFixture, TestKeyboardMovementSystemCalculateForceWest) {
  registry.get_component<KeyboardMovement>(0)->moving_west = true;
  ASSERT_EQ(get_keyboard_movement_system()->calculate_force(0), Vec2d(-100, 0));
}

/// Test if the correct force is calculated if east and west are pressed.
TEST_F(KeyboardMovementFixture, TestKeyboardMovementSystemCalculateForceEastWest) {
  auto keyboard_movement{registry.get_component<KeyboardMovement>(0)};
  keyboard_movement->moving_east = true;
  keyboard_movement->moving_west = true;
  ASSERT_EQ(get_keyboard_movement_system()->calculate_force(0), Vec2d(0, 0));
}

/// Test if the correct force is calculated if north and south are pressed.
TEST_F(KeyboardMovementFixture, TestKeyboardMovementSystemCalculateForceNorthSouth) {
  auto keyboard_movement{registry.get_component<KeyboardMovement>(0)};
  keyboard_movement->moving_east = true;
  keyboard_movement->moving_west = true;
  ASSERT_EQ(get_keyboard_movement_system()->calculate_force(0), Vec2d(0, 0));
}

/// Test if the correct force is calculated if north and west are pressed.
TEST_F(KeyboardMovementFixture, TestKeyboardMovementSystemCalculateForceNorthWest) {
  auto keyboard_movement{registry.get_component<KeyboardMovement>(0)};
  keyboard_movement->moving_north = true;
  keyboard_movement->moving_west = true;
  ASSERT_EQ(get_keyboard_movement_system()->calculate_force(0), Vec2d(-100, 100));
}

/// Test if the correct force is calculated if south and east are pressed.
TEST_F(KeyboardMovementFixture, TestKeyboardMovementSystemCalculateForceSouthEast) {
  auto keyboard_movement{registry.get_component<KeyboardMovement>(0)};
  keyboard_movement->moving_south = true;
  keyboard_movement->moving_east = true;
  ASSERT_EQ(get_keyboard_movement_system()->calculate_force(0), Vec2d(100, -100));
}

/// Test that an exception is thrown if an invalid game object ID is provided.
TEST_F(KeyboardMovementFixture, TestKeyboardMovementSystemCalculateForceInvalidGameObjectId){ASSERT_THROW_MESSAGE(
    (get_keyboard_movement_system()->calculate_force(-1)), RegistryError,
    "The game object `-1` is not registered with the registry or does not have the required component.")}

/// Test if the state is correctly changed to the default state.
TEST_F(SteeringMovementFixture, TestSteeringMovementSystemCalculateForceOutsideDistanceEmptyPathList) {
  registry.get_kinematic_object(1)->position = {500, 500};
  static_cast<void>(get_steering_movement_system()->calculate_force(1));
  ASSERT_EQ(registry.get_component<SteeringMovement>(1)->movement_state, SteeringMovementState::Default);
}

/// Test if the state is correctly changed to the footprint state.
TEST_F(SteeringMovementFixture, TestSteeringMovementSystemCalculateForceOutsideDistanceNonEmptyPathList) {
  registry.get_kinematic_object(1)->position = {500, 500};
  auto steering_movement{registry.get_component<SteeringMovement>(1)};
  steering_movement->path_list = {{300, 300}, {400, 400}};
  static_cast<void>(get_steering_movement_system()->calculate_force(1));
  ASSERT_EQ(steering_movement->movement_state, SteeringMovementState::Footprint);
}

/// Test if the state is correctly changed to the target stat.
TEST_F(SteeringMovementFixture, TestSteeringMovementSystemCalculateForceWithinDistance) {
  static_cast<void>(get_steering_movement_system()->calculate_force(1));
  ASSERT_EQ(registry.get_component<SteeringMovement>(1)->movement_state, SteeringMovementState::Target);
}

/// Test if a zero force is calculated if the state is missing.
TEST_F(SteeringMovementFixture, TestSteeringMovementSystemCalculateForceMissingState) {
  ASSERT_EQ(get_steering_movement_system()->calculate_force(1), Vec2d(0, 0));
}

/// Test if the correct force is calculated for the arrive behaviour.
TEST_F(SteeringMovementFixture, TestSteeringMovementSystemCalculateForceArrive) {
  create_steering_movement_component({{SteeringMovementState::Target, {SteeringBehaviours::Arrive}}});
  registry.get_kinematic_object(0)->position = {0, 0};
  registry.get_kinematic_object(2)->position = {100, 100};
  test_force_double(get_steering_movement_system()->calculate_force(2), -70.71067811865476, -70.71067811865476);
}

/// Test if the correct force is calculated for the evade behaviour.
TEST_F(SteeringMovementFixture, TestSteeringMovementSystemCalculateForceEvade) {
  create_steering_movement_component({{SteeringMovementState::Target, {SteeringBehaviours::Evade}}});
  registry.get_kinematic_object(0)->position = {100, 100};
  registry.get_kinematic_object(0)->velocity = {-50, 0};
  test_force_double(get_steering_movement_system()->calculate_force(2), -54.28888213891886, -83.98045770360257);
}

/// Test if the correct force is calculated for the flee behaviour.
TEST_F(SteeringMovementFixture, TestSteeringMovementSystemCalculateForceFlee) {
  create_steering_movement_component({{SteeringMovementState::Target, {SteeringBehaviours::Flee}}});
  registry.get_kinematic_object(0)->position = {50, 50};
  registry.get_kinematic_object(2)->position = {100, 100};
  test_force_double(get_steering_movement_system()->calculate_force(2), 70.71067811865476, 70.71067811865476);
}

/// Test if the correct force is calculated for the follow path behaviour.
TEST_F(SteeringMovementFixture, TestSteeringMovementSystemCalculateForceFollowPath) {
  create_steering_movement_component({{SteeringMovementState::Footprint, {SteeringBehaviours::FollowPath}}});
  registry.get_kinematic_object(2)->position = {200, 200};
  registry.get_component<SteeringMovement>(2)->path_list = {{350, 350}, {500, 500}};
  test_force_double(get_steering_movement_system()->calculate_force(2), 70.71067811865475, 70.71067811865475);
}

/// Test if the correct force is calculated for the obstacle avoidance behaviour.
TEST_F(SteeringMovementFixture, TestSteeringMovementSystemCalculateForceObstacleAvoidance) {
  create_steering_movement_component({{SteeringMovementState::Target, {SteeringBehaviours::ObstacleAvoidance}}});
  registry.get_kinematic_object(2)->position = {100, 100};
  registry.get_kinematic_object(2)->velocity = {100, 100};
  registry.add_wall({1, 2});
  test_force_double(get_steering_movement_system()->calculate_force(2), 25.881904510252056, -96.59258262890683);
}

/// Test if the correct force is calculated for the pursue behaviour.
TEST_F(SteeringMovementFixture, TestSteeringMovementSystemCalculateForcePursue) {
  create_steering_movement_component({{SteeringMovementState::Target, {SteeringBehaviours::Pursue}}});
  registry.get_kinematic_object(0)->position = {100, 100};
  registry.get_kinematic_object(0)->velocity = {-50, 0};
  test_force_double(get_steering_movement_system()->calculate_force(2), 54.28888213891886, 83.98045770360257);
}

/// Test if the correct force is calculated for the seek behaviour.
TEST_F(SteeringMovementFixture, TestSteeringMovementSystemCalculateForceSeek) {
  create_steering_movement_component({{SteeringMovementState::Target, {SteeringBehaviours::Seek}}});
  registry.get_kinematic_object(0)->position = {50, 50};
  registry.get_kinematic_object(2)->position = {100, 100};
  test_force_double(get_steering_movement_system()->calculate_force(2), -70.71067811865475, -70.7106781186547);
}

/// Test if the correct force is calculated for the wander behaviour.
TEST_F(SteeringMovementFixture, TestSteeringMovementSystemCalculateForceWander) {
  create_steering_movement_component({{SteeringMovementState::Target, {SteeringBehaviours::Wander}}});
  registry.get_kinematic_object(2)->velocity = {100, -100};
  const Vec2d steering_force{get_steering_movement_system()->calculate_force(2)};
  ASSERT_EQ(round(steering_force.magnitude()), 100);
  ASSERT_NE(steering_force, get_steering_movement_system()->calculate_force(2));
}

/// Test if the correct force is calculated when multiple behaviours are selected.
TEST_F(SteeringMovementFixture, TestSteeringMovementSystemCalculateForceMultipleBehaviours) {
  create_steering_movement_component(
      {{SteeringMovementState::Footprint, {SteeringBehaviours::FollowPath, SteeringBehaviours::Seek}}});
  registry.get_kinematic_object(2)->position = {300, 300};
  registry.get_component<SteeringMovement>(2)->path_list = {{100, 200}, {-100, 0}};
  test_force_double(get_steering_movement_system()->calculate_force(2), -81.12421851755609, -58.47102846637651);
}

/// Test if the correct force is calculated when multiple states are initialised.
TEST_F(SteeringMovementFixture, TestSteeringMovementSystemCalculateForceMultipleStates) {
  // Initialise the steering movement component with multiple states
  create_steering_movement_component({{SteeringMovementState::Target, {SteeringBehaviours::Pursue}},
                                      {SteeringMovementState::Default, {SteeringBehaviours::Seek}}});

  // Test the target state
  registry.get_kinematic_object(0)->velocity = {-50, 100};
  registry.get_kinematic_object(2)->position = {100, 100};
  test_force_double(get_steering_movement_system()->calculate_force(2), -97.73793955511094, -21.14935392681019);

  // Test the default state
  registry.get_kinematic_object(2)->position = {300, 300};
  test_force_double(get_steering_movement_system()->calculate_force(2), -70.71067811865476, -70.71067811865476);
}

/// Test that an exception is thrown if an invalid game object ID is provided.
TEST_F(SteeringMovementFixture, TestSteeringMovementSystemCalculateForceInvalidGameObjectId){ASSERT_THROW_MESSAGE(
    (get_steering_movement_system()->calculate_force(-1)), RegistryError,
    "The game object `-1` is not registered with the registry or does not have the required component.")}

/// Test if the path list is updated if the position is within the view distance.
TEST_F(SteeringMovementFixture, TestSteeringMovementSystemUpdatePathListWithinDistance) {
  get_steering_movement_system()->update_path_list(0, {{100, 100}, {300, 300}});
  const std::vector<Vec2d> expected_path_list{{100, 100}, {300, 300}};
  ASSERT_EQ(registry.get_component<SteeringMovement>(1)->path_list, expected_path_list);
}

/// Test if the path list is updated if the position is outside the view distance.
TEST_F(SteeringMovementFixture, TestSteeringMovementSystemUpdatePathListOutsideDistance) {
  get_steering_movement_system()->update_path_list(0, {{300, 300}, {500, 500}});
  ASSERT_EQ(registry.get_component<SteeringMovement>(1)->path_list, std::vector<Vec2d>{});
}

/// Test if the path list is updated if the position is equal to the view distance.
TEST_F(SteeringMovementFixture, TestSteeringMovementSystemUpdatePathListEqualDistance) {
  get_steering_movement_system()->update_path_list(0, {{135.764501987, 135.764501987}});
  ASSERT_EQ(registry.get_component<SteeringMovement>(1)->path_list,
            std::vector<Vec2d>{Vec2d(135.764501987, 135.764501987)});
}

/// Test if the path list is updated if multiple footprints are within view distance.
TEST_F(SteeringMovementFixture, TestSteeringMovementSystemUpdatePathListMultiplePoints) {
  get_steering_movement_system()->update_path_list(0, {{100, 100}, {300, 300}, {50, 100}, {500, 500}});
  const std::vector<Vec2d> expected_path_list{{50, 100}, {500, 500}};
  ASSERT_EQ(registry.get_component<SteeringMovement>(1)->path_list, expected_path_list);
}

/// Test if the path list is updated if the footprints list is empty.
TEST_F(SteeringMovementFixture, TestSteeringMovementSystemUpdatePathListEmptyList) {
  get_steering_movement_system()->update_path_list(0, {});
  ASSERT_EQ(registry.get_component<SteeringMovement>(1)->path_list, std::vector<Vec2d>{});
}

/// Test if the path list is not updated if the target ID doesn't match.
TEST_F(SteeringMovementFixture, TestSteeringMovementUpdatePathListDifferentTargetId) {
  registry.get_component<SteeringMovement>(1)->target_id = -1;
  get_steering_movement_system()->update_path_list(0, {{100, 100}});
  ASSERT_EQ(registry.get_component<SteeringMovement>(1)->path_list, std::vector<Vec2d>{});
}

/// Test if the path list is updated correctly if the Footprints component updates it.
TEST_F(SteeringMovementFixture, TestSteeringMovementSystemUpdatePathListFootprintUpdate) {
  registry.get_system<FootprintSystem>()->update(0.5);
  ASSERT_EQ(registry.get_component<SteeringMovement>(1)->path_list, std::vector<Vec2d>{Vec2d(0, 0)});
}
