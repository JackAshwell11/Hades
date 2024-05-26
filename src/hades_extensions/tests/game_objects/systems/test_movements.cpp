// External headers
#include <chipmunk/chipmunk_structs.h>

// Std headers
#include <numbers>

// Local headers
#include "game_objects/stats.hpp"
#include "game_objects/systems/movements.hpp"
#include "game_objects/systems/physics.hpp"
#include "macros.hpp"

// ----- FIXTURES ------------------------------
/// Implements the fixture for the FootprintSystem tests.
class FootprintSystemFixture : public testing::Test {
 protected:
  /// The registry that manages the game objects, components, and systems.
  Registry registry;

  /// Set up the fixture for the tests.
  void SetUp() override {
    registry.create_game_object(
        GameObjectType::Player, cpvzero,
        {std::make_shared<Footprints>(), std::make_shared<KinematicComponent>(std::vector<cpVect>{})});
    registry.add_system<FootprintSystem>();
    registry.add_system<SteeringMovementSystem>();

    // Set the position of the game object to (0, 0) since grid_pos_to_pixel
    // sets the position to (32, 32)
    registry.get_component<KinematicComponent>(0)->body->p = cpvzero;
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
                                 std::make_shared<KinematicComponent>(std::vector<cpVect>{})});
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
    registry.create_game_object(
        GameObjectType::Player, cpvzero,
        {std::make_shared<Footprints>(), std::make_shared<KinematicComponent>(std::vector<cpVect>{})});
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
        {std::make_shared<MovementForce>(100, -1), std::make_shared<SteeringMovement>(steering_behaviours),
         std::make_shared<KinematicComponent>(std::vector<cpVect>{})})};
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
/// @details This function is needed because ASSERT_DOUBLE_EQ does not work great with cpVect which uses floats
/// internally.
/// @param actual - The actual vector.
/// @param expected_x - The expected x value.
/// @param expected_y - The expected y value.
inline void test_force_double(const cpVect &actual, const double expected_x, const double expected_y) {
  ASSERT_DOUBLE_EQ(actual.x, expected_x);
  ASSERT_DOUBLE_EQ(actual.y, expected_y);
}

// ----- TESTS ----------------------------------
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
  footprints->footprints = {{1, 1}, {2, 2}, {3, 3}};
  get_footprint_system()->update(0.5);
  const std::deque<cpVect> expected_footprints{{1, 1}, {2, 2}, {3, 3}, {0, 0}};
  ASSERT_EQ(footprints->footprints, expected_footprints);
}

/// Test that the footprint system creates a footprint and removes the oldest one.
TEST_F(FootprintSystemFixture, TestFootprintSystemUpdateLargeDeltaTimeFullList) {
  const auto footprints{registry.get_component<Footprints>(0)};
  footprints->footprints = {{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}, {7, 7}, {8, 8}, {9, 9}, {10, 10}};
  get_footprint_system()->update(0.5);
  const std::deque<cpVect> expected_footprints{{2, 2}, {3, 3}, {4, 4}, {5, 5},   {6, 6},
                                               {7, 7}, {8, 8}, {9, 9}, {10, 10}, {0, 0}};
  ASSERT_EQ(footprints->footprints, expected_footprints);
}

/// Test that the footprint system is updated correctly multiple times.
TEST_F(FootprintSystemFixture, TestFootprintSystemUpdateMultipleUpdates) {
  const auto footprints{registry.get_component<Footprints>(0)};
  get_footprint_system()->update(0.6);
  ASSERT_EQ(footprints->footprints, std::deque{cpvzero});
  ASSERT_EQ(footprints->time_since_last_footprint, 0);
  registry.get_component<KinematicComponent>(0)->body->p = {1, 1};
  get_footprint_system()->update(0.7);
  const std::deque<cpVect> expected_footprints{{0, 0}, {1, 1}};
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
  ASSERT_EQ(registry.get_component<KinematicComponent>(0)->body->f, cpvzero);
}

/// Test that the new force is updated correctly if the north key is pressed.
TEST_F(KeyboardMovementSystemFixture, TestKeyboardMovementSystemUpdateNorth) {
  registry.get_component<KeyboardMovement>(0)->moving_north = true;
  get_keyboard_movement_system()->update(0);
  ASSERT_EQ(registry.get_component<KinematicComponent>(0)->body->f, cpv(0, 100));
}

/// Test that the new force is updated correctly if the south key is pressed.
TEST_F(KeyboardMovementSystemFixture, TestKeyboardMovementSystemUpdateSouth) {
  registry.get_component<KeyboardMovement>(0)->moving_south = true;
  get_keyboard_movement_system()->update(0);
  ASSERT_EQ(registry.get_component<KinematicComponent>(0)->body->f, cpv(0, -100));
}

/// Test that the new force is updated correctly if the east key is pressed.
TEST_F(KeyboardMovementSystemFixture, TestKeyboardMovementSystemUpdateEast) {
  registry.get_component<KeyboardMovement>(0)->moving_east = true;
  get_keyboard_movement_system()->update(0);
  ASSERT_EQ(registry.get_component<KinematicComponent>(0)->body->f, cpv(100, 0));
}

/// Test that the new force is updated correctly if the west key is pressed.
TEST_F(KeyboardMovementSystemFixture, TestKeyboardMovementSystemUpdateWest) {
  registry.get_component<KeyboardMovement>(0)->moving_west = true;
  get_keyboard_movement_system()->update(0);
  ASSERT_EQ(registry.get_component<KinematicComponent>(0)->body->f, cpv(-100, 0));
}

/// Test that the correct force is calculated if the east and west keys are pressed.
TEST_F(KeyboardMovementSystemFixture, TestKeyboardMovementSystemUpdateEastWest) {
  const auto keyboard_movement{registry.get_component<KeyboardMovement>(0)};
  keyboard_movement->moving_east = true;
  keyboard_movement->moving_west = true;
  get_keyboard_movement_system()->update(0);
  ASSERT_EQ(registry.get_component<KinematicComponent>(0)->body->f, cpvzero);
}

/// Test that the correct force is calculated if the north and south are pressed.
TEST_F(KeyboardMovementSystemFixture, TestKeyboardMovementSystemUpdateNorthSouth) {
  const auto keyboard_movement{registry.get_component<KeyboardMovement>(0)};
  keyboard_movement->moving_east = true;
  keyboard_movement->moving_west = true;
  get_keyboard_movement_system()->update(0);
  ASSERT_EQ(registry.get_component<KinematicComponent>(0)->body->f, cpvzero);
}

/// Test that the correct force is calculated if north and west keys are pressed.
TEST_F(KeyboardMovementSystemFixture, TestKeyboardMovementSystemUpdateNorthWest) {
  const auto keyboard_movement{registry.get_component<KeyboardMovement>(0)};
  keyboard_movement->moving_north = true;
  keyboard_movement->moving_west = true;
  get_keyboard_movement_system()->update(0);
  ASSERT_EQ(registry.get_component<KinematicComponent>(0)->body->f, cpv(-70.710678118654741, 70.710678118654741));
}

/// Test that the correct force is calculated if south and east keys are pressed.
TEST_F(KeyboardMovementSystemFixture, TestKeyboardMovementSystemUpdateSouthEast) {
  const auto keyboard_movement{registry.get_component<KeyboardMovement>(0)};
  keyboard_movement->moving_south = true;
  keyboard_movement->moving_east = true;
  get_keyboard_movement_system()->update(0);
  ASSERT_EQ(registry.get_component<KinematicComponent>(0)->body->f, cpv(70.710678118654741, -70.710678118654741));
}

/// Test that the correct force is not applied if the game object does not have the required components.
TEST_F(KeyboardMovementSystemFixture, TestKeyboardMovementSystemUpdateIncompleteComponents) {
  registry.create_game_object(
      GameObjectType::Player, cpvzero,
      {std::make_shared<KinematicComponent>(std::vector<cpVect>{}), std::make_shared<MovementForce>(100, -1)});
  get_keyboard_movement_system()->update(0);
  ASSERT_EQ(registry.get_component<KinematicComponent>(0)->body->f, cpvzero);
}

/// Test that a game object is not updated if the target ID is not set
TEST_F(SteeringMovementSystemFixture, TestSteeringMovementSystemUpdateNoTargetId) {
  create_steering_movement_component({});
  registry.get_component<SteeringMovement>(1)->target_id = -1;
  get_steering_movement_system()->update(0);
  ASSERT_EQ(registry.get_component<KinematicComponent>(1)->body->f, cpvzero);
}

/// Test that the state is correctly changed to the default state.
TEST_F(SteeringMovementSystemFixture, TestSteeringMovementSystemUpdateOutsideDistanceEmptyPathList) {
  create_steering_movement_component({});
  registry.get_component<KinematicComponent>(1)->body->p = {500, 500};
  get_steering_movement_system()->update(0);
  ASSERT_EQ(registry.get_component<SteeringMovement>(1)->movement_state, SteeringMovementState::Default);
}

/// Test that the state is correctly changed to the footprint state.
TEST_F(SteeringMovementSystemFixture, TestSteeringMovementSystemUpdateOutsideDistanceNonEmptyPathList) {
  create_steering_movement_component({});
  registry.get_component<KinematicComponent>(1)->body->p = {500, 500};
  const auto steering_movement{registry.get_component<SteeringMovement>(1)};
  steering_movement->path_list = {{300, 300}, {400, 400}};
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
  ASSERT_EQ(registry.get_component<KinematicComponent>(1)->body->f, cpvzero);
  ASSERT_DOUBLE_EQ(registry.get_component<KinematicComponent>(1)->rotation, 0);
}

/// Test that the correct force is calculated for the arrive behaviour.
TEST_F(SteeringMovementSystemFixture, TestSteeringMovementSystemUpdateArrive) {
  create_steering_movement_component({{SteeringMovementState::Target, {SteeringBehaviours::Arrive}}});
  registry.get_component<KinematicComponent>(0)->body->p = cpvzero;
  registry.get_component<KinematicComponent>(1)->body->p = {100, 100};
  get_steering_movement_system()->update(0);
  test_force_double(registry.get_component<KinematicComponent>(1)->body->f, -70.71067811865476, -70.71067811865476);
  ASSERT_DOUBLE_EQ(registry.get_component<KinematicComponent>(1)->rotation, -2.3561944901923448);
}

/// Test that the correct force is calculated for the evade behaviour.
TEST_F(SteeringMovementSystemFixture, TestSteeringMovementSystemUpdateEvade) {
  create_steering_movement_component({{SteeringMovementState::Target, {SteeringBehaviours::Evade}}});
  registry.get_component<KinematicComponent>(0)->body->p = {100, 100};
  registry.get_component<KinematicComponent>(0)->body->v = {-50, 0};
  get_steering_movement_system()->update(0);
  test_force_double(registry.get_component<KinematicComponent>(1)->body->f, -54.28888213891886, -83.98045770360257);
  ASSERT_DOUBLE_EQ(registry.get_component<KinematicComponent>(1)->rotation, -2.1446695001689107);
}

/// Test that the correct force is calculated for the flee behaviour.
TEST_F(SteeringMovementSystemFixture, TestSteeringMovementSystemUpdateFlee) {
  create_steering_movement_component({{SteeringMovementState::Target, {SteeringBehaviours::Flee}}});
  registry.get_component<KinematicComponent>(0)->body->p = {50, 50};
  registry.get_component<KinematicComponent>(1)->body->p = {100, 100};
  get_steering_movement_system()->update(0);
  test_force_double(registry.get_component<KinematicComponent>(1)->body->f, 70.71067811865475, 70.7106781186547);
  test_force_double(registry.get_component<KinematicComponent>(1)->body->f, 70.71067811865476, 70.71067811865476);
  ASSERT_DOUBLE_EQ(registry.get_component<KinematicComponent>(1)->rotation, 0.7853981633974483);
}

/// Test that the correct force is calculated for the follow path behaviour.
TEST_F(SteeringMovementSystemFixture, TestSteeringMovementSystemUpdateFollowPath) {
  create_steering_movement_component({{SteeringMovementState::Footprint, {SteeringBehaviours::FollowPath}}});
  registry.get_component<KinematicComponent>(1)->body->p = {200, 200};
  registry.get_component<SteeringMovement>(1)->path_list = {{350, 350}, {500, 500}};
  get_steering_movement_system()->update(0);
  test_force_double(registry.get_component<KinematicComponent>(1)->body->f, 70.71067811865475, 70.71067811865475);
  ASSERT_DOUBLE_EQ(registry.get_component<KinematicComponent>(1)->rotation, 0.7853981633974483);
}

/// Test that the correct force is calculated for the obstacle avoidance behaviour.
TEST_F(SteeringMovementSystemFixture, TestSteeringMovementSystemUpdateObstacleAvoidance) {
  create_steering_movement_component({{SteeringMovementState::Target, {SteeringBehaviours::ObstacleAvoidance}}});
  registry.get_component<KinematicComponent>(1)->body->p = {32, 96};
  registry.get_component<KinematicComponent>(1)->body->v = {100, 100};
  registry.create_game_object(GameObjectType::Wall, {1, 2}, {std::make_shared<KinematicComponent>(true)});
  get_steering_movement_system()->update(0);
  test_force_double(registry.get_component<KinematicComponent>(1)->body->f, -70.710678118654755, -70.710678118654755);
  ASSERT_DOUBLE_EQ(registry.get_component<KinematicComponent>(1)->rotation, -2.3561944901923448);
}

/// Test that the correct force is calculated for the pursue behaviour.
TEST_F(SteeringMovementSystemFixture, TestSteeringMovementSystemUpdatePursue) {
  create_steering_movement_component({{SteeringMovementState::Target, {SteeringBehaviours::Pursue}}});
  registry.get_component<KinematicComponent>(0)->body->p = {100, 100};
  registry.get_component<KinematicComponent>(0)->body->v = {-50, 0};
  get_steering_movement_system()->update(0);
  test_force_double(registry.get_component<KinematicComponent>(1)->body->f, 54.28888213891886, 83.98045770360257);
  ASSERT_DOUBLE_EQ(registry.get_component<KinematicComponent>(1)->rotation, 0.99692315342088256);
}

/// Test that the correct force is calculated for the seek behaviour.
TEST_F(SteeringMovementSystemFixture, TestSteeringMovementSystemUpdateSeek) {
  create_steering_movement_component({{SteeringMovementState::Target, {SteeringBehaviours::Seek}}});
  registry.get_component<KinematicComponent>(0)->body->p = {50, 50};
  registry.get_component<KinematicComponent>(1)->body->p = {100, 100};
  get_steering_movement_system()->update(0);
  test_force_double(registry.get_component<KinematicComponent>(1)->body->f, -70.71067811865475, -70.7106781186547);
  ASSERT_DOUBLE_EQ(registry.get_component<KinematicComponent>(1)->rotation, -2.3561944901923448);
}

/// Test that the correct force is calculated for the wander behaviour.
TEST_F(SteeringMovementSystemFixture, TestSteeringMovementSystemUpdateWander) {
  create_steering_movement_component({{SteeringMovementState::Target, {SteeringBehaviours::Wander}}});
  registry.get_component<KinematicComponent>(1)->body->v = {100, -100};
  const auto before_force{registry.get_component<KinematicComponent>(1)->body->f};
  get_steering_movement_system()->update(0);
  const auto after_force{registry.get_component<KinematicComponent>(1)->body->f};
  ASSERT_EQ(round(cpvlength(after_force)), 100);
  ASSERT_NE(before_force, after_force);
  ASSERT_GE(registry.get_component<KinematicComponent>(1)->rotation, -std::numbers::pi);
  ASSERT_LE(registry.get_component<KinematicComponent>(1)->rotation, std::numbers::pi);
}

/// Test that the correct force is calculated when multiple behaviours are selected.
TEST_F(SteeringMovementSystemFixture, TestSteeringMovementSystemUpdateMultipleBehaviours) {
  create_steering_movement_component(
      {{SteeringMovementState::Footprint, {SteeringBehaviours::FollowPath, SteeringBehaviours::Seek}}});
  registry.get_component<KinematicComponent>(1)->body->p = {300, 300};
  registry.get_component<SteeringMovement>(1)->path_list = {{100, 200}, {-100, 0}};
  get_steering_movement_system()->update(0);
  test_force_double(registry.get_component<KinematicComponent>(1)->body->f, -81.12421851755609, -58.47102846637651);
  ASSERT_DOUBLE_EQ(registry.get_component<KinematicComponent>(1)->rotation, -2.5170697673906659);
}

/// Test that the correct force is calculated when multiple states are initialised.
TEST_F(SteeringMovementSystemFixture, TestSteeringMovementSystemUpdateMultipleStates) {
  // Initialise the steering movement component with multiple states
  create_steering_movement_component({{SteeringMovementState::Target, {SteeringBehaviours::Pursue}},
                                      {SteeringMovementState::Default, {SteeringBehaviours::Seek}}});

  // Test the target state
  registry.get_component<KinematicComponent>(0)->body->v = {-50, 100};
  registry.get_component<KinematicComponent>(1)->body->p = {100, 100};
  get_steering_movement_system()->update(0);
  test_force_double(registry.get_component<KinematicComponent>(1)->body->f, -97.73793955511094, -21.14935392681019);
  ASSERT_DOUBLE_EQ(registry.get_component<KinematicComponent>(1)->rotation, -2.9284898398506929);

  // Test the default state making sure to clear the force
  registry.get_component<KinematicComponent>(1)->body->f = cpvzero;
  registry.get_component<KinematicComponent>(1)->body->p = {300, 300};
  get_steering_movement_system()->update(0);
  test_force_double(registry.get_component<KinematicComponent>(1)->body->f, -70.71067811865476, -70.71067811865476);
  ASSERT_DOUBLE_EQ(registry.get_component<KinematicComponent>(1)->rotation, -2.3561944901923448);
}

/// Test that the correct force is not applied if the game object does not have the required components.
TEST_F(SteeringMovementSystemFixture, TestSteeringMovementSystemUpdateIncompleteComponents) {
  registry.create_game_object(
      GameObjectType::Player, cpvzero,
      {std::make_shared<KinematicComponent>(std::vector<cpVect>{}), std::make_shared<MovementForce>(100, -1)});
  get_steering_movement_system()->update(0);
  ASSERT_EQ(registry.get_component<KinematicComponent>(0)->body->f, cpvzero);
}

/// Test if the path list is updated if the position is within the view distance.
TEST_F(SteeringMovementSystemFixture, TestSteeringMovementSystemUpdatePathListWithinDistance) {
  create_steering_movement_component({});
  get_steering_movement_system()->update_path_list(0, {{100, 100}, {300, 300}});
  const std::vector<cpVect> expected_path_list{{100, 100}, {300, 300}};
  ASSERT_EQ(registry.get_component<SteeringMovement>(1)->path_list, expected_path_list);
}

/// Test if the path list is updated if the position is outside the view distance.
TEST_F(SteeringMovementSystemFixture, TestSteeringMovementSystemUpdatePathListOutsideDistance) {
  create_steering_movement_component({});
  get_steering_movement_system()->update_path_list(0, {{300, 300}, {500, 500}});
  ASSERT_EQ(registry.get_component<SteeringMovement>(1)->path_list, std::vector<cpVect>{});
}

/// Test if the path list is updated if the position is equal to the view distance.
TEST_F(SteeringMovementSystemFixture, TestSteeringMovementSystemUpdatePathListEqualDistance) {
  create_steering_movement_component({});
  get_steering_movement_system()->update_path_list(0, {{135.764501987, 135.764501987}});
  ASSERT_EQ(registry.get_component<SteeringMovement>(1)->path_list, std::vector{cpv(135.764501987, 135.764501987)});
}

/// Test if the path list is updated if multiple footprints are within view distance.
TEST_F(SteeringMovementSystemFixture, TestSteeringMovementSystemUpdatePathListMultiplePoints) {
  create_steering_movement_component({});
  get_steering_movement_system()->update_path_list(0, {{100, 100}, {300, 300}, {50, 100}, {500, 500}});
  const std::vector<cpVect> expected_path_list{{50, 100}, {500, 500}};
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
  get_steering_movement_system()->update_path_list(0, {{100, 100}});
  ASSERT_EQ(registry.get_component<SteeringMovement>(1)->path_list, std::vector<cpVect>{});
}

/// Test if the path list is updated correctly if the Footprints component updates it.
TEST_F(SteeringMovementSystemFixture, TestSteeringMovementSystemUpdatePathListFootprintUpdate) {
  create_steering_movement_component({});
  registry.get_system<FootprintSystem>()->update(0.5);
  ASSERT_EQ(registry.get_component<SteeringMovement>(1)->path_list, std::vector{cpv(32, 32)});
}
