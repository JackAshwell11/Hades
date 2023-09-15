// External includes
#include "gtest/gtest.h"

// Custom includes
#include "macros.hpp"
#include "game_objects/registry.hpp"

// ----- CLASSES ------------------------------
/// Represents a game object component useful for testing.
struct TestGameObjectComponentOne : public ComponentBase {};

/// Represents a game object component with data useful for testing.
struct TestGameObjectComponentTwo : public ComponentBase {
  /// A test list of integers.
  std::vector<int> test_list;

  /// Initialise the object.
  ///
  /// @param test_lst - The list to be used for testing.
  TestGameObjectComponentTwo(std::initializer_list<int> &test_lst) : test_list(test_lst) {}
};

/// Represents a test system useful for testing.
struct TestSystem : public SystemBase {
  /// Whether the system has been called or not.
  bool called = false;

  /// Initialise the system.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  explicit TestSystem(Registry &registry) : SystemBase(registry) {}

  /// Update the system.
  void update(double delta_time) final {
    called = true;
  }
};

// ----- FIXTURES ------------------------------
/// Implements the fixture for the game_objects/registry.hpp tests.
class RegistryFixture : public testing::Test {
 protected:
  /// The registry that manages the game objects, components, and systems.
  Registry registry{};

  /// Create a component or system unique pointer.
  ///
  /// @tparam T - The type of the component or system.
  /// @param list - The initializer list to pass to the constructor.
  /// @return A unique pointer to the component or system.
  template<typename T>
  static inline std::unique_ptr<T> create_object(std::initializer_list<int> list) {
    return std::make_unique<T>(list);
  }

  /// Create a component or system unique pointer.
  ///
  /// @tparam T - The type of the component or system.
  /// @param args - The arguments to pass to the constructor.
  /// @return A unique pointer to the component or system.
  template<typename T, typename ... Args>
  static inline std::unique_ptr<T> create_object() {
    return std::make_unique<T>();
  }
};

// ----- TESTS ------------------------------
/// Test that an exception is thrown if a component is not registered.
TEST_F(RegistryFixture, TestRegistryEmptyGameObject) {
  ASSERT_EQ(registry.create_game_object(false, {}), 0);
  ASSERT_THROW_MESSAGE(registry.get_component<TestGameObjectComponentOne>(0),
                       RegistryException,
                       "The game object `0` is not registered with the registry.")
  ASSERT_THROW_MESSAGE(registry.get_component<TestGameObjectComponentTwo>(0),
                       RegistryException,
                       "The game object `0` is not registered with the registry.")
  ASSERT_EQ(registry.find_components<TestGameObjectComponentOne>().size(), {});
  ASSERT_EQ(registry.find_components<TestGameObjectComponentTwo>().size(), {});
  ASSERT_EQ(registry.get_walls().size(), 0);
  ASSERT_THROW_MESSAGE(registry.get_kinematic_object(0),
                       RegistryException,
                       "The game object `0` is not registered with the registry.")
}

/// Test that multiple components are added to the registry correctly.
TEST_F(RegistryFixture, TestRegistryGameObjectComponents) {
  // Construct the components list
  std::vector<std::unique_ptr<ComponentBase>> components;
  components.emplace_back(create_object<TestGameObjectComponentOne>());
  components.emplace_back(create_object<TestGameObjectComponentTwo>({10}));

  // Test that creating the game object works correctly
  registry.create_game_object(false, std::move(components));
  ASSERT_TRUE(registry.get_component<TestGameObjectComponentOne>(0) != nullptr);
  ASSERT_TRUE(registry.get_component<TestGameObjectComponentTwo>(0) != nullptr);
  ASSERT_EQ(registry.find_components<TestGameObjectComponentOne>().size(), 1);
  ASSERT_EQ(registry.find_components<TestGameObjectComponentTwo>().size(), 1);
  auto multiple_result_one = registry.find_components<TestGameObjectComponentOne, TestGameObjectComponentTwo>().size();
  ASSERT_EQ(multiple_result_one, 1);

  // Test that deleting the game object works correctly
  registry.delete_game_object(0);
  ASSERT_THROW_MESSAGE(registry.get_component<TestGameObjectComponentOne>(0),
                       RegistryException,
                       "The game object `0` is not registered with the registry.")
  ASSERT_THROW_MESSAGE(registry.get_component<TestGameObjectComponentTwo>(0),
                       RegistryException,
                       "The game object `0` is not registered with the registry.")
  ASSERT_EQ(registry.find_components<TestGameObjectComponentOne>().size(), 0);
  ASSERT_EQ(registry.find_components<TestGameObjectComponentTwo>().size(), 0);
  auto multiple_result_two = registry.find_components<TestGameObjectComponentOne, TestGameObjectComponentTwo>().size();
  ASSERT_EQ(multiple_result_two, 0);
}

/// Test that a kinematic game object is added to the registry correctly.
TEST_F(RegistryFixture, TestRegistryGameObjectKinematic) {
  // Test that creating the kinematic game object works correctly
  registry.create_game_object(true, {});
  std::shared_ptr<KinematicObject> kinematic_object = registry.get_kinematic_object(0);
  ASSERT_EQ(kinematic_object->position, Vec2d(0, 0));
  ASSERT_EQ(kinematic_object->velocity, Vec2d(0, 0));
  ASSERT_EQ(kinematic_object->rotation, 0);

  // Test that deleting the kinematic game object works correctly
  registry.delete_game_object(0);
  ASSERT_THROW_MESSAGE(registry.get_kinematic_object(0),
                       RegistryException,
                       "The game object `0` is not registered with the registry.")
}

/// Test that multiple game objects are added to the registry correctly.
TEST_F(RegistryFixture, TestRegistryMultipleGameObjects) {
  // Construct the components list for both game objects
  std::vector<std::unique_ptr<ComponentBase>> components_one, components_two;
  components_one.emplace_back(create_object<TestGameObjectComponentOne>());
  components_two.emplace_back(create_object<TestGameObjectComponentOne>());
  components_two.emplace_back(create_object<TestGameObjectComponentTwo>({10}));

  // Test that creating two game objects works correctly
  ASSERT_EQ(registry.create_game_object(false, std::move(components_one)), 0);
  ASSERT_EQ(registry.create_game_object(false, std::move(components_two)), 1);
  ASSERT_EQ(registry.find_components<TestGameObjectComponentOne>().size(), 2);
  ASSERT_EQ(registry.find_components<TestGameObjectComponentTwo>().size(), 1);
  auto multiple_result_one = registry.find_components<TestGameObjectComponentOne, TestGameObjectComponentTwo>().size();
  ASSERT_EQ(multiple_result_one, 1);

  // Test that deleting the first game object works correctly
  registry.delete_game_object(0);
  ASSERT_EQ(registry.find_components<TestGameObjectComponentOne>().size(), 1);
  ASSERT_EQ(registry.find_components<TestGameObjectComponentTwo>().size(), 1);
  auto multiple_result_two = registry.find_components<TestGameObjectComponentOne, TestGameObjectComponentTwo>().size();
  ASSERT_EQ(multiple_result_two, 1);
}

/// Test that a game object with duplicate components is added to the registry correctly.
TEST_F(RegistryFixture, TestRegistryGameObjectDuplicateComponents) {
  // Construct the components list
  std::vector<std::unique_ptr<ComponentBase>> components;
  components.emplace_back(create_object<TestGameObjectComponentTwo>({10}));
  components.emplace_back(create_object<TestGameObjectComponentTwo>({20}));

  // Test that creating a game object with two of the same components only adds
  // the first one
  registry.create_game_object(false, std::move(components));
  ASSERT_EQ(registry.get_component<TestGameObjectComponentTwo>(0)->test_list[0], 10);
}

/// Test that an exception is thrown if a system is not registered.
TEST_F(RegistryFixture, TestRegistryZeroSystems) {
  ASSERT_THROW_MESSAGE(registry.find_system<TestSystem>(),
                       RegistryException,
                       "The system `struct TestSystem` is not registered with the registry.")
}

/// Test that a system is updated correctly.
TEST_F(RegistryFixture, TestRegistrySystemUpdate) {
  registry.add_system(std::make_shared<TestSystem>(registry));
  auto system_result = registry.find_system<TestSystem>();
  ASSERT_TRUE(system_result != nullptr);
  registry.update(0);
  ASSERT_TRUE(system_result->called);
}

/// Test that multiple walls are added to the system correctly.
TEST_F(RegistryFixture, TestRegistryWalls) {
  ASSERT_EQ(registry.get_walls(), std::unordered_set<Vec2d>());
  registry.add_wall({0, 0});
  registry.add_wall({1, 1});
  registry.add_wall({0, 0});
  ASSERT_EQ(registry.get_walls(), std::unordered_set<Vec2d>({{0, 0}, {1, 1}}));
}
