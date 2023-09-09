// External includes
#include "gtest/gtest.h"

// Custom includes
#include "game_objects/components.hpp"

// ----- CLASSES ------------------------------
/// Represents a game object attribute useful for testing.
class TestGameObjectAttribute : public GameObjectAttributeBase {
 public:
  /// Initialise the object.
  ///
  /// @param initial_value - The initial value of the movement force attribute.
  /// @param level_limit - The level limit of the movement force attribute.
  TestGameObjectAttribute(float initial_value, int level_limit) : GameObjectAttributeBase(initial_value, level_limit) {}
};

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

  /// Update the system.
  void update(Registry &registry, float delta_time) final {
    called = true;
  }
};

// ----- FIXTURES ------------------------------
/// Hold fixtures relating to the game_objects/ C++ tests.
class GameObjectsFixtures : public testing::Test {
 protected:
  /// The registry that manages the game objects, components, and systems.
  Registry registry{};

  /// A test game object attribute.
  TestGameObjectAttribute test_game_object_attribute{150, 3};

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
