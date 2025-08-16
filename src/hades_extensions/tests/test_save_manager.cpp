// Std headers
#include <fstream>

// External headers
#include <nlohmann/json.hpp>

// Local headers
#include "ecs/registry.hpp"
#include "ecs/systems/level.hpp"
#include "events.hpp"
#include "factories.hpp"
#include "game_state.hpp"
#include "macros.hpp"
#include "save_manager.hpp"

/// Implements the fixture for the save_manager.hpp tests.
class SaveManagerFixture : public testing::Test {
 protected:
  /// The registry which manages the game objects, components, and systems.
  std::shared_ptr<Registry> registry;

  /// The game state which stores the state of the game.
  std::shared_ptr<GameState> game_state;

  /// The save manager which manages the saving and loading of game states.
  std::shared_ptr<SaveManager> save_manager;

  /// The file path to the test save directory.
  std::filesystem::path test_save_path;

  /// Set up shared logic for the tests.
  static void SetUpTestSuite() {
    load_hitbox(GameObjectType::Player, {{0.0, 1.0}, {1.0, 2.0}, {2.0, 0.0}});
    load_hitbox(GameObjectType::Enemy, {{0.0, 1.0}, {1.0, 2.0}, {2.0, 0.0}});
  }

  /// Set up the fixture for the tests.
  void SetUp() override {
    registry = std::make_shared<Registry>();
    game_state = std::make_shared<GameState>(registry);
    save_manager = std::make_shared<SaveManager>(registry, game_state);
    test_save_path = std::filesystem::temp_directory_path() / "test_saves";
    std::filesystem::create_directory(test_save_path);
    save_manager->set_save_path(test_save_path.string());
  }

  /// Tear down the fixture after the tests.
  void TearDown() override {
    std::filesystem::remove_all(test_save_path);
    clear_listeners();
  }
};

/// Test that setting the save path to a valid path works correctly.
TEST_F(SaveManagerFixture, TestSaveManagerSetSavePathValidPath) {
  const std::filesystem::path valid_save_path{"valid_save_path"};
  std::filesystem::create_directory(valid_save_path);
  ASSERT_NO_THROW(save_manager->set_save_path(valid_save_path.string()));
}

/// Test that setting the save path to an invalid path throws an exception.
TEST_F(SaveManagerFixture, TestSaveManagerSetSavePathInvalidPath) {
  const std::filesystem::path invalid_path{"test_save_path"};
  ASSERT_THROW(save_manager->set_save_path(invalid_path.string()), std::runtime_error);
}

/// Test that creating a new game resets the game state to the lobby level.
TEST_F(SaveManagerFixture, TestSaveManagerNewGameResetGameState) {
  ASSERT_EQ(game_state->get_dungeon_level(), LevelType::None);
  save_manager->new_game();
  ASSERT_EQ(game_state->get_dungeon_level(), LevelType::Lobby);
}

/// Test that loading a save works correctly.
TEST_F(SaveManagerFixture, TestSaveManagerLoadSaveValid) {
  game_state->reset_level(LevelType::SecondDungeon);
  save_manager->save_game();
  game_state->reset_level(LevelType::Lobby);
  ASSERT_EQ(game_state->get_dungeon_level(), LevelType::Lobby);
  ASSERT_NO_THROW(save_manager->load_save(0));
  ASSERT_EQ(game_state->get_dungeon_level(), LevelType::SecondDungeon);
}

/// Test that loading a save which doesn't exist throws an exception.
TEST_F(SaveManagerFixture, TestSaveManagerLoadSaveNonExistent) {
  ASSERT_THROW(save_manager->load_save(0), std::out_of_range);
}

/// Test that loading an invalid JSON save throws an exception.
TEST_F(SaveManagerFixture, TestSaveManagerLoadSaveInvalidJson) {
  save_manager->save_game();
  const std::filesystem::path save_file{(*std::filesystem::directory_iterator(test_save_path)).path()};
  std::ofstream stream{save_file};
  stream << "{\"invalid json content}";
  stream.close();
  ASSERT_THROW_MESSAGE(
      save_manager->load_save(0), nlohmann::json::parse_error,
      "[json.exception.parse_error.101] parse error at line 1, column 24: syntax error while parsing object key - "
      "invalid string: missing closing quote; last read: '\"invalid json content}'; expected string literal");
}

/// Test that saving a game works correctly.
TEST_F(SaveManagerFixture, TestSaveManagerSaveGameValid) {
  game_state->reset_level(LevelType::Lobby);
  ASSERT_NO_THROW(save_manager->save_game());
  ASSERT_EQ(std::ranges::distance(std::filesystem::directory_iterator(test_save_path)), 1);
}

/// Test that saving multiple games works correctly.
TEST_F(SaveManagerFixture, TestSaveManagerSaveGameMultipleGames) {
  game_state->reset_level(LevelType::Lobby);
  ASSERT_NO_THROW(save_manager->save_game());
  game_state->reset_level(LevelType::FirstDungeon);
  ASSERT_NO_THROW(save_manager->save_game());
  ASSERT_EQ(std::ranges::distance(std::filesystem::directory_iterator(test_save_path)), 2);
}

/// Test that saving too many files works correctly.
TEST_F(SaveManagerFixture, TestSaveManagerSaveGameLimitReached) {
  game_state->reset_level(LevelType::Lobby);
  for (int i{0}; i < 50; i++) {
    ASSERT_NO_THROW(save_manager->save_game());
  }
  auto files{std::filesystem::directory_iterator(test_save_path)};
  ASSERT_EQ(std::distance(begin(files), end(files)), 20);
}

/// Test that deleting a save works correctly.
TEST_F(SaveManagerFixture, TestSaveManagerDeleteSaveValid) {
  save_manager->save_game();
  save_manager->delete_save(0);
  auto files{std::filesystem::directory_iterator(test_save_path)};
  ASSERT_EQ(std::distance(begin(files), end(files)), 0);
}

/// Test that deleting a save which doesn't exist throws an exception.
TEST_F(SaveManagerFixture, TestSaveManagerDeleteSaveNonExistent) {
  ASSERT_THROW_MESSAGE(save_manager->delete_save(0), std::out_of_range, "Invalid save index: 0");
}

/// Test that updating the save directory calls the correct callbacks.
TEST_F(SaveManagerFixture, TestSaveManagerUpdateSavesCallbacks) {
  std::vector<SaveFileInfo> saves;
  add_callback<EventType::SaveFilesUpdated>([&saves](const std::vector<SaveFileInfo> &saveData) { saves = saveData; });
  for (int i{1}; i < 6; i++) {
    registry->get_component<PlayerLevel>(game_state->get_player_id())->level = i;
    game_state->reset_level(LevelType::Lobby);
    save_manager->save_game();
  }
  ASSERT_EQ(saves.size(), 5);
  ASSERT_EQ(saves[0].player_level, 5);
  ASSERT_EQ(saves[1].player_level, 4);
  ASSERT_EQ(saves[2].player_level, 3);
  ASSERT_EQ(saves[3].player_level, 2);
  ASSERT_EQ(saves[4].player_level, 1);
}
