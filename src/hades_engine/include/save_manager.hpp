// Ensure this file is only included once
#pragma once

// Std headers
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

// Forward declarations
class Registry;
class GameState;

/// Represents information about a save file.
struct SaveFileInfo {
  /// The name of the save file.
  std::string name;

  /// The path to the save file.
  std::string path;

  /// The last modified time of the save file.
  std::string last_modified;

  /// The player level of the save file.
  int player_level;
};

/// Manages the saving and loading of game states.
class SaveManager {
 public:
  /// Initialise the object.
  ///
  /// @param registry - The registry which manages the game objects, components, and systems.
  /// @param game_state - The storage for the state of the game.
  explicit SaveManager(const std::shared_ptr<Registry>& registry, const std::shared_ptr<GameState>& game_state);

  /// Set the path where the game state will be saved.
  ///
  /// @param path - The path to save the game state to.
  void set_save_path(const std::string& path);

  /// Create a new game.
  void new_game() const;

  /// Load a save from a file.
  ///
  /// @param save_index - The index of the save file to load.
  /// @throws std::runtime_error if the file cannot be opened or read or there was an error parsing the JSON file.
  void load_save(int save_index) const;

  /// Save the current game state to a file.
  void save_game();

  /// Delete a save file.
  ///
  /// @param save_index - The index of the save file to delete.
  /// @throws std::out_of_range if the save index is invalid or does not exist.
  void delete_save(int save_index);

 private:
  /// Refreshes the cache of save files.
  ///
  /// @throws nlohmann::json::parse_error if there is an error parsing the JSON file.
  void refresh_save_files();

  /// The registry that manages game objects, components, and systems.
  std::shared_ptr<Registry> registry_;

  /// Stores the state of the game.
  std::shared_ptr<GameState> game_state_;

  /// The path where the game state will be saved.
  std::filesystem::path save_path_;

  /// The cached save files.
  std::vector<SaveFileInfo> save_files_;
};
