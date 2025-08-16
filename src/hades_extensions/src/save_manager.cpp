// Related header
#include "save_manager.hpp"

// Std headers
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>

// External headers
#include <nlohmann/json.hpp>

// Local headers
#include "ecs/registry.hpp"
#include "events.hpp"
#include "game_state.hpp"

namespace {
/// The name of the save files.
constexpr auto SAVE_FILE_NAME{"autosave"};

/// The maximum number of save files to keep.
constexpr size_t MAX_SAVE_FILES{20};

/// Generate a unique save file name.
///
/// @return A unique save file name.
auto generate_save_file_name() -> std::string {
  static std::mt19937_64 gen(std::random_device{}());
  return std::format("{}_{:016x}.json", SAVE_FILE_NAME, gen());
}
}  // namespace

SaveManager::SaveManager(const std::shared_ptr<Registry> &registry, const std::shared_ptr<GameState> &game_state)
    : registry_(registry), game_state_(game_state) {}

void SaveManager::set_save_path(const std::string &path) {
  save_path_ = path;
  refresh_save_files();
}

void SaveManager::new_game() const { game_state_->reset_level(LevelType::Lobby); }

void SaveManager::load_save(const int save_index) const {
  const std::string file_path{save_files_.at(save_index).path};
  std::ifstream stream{file_path};
  if (!stream.is_open()) {
    throw std::runtime_error("Could not open save file: " + file_path);
  }

  nlohmann::json json_data;
  stream >> json_data;
  for (const auto &component : registry_->get_game_object_components(game_state_->get_player_id())) {
    component->from_file(json_data);
    component->from_file(json_data, registry_.get());
  }
  game_state_->reset_level(json_data.at("dungeon_level").get<LevelType>());
}

void SaveManager::save_game() {
  const std::filesystem::path file_path{save_path_ / generate_save_file_name()};
  std::ofstream stream{file_path};
  if (!stream.is_open()) {
    throw std::runtime_error("Could not open save file: " + file_path.string());
  }

  try {
    nlohmann::json json_data;
    const std::chrono::zoned_time local_time{std::chrono::current_zone(), std::chrono::system_clock::now()};
    json_data["save_date"] = std::format("{0:%FT%T%z}", local_time);
    for (const auto &component : registry_->get_game_object_components(game_state_->get_player_id())) {
      component->to_file(json_data);
      component->to_file(json_data, registry_.get());
    }
    json_data["dungeon_level"] = game_state_->get_dungeon_level();
    stream << json_data.dump(4);
    stream.close();
    refresh_save_files();
  } catch (...) {
    stream.close();
    std::filesystem::remove(file_path);
    throw;
  }
}

void SaveManager::delete_save(const int save_index) {
  if (save_index < 0 || std::cmp_greater_equal(save_index, save_files_.size())) {
    throw std::out_of_range("Invalid save index: " + std::to_string(save_index));
  }
  if (const std::string file_path{save_files_.at(save_index).path}; std::filesystem::remove(file_path)) {
    refresh_save_files();
  } else {
    throw std::runtime_error("Could not delete save file: " + file_path);
  }
}

void SaveManager::refresh_save_files() {
  save_files_.clear();
  for (const auto &entry : std::filesystem::directory_iterator(save_path_)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".json") {
      continue;
    }
    std::ifstream file_stream(entry.path());
    nlohmann::json json_data;
    file_stream >> json_data;

    const SaveFileInfo info{.name = entry.path().stem().string(),
                            .path = entry.path().string(),
                            .last_modified = json_data.at("save_date").get<std::string>(),
                            .player_level = json_data.at("player_level").get<int>()};
    save_files_.push_back(info);
  }

  std::ranges::sort(save_files_.begin(), save_files_.end(), [](const SaveFileInfo &lhs, const SaveFileInfo &rhs) {
    return lhs.last_modified > rhs.last_modified;
  });
  if (save_files_.size() > MAX_SAVE_FILES) {
    for (auto i{MAX_SAVE_FILES}; i < save_files_.size(); i++) {
      std::filesystem::remove(save_files_[i].path);
    }
    save_files_.resize(MAX_SAVE_FILES);
  }
  notify<EventType::SaveFilesUpdated>(save_files_);
}
