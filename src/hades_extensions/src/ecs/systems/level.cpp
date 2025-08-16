// Related header
#include "ecs/systems/level.hpp"

// External headers
#include <nlohmann/json.hpp>

void PlayerLevel::to_file(nlohmann::json &json) const {
  json["player_level"] = level;
  json["experience"] = experience;
}

void PlayerLevel::from_file(const nlohmann::json &json) {
  level = json.at("player_level").get<int>();
  experience = json.at("experience").get<double>();
}
