// Related header
#include "ecs/stats.hpp"

// External headers
#include <nlohmann/json.hpp>

void Stat::reset() { value_ = max_value_; }

void Stat::to_file_base(nlohmann::json &json) const {
  json = {{"value", value_}, {"max_value", max_value_}, {"current_level", current_level_}, {"max_level", max_level_}};
}

void Stat::from_file_base(const nlohmann::json &json) {
  value_ = json.at("value").get<double>();
  max_value_ = json.at("max_value").get<double>();
  current_level_ = json.at("current_level").get<int>();
  max_level_ = json.at("max_level").get<int>();
}

void Armour::to_file(nlohmann::json &json) const { to_file_base(json["armour"]); }

void Armour::from_file(const nlohmann::json &json) { from_file_base(json.at("armour")); }

void Health::to_file(nlohmann::json &json) const { to_file_base(json["health"]); }

void Health::from_file(const nlohmann::json &json) { from_file_base(json.at("health")); }
