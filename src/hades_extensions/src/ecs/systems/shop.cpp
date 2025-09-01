// Related header
#include "ecs/systems/shop.hpp"

// Std headers
#include <utility>

// External headers
#include <nlohmann/json.hpp>

// Local headers
#include "ecs/registry.hpp"
#include "ecs/stats.hpp"
#include "events.hpp"

namespace {
/// Calculate an exponential value based on the base, level, and multiplier values.
///
/// @param base - The base value.
/// @param level - The level value.
/// @param multiplier - The multiplier value.
/// @return The calculated exponential value.
auto calculate_exponential(const double base, const int level, const double multiplier) -> double {
  return base + std::pow(level, multiplier);
}

/// Get a component type from a string.
///
/// @param type - The string representation of the component type.
/// @throws std::runtime_error if the type is not recognised.
/// @return The type index of the component type.
auto get_component_type_from_string(const std::string& type) -> std::type_index {
  static const std::unordered_map<std::string, std::type_index> type_map{{"Health", typeid(Health)},
                                                                         {"Armour", typeid(Armour)}};
  if (type_map.contains(type)) {
    return type_map.at(type);
  }
  throw std::runtime_error("Unknown component type: " + type);
}
}  // namespace

void Money::to_file(nlohmann::json& json) const { json["money"] = money; }

void Money::from_file(const nlohmann::json& json) { money = json.at("money").get<int>(); }

auto ShopOffering::get_cost(const Registry* /*registry*/, const GameObjectID /*buyer_id*/) const -> double {
  return calculate_exponential(base_cost, 0, cost_multiplier);
}

auto StatUpgradeOffering::get_cost(const Registry* registry, const GameObjectID buyer_id) const -> double {
  const auto component{std::static_pointer_cast<Stat>(registry->get_component(buyer_id, component_type))};
  return calculate_exponential(base_cost, component->get_current_level(), cost_multiplier);
}

auto StatUpgradeOffering::apply(const Registry* registry, const GameObjectID buyer_id) const -> bool {
  const auto& component{std::static_pointer_cast<Stat>(registry->get_component(buyer_id, component_type))};
  if (component->get_current_level() >= component->get_max_level()) {
    return false;
  }
  const auto diff{calculate_exponential(base_value, component->get_current_level(), value_multiplier)};
  component->add_to_max_value(diff);
  component->increment_current_level();
  component->set_value(component->get_value() + diff);
  return true;
}

auto ComponentUnlockOffering::apply(const Registry* /*registry*/, const GameObjectID /*buyer_id*/) const -> bool {
  return true;
}

auto ItemOffering::apply(const Registry* /*registry*/, const GameObjectID /*buyer_id*/) const -> bool { return true; }

void ShopSystem::add_offerings(std::istream& stream, const GameObjectID player_id) {
  nlohmann::json offerings;
  stream >> offerings;
  for (int i{0}; std::cmp_less(i, offerings.size()); i++) {
    const auto& offering{offerings.at(i)};
    const auto type{offering.at("type").get<std::string>()};
    const auto name{offering.at("name").get<std::string>()};
    const auto description{offering.at("description").get<std::string>()};
    const auto icon_type{offering.at("icon_type").get<std::string>()};
    const auto base_cost{offering.at("base_cost").get<double>()};
    const auto cost_multiplier{offering.at("cost_multiplier").get<double>()};
    if (type == "stat") {
      const auto component_type{get_component_type_from_string(offering.at("stat_type").get<std::string>())};
      const auto base_value{offering.at("base_value").get<double>()};
      const auto value_multiplier{offering.at("value_multiplier").get<double>()};
      offerings_.push_back(std::make_unique<StatUpgradeOffering>(name, description, component_type, base_cost,
                                                                 cost_multiplier, base_value, value_multiplier));
    } else if (type == "component") {
      offerings_.push_back(std::make_unique<ComponentUnlockOffering>(name, description, base_cost, cost_multiplier));
    } else if (type == "item") {
      offerings_.push_back(std::make_unique<ItemOffering>(name, description, base_cost, cost_multiplier));
    } else {
      throw std::runtime_error("Unknown offering type: " + type);
    }
    notify<EventType::ShopItemLoaded>(i, std::make_tuple(name, description, icon_type),
                                      get_offering_cost(i, player_id));
  }
}

auto ShopSystem::get_offering(const int offering_index) const -> const ShopOffering* {
  if (offering_index < 0 || std::cmp_greater_equal(offering_index, offerings_.size())) {
    return nullptr;
  }
  return offerings_[offering_index].get();
}

auto ShopSystem::get_offering_cost(const int offering_index, const GameObjectID buyer_id) const -> int {
  const auto offering{get_offering(offering_index)};
  if (offering == nullptr) {
    return -1;
  }
  return static_cast<int>(std::round(offering->get_cost(get_registry(), buyer_id)));
}

auto ShopSystem::purchase(const GameObjectID buyer_id, const int offering_index) const -> bool {
  // Check if the buyer can purchase the offering or not
  if (std::cmp_greater_equal(offering_index, offerings_.size())) {
    return false;
  }
  const auto money{get_registry()->get_component<Money>(buyer_id)};
  const auto cost{get_offering_cost(offering_index, buyer_id)};
  if (money->money < cost) {
    return false;
  }

  // Try to apply the offering
  const bool success{offerings_[offering_index]->apply(get_registry(), buyer_id)};
  if (success) {
    money->money -= cost;
    notify<EventType::ShopItemPurchased>(offering_index, get_offering_cost(offering_index, buyer_id));
  }
  return success;
}
