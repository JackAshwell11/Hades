// Related header
#include "ecs/systems/shop.hpp"

// Std headers
#include <utility>

// External headers
#include <nlohmann/json.hpp>

// Local headers
#include "ecs/registry.hpp"
#include "ecs/stats.hpp"
#include "ecs/systems/level.hpp"
#include "events.hpp"

auto ShopOffering::apply(const Registry* /*registry*/, const GameObjectID /*buyer_id*/) const -> bool { return true; }

auto ShopOffering::get_cost(const Registry* registry, const GameObjectID buyer_id) const -> int {
  return static_cast<int>(
      std::round(base_cost + std::pow(registry->get_component<PlayerLevel>(buyer_id)->level, cost_multiplier)));
}

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
    if (type == "component") {
      const ShopOffering shop_offering{name, description, base_cost, cost_multiplier};
      offerings_.push_back(shop_offering);
      notify<EventType::ShopItemLoaded>(i, std::make_tuple(name, description, icon_type),
                                        shop_offering.get_cost(get_registry(), player_id));
    } else {
      throw std::runtime_error("Unknown offering type: " + type);
    }
  }
}

auto ShopSystem::get_offering(const int offering_index) const -> const ShopOffering& {
  if (offering_index < 0 || std::cmp_greater_equal(offering_index, offerings_.size())) {
    throw std::out_of_range("Offering index out of range");
  }
  return offerings_.at(offering_index);
}

auto ShopSystem::purchase(const GameObjectID buyer_id, const int offering_index) const -> bool {
  // Check if the buyer can purchase the offering or not
  if (std::cmp_greater_equal(offering_index, offerings_.size())) {
    return false;
  }
  const auto offering{get_offering(offering_index)};
  const auto money{get_registry()->get_component<Money>(buyer_id)};
  const auto cost{offering.get_cost(get_registry(), buyer_id)};
  if (money->money < cost) {
    return false;
  }

  // Try to apply the offering
  const bool success{offering.apply(get_registry(), buyer_id)};
  if (success) {
    money->money -= cost;
    notify<EventType::ShopItemPurchased>(offering_index, offering.get_cost(get_registry(), buyer_id));
  }
  return success;
}
