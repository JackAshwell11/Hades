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

/// Represents an upgradable stat offering in the shop.
template <typename StatComponent>
struct StatUpgradeOffering final : ShopOffering {
  /// Initialise the object.
  ///
  /// @param name - The name of the offering.
  /// @param description - The description of the offering.
  /// @param base_cost - The base cost of the offering.
  /// @param cost_multiplier - The cost multiplier of the offering.
  /// @param base_value - The base value of the offering.
  /// @param value_multiplier - The value multiplier of the offering.
  StatUpgradeOffering(const std::string& name, const std::string& description, const double base_cost,
                      const double cost_multiplier, const double base_value, const double value_multiplier)
      : ShopOffering(name, description, base_cost, cost_multiplier),
        base_value(base_value),
        value_multiplier(value_multiplier) {}

  /// Apply the offering to the buyer.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  /// @param buyer_id - The ID of the buyer.
  /// @throws RegistryError - If the game object does not exist or does not have the required components.
  /// @return true if the application was successful, false otherwise.
  auto apply(const Registry* registry, const GameObjectID buyer_id) const -> bool override {
    const auto& component{registry->get_component<StatComponent>(buyer_id)};
    if (component->get_current_level() >= component->get_max_level()) {
      return false;
    }
    const auto diff{calculate_exponential(base_value, component->get_current_level(), value_multiplier)};
    component->add_to_max_value(diff);
    component->increment_current_level();
    component->set_value(component->get_value() + diff);
    return true;
  }

  /// Get the cost of the offering.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  /// @param buyer_id - The ID of the buyer.
  /// @return The cost of the offering.
  [[nodiscard]] auto get_cost(const Registry* registry, const GameObjectID buyer_id) const -> double override {
    const auto component{registry->get_component<StatComponent>(buyer_id)};
    return calculate_exponential(base_cost, component->get_current_level(), cost_multiplier);
  }

  /// The base value of the offering.
  double base_value;

  /// The value multiplier of the offering.
  double value_multiplier;
};

/// Represents a one-time component unlock offering in the shop.
struct ComponentUnlockOffering final : ShopOffering {
  /// Initialise the object.
  ///
  /// @param name - The name of the offering.
  /// @param description - The description of the offering.
  /// @param cost - The cost of the offering.
  /// @param cost_multiplier - The cost multiplier of the offering.
  ComponentUnlockOffering(const std::string& name, const std::string& description, const double cost,
                          const double cost_multiplier)
      : ShopOffering(name, description, cost, cost_multiplier) {}

  /// Apply the offering to the buyer.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  /// @param buyer_id - The ID of the buyer.
  /// @throws RegistryError - If the game object does not exist or does not have the required components.
  /// @return true if the application was successful, false otherwise.
  auto apply(const Registry* registry, GameObjectID buyer_id) const -> bool override;
};

/// Represents a repeatable item offering in the shop.
struct ItemOffering final : ShopOffering {
  /// Initialise the object.
  ///
  /// @param name - The name of the offering.
  /// @param description - The description of the offering.
  /// @param base_cost - The base cost of the offering.
  /// @param cost_multiplier - The cost multiplier of the offering.
  ItemOffering(const std::string& name, const std::string& description, const double base_cost,
               const double cost_multiplier)
      : ShopOffering(name, description, base_cost, cost_multiplier) {}

  /// Apply the offering to the buyer.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  /// @param buyer_id - The ID of the buyer.
  /// @throws RegistryError - If the game object does not exist or does not have the required components.
  /// @return true if the application was successful, false otherwise.
  auto apply(const Registry* registry, GameObjectID buyer_id) const -> bool override;
};

/// Get a stat upgrade offering based on the stat type.
///
/// @param name - The name of the offering.
/// @param description - The description of the offering.
/// @param stat_type - The type of stat to upgrade.
/// @param base_cost - The base cost of the offering.
/// @param cost_multiplier - The cost multiplier of the offering.
/// @param base_value - The base value of the offering.
/// @param value_multiplier - The value multiplier of the offering.
/// @throws std::runtime_error if the stat type is not recognised.
/// @return A unique pointer to the stat upgrade offering.
auto get_stat_upgrade_offering(const std::string& name, const std::string& description, const std::string& stat_type,
                               const double base_cost, const double cost_multiplier, const double base_value,
                               const double value_multiplier) -> std::unique_ptr<ShopOffering> {
  if (stat_type == "Health") {
    return std::make_unique<StatUpgradeOffering<Health>>(name, description, base_cost, cost_multiplier, base_value,
                                                         value_multiplier);
  }
  if (stat_type == "Armour") {
    return std::make_unique<StatUpgradeOffering<Armour>>(name, description, base_cost, cost_multiplier, base_value,
                                                         value_multiplier);
  }
  throw std::runtime_error("Unknown component type: " + stat_type);
}
}  // namespace

void Money::to_file(nlohmann::json& json) const { json["money"] = money; }

void Money::from_file(const nlohmann::json& json) { money = json.at("money").get<int>(); }

auto ShopOffering::get_cost(const Registry* /*registry*/, const GameObjectID /*buyer_id*/) const -> double {
  return calculate_exponential(base_cost, 0, cost_multiplier);
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
      const auto stat_type{offering.at("stat_type").get<std::string>()};
      const auto base_value{offering.at("base_value").get<double>()};
      const auto value_multiplier{offering.at("value_multiplier").get<double>()};
      offerings_.push_back(get_stat_upgrade_offering(name, description, stat_type, base_cost, cost_multiplier,
                                                     base_value, value_multiplier));
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
