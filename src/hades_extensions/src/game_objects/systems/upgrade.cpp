// Related header
#include "game_objects/systems/upgrade.hpp"

// Local headers
#include "game_objects/registry.hpp"
#include "game_objects/stats.hpp"

// ----- FUNCTIONS ------------------------------
auto UpgradeSystem::upgrade_component(const GameObjectID game_object_id, const std::type_index &target_component) const
    -> bool {
  // Get the component to upgrade as well as the upgrade function
  const auto component{std::static_pointer_cast<Stat>(get_registry()->get_component(game_object_id, target_component))};
  const auto upgrades_component{get_registry()->get_component<Upgrades>(game_object_id)};

  // Check if the component can be upgraded
  if (!upgrades_component->upgrades.contains(target_component) ||
      component->get_current_level() >= component->get_max_level()) {
    return false;
  }

  // Check if the player has enough money
  const auto [increase, cost]{upgrades_component->upgrades[target_component]};
  const auto money{get_registry()->get_component<Money>(game_object_id)};
  if (money->money < cost(component->get_current_level())) {
    return false;
  }

  // Upgrade the component
  const auto diff{increase(component->get_current_level())};
  money->money -= static_cast<int>(cost(component->get_current_level()));
  component->add_to_max_value(diff);
  component->increment_current_level();
  component->set_value(component->get_value() + diff);
  return true;
}
