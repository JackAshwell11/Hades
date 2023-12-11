// Related header
#include "game_objects/systems/upgrade.hpp"

// Local headers
#include "game_objects/stats.hpp"

// ----- FUNCTIONS ------------------------------
auto UpgradeSystem::upgrade_component(const GameObjectID game_object_id, const std::type_index &target_component) const
    -> bool {
  // Get the component to upgrade as well as the upgrade function
  const auto component{std::static_pointer_cast<Stat>(get_registry()->get_component(game_object_id, target_component))};
  const auto upgrades_component{get_registry()->get_component<Upgrades>(game_object_id)};

  // Check if the component can be upgraded
  if (upgrades_component == nullptr || !upgrades_component->upgrades.contains(target_component) ||
      component->get_current_level() >= component->get_maximum_level()) {
    return false;
  }

  // Upgrade the component
  const auto diff{upgrades_component->upgrades[target_component](component->get_current_level())};
  component->add_to_max_value(diff);
  component->increment_current_level();
  component->set_value(component->get_value() + diff);
  return true;
}
