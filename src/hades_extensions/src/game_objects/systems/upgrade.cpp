// Related header
#include "game_objects/systems/upgrade.hpp"

// Local headers
#include "game_objects/stats.hpp"

// ----- STRUCTURES ------------------------------
bool UpgradeSystem::upgrade_component(GameObjectID game_object_id, const std::type_index &target_component) {
  // Get the component to upgrade as well as the upgrade function
  auto component = std::static_pointer_cast<Stat>(registry.get_component(game_object_id, target_component));
  auto upgrades_component = registry.get_component<Upgrades>(game_object_id);

  // Check if the component can be upgraded
  if (upgrades_component == nullptr || !upgrades_component->upgrades.contains(target_component) ||
      component->current_level >= component->maximum_level) {
    return false;
  }

  // Upgrade the component
  auto diff = upgrades_component->upgrades[target_component](component->current_level);
  component->max_value += diff;
  component->current_level++;
  component->set_value(component->get_value() + diff);
  return true;
}
