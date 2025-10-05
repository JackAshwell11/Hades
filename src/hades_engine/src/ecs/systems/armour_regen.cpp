// Related header
#include "ecs/systems/armour_regen.hpp"

// Local headers
#include "ecs/registry.hpp"
#include "ecs/stats.hpp"

namespace {
/// The cooldown time between each armour regeneration.
constexpr auto ARMOUR_REGEN_COOLDOWN{1.0};

/// The amount of armour to regenerate each time.
constexpr auto ARMOUR_REGEN_AMOUNT{1};
}  // namespace

void ArmourRegenSystem::update(const double delta_time) const {
  // Update the time since the last armour regen then check if the armour should be regenerated
  for (const auto& [_, component_tuple] : get_registry()->get_game_object_components<Armour, ArmourRegen>()) {
    const auto& [armour, armour_regen]{component_tuple};
    armour_regen->time_since_armour_regen += delta_time;
    if (armour_regen->time_since_armour_regen >= ARMOUR_REGEN_COOLDOWN) {
      armour->set_value(armour->get_value() + ARMOUR_REGEN_AMOUNT);
      armour_regen->time_since_armour_regen = 0;
    }
  }
}
