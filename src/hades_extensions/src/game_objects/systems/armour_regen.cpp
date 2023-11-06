// Related header
#include "game_objects/systems/armour_regen.hpp"

// Local headers
#include "game_objects/stats.hpp"

// ----- CONSTANTS -------------------------------
constexpr int ARMOUR_REGEN_AMOUNT{1};

// ----- FUNCTIONS ------------------------------
void ArmourRegenSystem::update(const double delta_time) const {
  // Update the time since the last armour regen then check if the armour should be regenerated
  for (auto &[_, component_tuple] : get_registry()->find_components<Armour, ArmourRegen>()) {
    auto &[armour, armour_regen]{component_tuple};
    armour_regen->time_since_armour_regen += delta_time;
    if (armour_regen->time_since_armour_regen >= armour_regen->get_value()) {
      armour->set_value(armour->get_value() + ARMOUR_REGEN_AMOUNT);
      armour_regen->time_since_armour_regen = 0;
    }
  }
}
