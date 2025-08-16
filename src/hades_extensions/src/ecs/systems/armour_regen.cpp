// Related header
#include "ecs/systems/armour_regen.hpp"

// External headers
#include <nlohmann/json.hpp>

// Local headers
#include "ecs/registry.hpp"

namespace {
/// The amount of armour to regenerate each time.
constexpr int ARMOUR_REGEN_AMOUNT{1};
}  // namespace

void ArmourRegen::to_file(nlohmann::json &json) const { to_file_base(json["armour_regen"]); }

void ArmourRegen::from_file(const nlohmann::json &json) { from_file_base(json.at("armour_regen")); }

void ArmourRegenSystem::update(const double delta_time) const {
  // Update the time since the last armour regen then check if the armour should be regenerated
  for (const auto &[_, component_tuple] : get_registry()->find_components<Armour, ArmourRegen>()) {
    const auto &[armour, armour_regen]{component_tuple};
    armour_regen->time_since_armour_regen += delta_time;
    if (armour_regen->time_since_armour_regen >= armour_regen->get_value()) {
      armour->set_value(armour->get_value() + ARMOUR_REGEN_AMOUNT);
      armour_regen->time_since_armour_regen = 0;
    }
  }
}
