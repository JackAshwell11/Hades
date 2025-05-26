// Local headers
#include "common.hpp"

void bind_systems(const pybind11::module &module) {
  pybind11::class_<PhysicsSystem, SystemBase, std::shared_ptr<PhysicsSystem>>(
      module, "PhysicsSystem", "Provides facilities to manipulate a game object's physics.")
      .def(
          "get_wall_distances",
          [](const PhysicsSystem &physics_system, const pybind11::tuple &current_position) {
            const auto wall_distances{physics_system.get_wall_distances(
                {pybind11::cast<double>(current_position[0]), pybind11::cast<double>(current_position[1])})};
            std::vector<std::pair<double, double>> distances;
            for (const auto &[x, y] : wall_distances) {
              distances.emplace_back(x, y);
            }
            return distances;
          },
          "Calculate the distance to the walls around a game object.\n\n"
          "Args:\n"
          "    current_position: The current position of the game object.\n\n"
          "Returns:\n"
          "    The distances to the walls around the game object.");
  pybind11::class_<UpgradeSystem, SystemBase, std::shared_ptr<UpgradeSystem>>(
      module, "UpgradeSystem", "Provides facilities to manipulate game object upgrades.")
      .def(
          "upgrade_component",
          [](const UpgradeSystem &upgrade_system, const GameObjectID game_object_id,
             const pybind11::object &target_component) {
            return upgrade_system.upgrade_component(game_object_id, get_type_index(target_component));
          },
          pybind11::arg("game_object_id"), pybind11::arg("target_component"),
          "Upgrade a component to the next level if possible.\n\n"
          "Args:\n"
          "    game_object_id: The ID of the game object to upgrade the component for.\n"
          "    target_component: The type of component to upgrade.\n\n"
          "Raises:\n"
          "    RegistryError: If the game object does not exist or does not have the target component.\n\n"
          "Returns:\n"
          "    Whether the component upgrade was successful or not.");
}
