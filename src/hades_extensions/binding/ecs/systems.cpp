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
  pybind11::class_<ShopSystem, SystemBase, std::shared_ptr<ShopSystem>>(module, "ShopSystem",
                                                                        "Provides facilities to manage a shop system.")
      .def("purchase", &ShopSystem::purchase, pybind11::arg("buyer_id"), pybind11::arg("offering_index"),
           "Purchase an offering from the shop for a buyer.\n\n"
           "Args:\n"
           "    buyer_id: The ID of the buyer.\n"
           "    offering_index: The index of the offering to purchase.\n\n"
           "Raises:\n"
           "    RegistryError: If the game object does not exist or does not have the required components.\n\n"
           "Returns:\n"
           "    Whether the purchase was successful or not.");
}
