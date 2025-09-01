// Local headers
#include "common.hpp"

void bind_systems(const pybind11::module& module) {
  pybind11::class_<InventorySystem, SystemBase, std::shared_ptr<InventorySystem>>(
      module, "InventorySystem", "Provides facilities to manage inventory components.")
      .def("use_item", &InventorySystem::use_item, pybind11::arg("target_id"), pybind11::arg("item_id"),
           "Use an item on a target game object.\n\n"
           "Args:\n"
           "    target_id: The game object ID of the target.\n"
           "    item_id: The game object ID of the item to use.");

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
