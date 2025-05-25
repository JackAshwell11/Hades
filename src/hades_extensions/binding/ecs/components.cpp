// Local headers
#include "common.hpp"

void bind_components(const pybind11::module_ &module) {
  // Add the stat components
  pybind11::class_<Stat, ComponentBase, std::shared_ptr<Stat>>(
      module, "Stat", "Represents a component that has a variable value and maximum value.")
      .def("get_value", &Stat::get_value,
           "Get the value of the stat.\n\n"
           "Returns:\n"
           "    The value of the stat.")
      .def("set_value", &Stat::set_value,
           "Set the value of the stat.\n\n"
           "Args:\n"
           "    new_value: The new value of the stat.")
      .def("get_max_value", &Stat::get_max_value,
           "Get the maximum value of the stat.\n\n"
           "Returns:\n"
           "    The maximum value of the stat.")
      .def("add_to_max_value", &Stat::add_to_max_value,
           "Add a value to the maximum value of the stat.\n\n"
           "Args:\n"
           "    value: The value to add to the maximum value of the stat.")
      .def("get_current_level", &Stat::get_current_level,
           "Get the current level of the stat.\n\n"
           "Returns:\n"
           "    The current level of the stat.")
      .def("increment_current_level", &Stat::increment_current_level, "Increment the current level of the stat.")
      .def("get_max_level", &Stat::get_max_level,
           "Get the maximum level of the stat.\n\n"
           "Returns:\n"
           "    The maximum level of the stat.");
  const pybind11::class_<Armour, Stat, std::shared_ptr<Armour>> armour(module, "Armour",
                                                                       "Allows a game object to have an armour stat.");
  const pybind11::class_<Health, Stat, std::shared_ptr<Health>> health(module, "Health",
                                                                       "Allows a game object to have a health stat.");
  const pybind11::class_<InventorySize, Stat, std::shared_ptr<InventorySize>> inventory_size(
      module, "InventorySize", "Allows a game object to change the size of its inventory.");

  // Add the other components
  pybind11::class_<Inventory, ComponentBase, std::shared_ptr<Inventory>>(
      module, "Inventory", "Allows a game object to have a fixed size inventory.")
      .def_readonly("items", &Inventory::items);
  pybind11::class_<KinematicComponent, ComponentBase, std::shared_ptr<KinematicComponent>>(
      module, "KinematicComponent", "Allows a game object to interact with the physics system.")
      .def(
          "get_position",
          [](const KinematicComponent &kinematic_component) {
            const auto [x, y]{cpBodyGetPosition(*kinematic_component.body)};
            return pybind11::make_tuple(x, y);
          },
          "Get the position of the game object.\n\n"
          "Returns:\n"
          "    The position of the game object.")
      .def(
          "get_velocity",
          [](const KinematicComponent &kinematic_component) {
            const auto [x, y]{cpBodyGetVelocity(*kinematic_component.body)};
            return pybind11::make_tuple(x, y);
          },
          "Get the velocity of the game object.\n\n"
          "Returns:\n"
          "    The velocity of the game object.")
      .def(
          "set_rotation",
          [](KinematicComponent &kinematic_component, const double angle) { kinematic_component.rotation = angle; },
          pybind11::arg("angle"),
          "Set the rotation of the game object.\n\n"
          "Args:\n"
          "    angle: The angle to set the game object to.");
  pybind11::class_<Money, ComponentBase, std::shared_ptr<Money>>(
      module, "Money", "Allows a game object to record the amount of money it has.")
      .def_readwrite("money", &Money::money);
  pybind11::class_<PythonSprite, ComponentBase, std::shared_ptr<PythonSprite>>(
      module, "PythonSprite", "Allows a game object to hold a reference to the Python sprite object.")
      .def_readwrite("sprite", &PythonSprite::sprite);
  pybind11::class_<Upgrades, ComponentBase, std::shared_ptr<Upgrades>>(module, "Upgrades",
                                                                       "Allows a game object to be upgraded.")
      .def_property_readonly("upgrades", [](const Upgrades &upgrades) {
        pybind11::dict target_upgrades;
        for (const auto &[type, func] : upgrades.upgrades) {
          target_upgrades[get_python_type(type)] = func;
        }
        return target_upgrades;
      });
}
