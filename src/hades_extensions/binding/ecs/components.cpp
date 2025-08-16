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
           "    The maximum value of the stat.");
  const pybind11::class_<Armour, Stat, std::shared_ptr<Armour>> armour(module, "Armour",
                                                                       "Allows a game object to have an armour stat.");
  const pybind11::class_<Health, Stat, std::shared_ptr<Health>> health(module, "Health",
                                                                       "Allows a game object to have a health stat.");

  // Add the other components
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
}
