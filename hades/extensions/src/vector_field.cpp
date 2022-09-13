#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <math.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <limits>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <ostream>
#include "hades_common.h"


struct FloatPair {
    /* Represents a namedtuple describing a pair of floats */
    float x, y;
};


IntPair py_handle_to_int_pair(pybind11::handle handle) {
    pybind11::tuple py_tup = pybind11::cast<pybind11::tuple>(handle);
    return {pybind11::cast<int>(py_tup[0]), pybind11::cast<int>(py_tup[1])};
}


FloatPair py_handle_to_float_pair(pybind11::handle handle) {
    pybind11::tuple py_tup = pybind11::cast<pybind11::tuple>(handle);
    return {pybind11::cast<float>(py_tup[0]), pybind11::cast<float>(py_tup[1])};
}


pybind11::tuple int_pair_to_py_tup(IntPair int_pair) {
    return pybind11::make_tuple(int_pair.x, int_pair.y);
}


pybind11::tuple float_pair_to_py_tup(FloatPair float_pair) {
    return pybind11::make_tuple(float_pair.x, float_pair.y);
}


class VectorField {
public:
    VectorField(int sprite_pixel_size, int grid_height, int grid_width, pybind11::list walls) {
        this->sprite_size = sprite_pixel_size;
        this->height = grid_height;
        this->width = grid_width;
        for (pybind11::handle current_pos_handle : walls) {
            IntPair current_pos = this->get_tile_pos_for_pixel(py_handle_to_float_pair(current_pos_handle));
            walls_dict[current_pos] = std::numeric_limits<int>::max();
        }
    }

    IntPair get_tile_pos_for_pixel(FloatPair position) {
        return {(int)floor(position.x / this->sprite_size), (int)floor(position.y / this->sprite_size)};
    }

    pybind11::tuple recalculate_map(pybind11::tuple player_pos, int player_view_distance) {
        pybind11::print(player_pos);
        pybind11::print(player_view_distance);
        return pybind11::make_tuple("test", "test");
    }

    int get_width() {return this->width;}
    int get_height() {return this->height;}

private:
    int sprite_size, height, width;
    std::unordered_map<IntPair, int> walls_dict;
    std::unordered_map<IntPair, int> distances;
    std::unordered_map<IntPair, IntPair> vector_dict;
};


PYBIND11_MODULE(vector_field, m) {
    m.doc() = "Creates a vector field useful for navigating enemies around the"
              "game map.";

    pybind11::class_<VectorField>(m, "VectorField")
        .def(pybind11::init<int, int, int, pybind11::list>())
        .def("recalculate_map", &VectorField::recalculate_map)
        .def_property_readonly("width", &VectorField::get_width)
        .def_property_readonly("height", &VectorField::get_height);
}
