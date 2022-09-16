#include "hades_common.h"
//#include <math.h>
#include <structmember.h>
#include <deque>
#include <string>
#include <iostream>


// ----- CONSTANTS ------------------------------
/* The width, height and sprite size constants for use in the vector field */
int WIDTH, HEIGHT, SPRITE_SIZE;


/* An unordered map of wall grid positions with the cost of traversing them. This is
used to update the distances unordered_map in recalculate_map() since the cost of
traversing a wall is always infinity */
std::unordered_map<IntPair, int> WALLS_DICT;


// ----- DOCSTRINGS ------------------------------
/* Stores the docstring for the vector_field module */
PyDoc_STRVAR(
    vector_field_module_docstring,
    "Creates a vector field useful for navigating enemies around the game map."
);


/* Stores the docstring for the VectorField type */
PyDoc_STRVAR(
    vector_field_type_docstring,
    "Represents a vector field allowing pathfinding for large amounts of enemies.\n\n"
    "The steps needed to accomplish this:\n"
    "    1. First, we start at the destination tile and work our way outwards using a"
    "    breadth first search. This is called a 'flood fill' and will construct the"
    "    Dijkstra map needed for the vector field.\n"
    "    2. Next, we iterate over each tile and find the neighbour with the lowest"
    "    Dijkstra distance. Using this we can create a vector from the source tile to"
    "    the neighbour tile making for more natural pathfinding since the enemy can go"
    "    in 6 directions instead of 4.\n"
    "    3. Finally, once the neighbour with the lowest Dijkstra distance is found, we"
    "    can create a vector from the current tile to that neighbour tile which the"
    "    enemy will follow. Repeating this for every tile gives us an efficient way to"
    "    calculate pathfinding for a large amount of entities.\n\n"
    "Further reading which may be useful:\n"
    "`Other uses of Dijkstra maps"
    "<http://www.roguebasin.com/index.php/The_Incredible_Power_of_Dijkstra_Maps>`_\n"
    "`Dijkstra maps visualized"
    "<http://www.roguebasin.com/index.php/Dijkstra_Maps_Visualized>`_\n"
    "`Understanding goal based pathfinding"
    "<https://gamedevelopment.tutsplus.com/tutorials/understanding-goal-based-vector/-"
    "field-pathfinding--gamedev-9007>`_\n\n"
    "Parameters\n"
    "----------\n"
    "walls: arcade.SpriteList\n"
    "    A list of wall sprites that can block the entities.\n"
    "width: int\n"
    "    The width of the grid.\n"
    "height: int\n"
    "    The height of the grid.\n\n"
    "Attributes\n"
    "----------\n"
    "vector_dict: dict[tuple[int, int], tuple[float, float]]\n"
    "    A dictionary which holds a tuple containing the tile position and the vector"
    "    the enemy should follow when on that tile."
);

/* Stores the docstring for the VectorField.pixel_to_tile_pos method */
PyDoc_STRVAR(
    pixel_to_tile_pos_docstring,
    "Calculate the tile position from a given screen position.\n\n"
    "Parameters\n"
    "----------\n"
    "position: tuple[float, float]\n"
    "    The sprite position on the screen.\n\n"
    "Raises\n"
    "------\n"
    "    The inputs must be bigger than or equal to 0.\n\n"
    "Returns\n"
    "-------\n"
    "    The path field grid tile position for the given sprite position."
);

/* Stores the docstring for the VectorField.recalculate_map method */
PyDoc_STRVAR(
    recalculate_map_docstring,
    ""
);


// ----- C STRUCTURES ------------------------------
struct FloatPair {
    /* Represents a namedtuple describing a pair of floats */
    float x, y;
};


typedef struct {
    /* Represents a VectorField type and what it contains */
    PyObject base;
    PyObject *vector_dict;
} VectorField;


// ----- C METHODS ------------------------------
IntPair get_tile_pos_for_pixel(FloatPair screen_position) {
    /* Calculates the tile position from a given screen position */
    return {(int) floor(screen_position.x / SPRITE_SIZE), (int) floor(screen_position.y / SPRITE_SIZE)};
}


// ----- VECTORFIELD METHODS ------------------------------
static PyObject *pixel_to_tile_pos(PyObject *self, PyObject *args) {
    /* Parse arguments */
    FloatPair position;
    if (!PyArg_ParseTuple(args, "(ff)", &position.x, &position.y)) {
        return Py_BuildValue("");
    }

    /* Converting logic */
    IntPair tile_pos = get_tile_pos_for_pixel(position);

    /* Return result and do cleanup */
    return Py_BuildValue("(ii)", position.x, position.y);
}


//static PyObject *recalculate_map(PyObject *obj, PyObject *args) {
//    /* Parse arguments */
//    VectorField *self = (VectorField*) obj;
//    FloatPair player_pos;
//    int player_view_distance;
//    if (!PyArg_ParseTuple(args, "(ff)i", &player_pos.x, &player_pos.y, &player_view_distance)) {
//        return Py_BuildValue("");
//    }
//
//    /* Vector field recalculation logic */
//    // To recalculate the map, we need a few things:
//    //      1. A distances dict to store the distances to each tile position from the
//    //      destination tile position. This needs to only include the elements inside
//    //      the walls dict.
//    //      2. A vector_dict dict to store the paths for the vector field. We also need
//    //      to make sure this is empty first.
//    //      3. A queue object, so we can explore the grid.
//    //      4. A possible_spawns list to hold all the grid positions where enemies can
//    //      spawn.
//    PyObject *start_tup = pixel_to_tile_pos(NULL, Py_BuildValue("(ff)", player_pos.x, player_pos.y));
//    self->distances = PyDict_Copy(self->walls_dict);
//    PyDict_SetItem(self->distances, start_tup, Py_BuildValue("i", 0));
//    PyDict_Clear(self->vector_dict);
//    PyObject *possible_spawns = PyList_New(0);
//    std::deque<PyObject*> queue;
//    queue.push_back(start_tup);
//
//    // Explore the grid using a breadth first search (or a flood fill) to generate the
//    // Dijkstra distances
//    while (!queue.empty()) {
//        // Get the current tile to explore
//        PyObject *current = queue.back();
//        queue.pop_back();
//
//        // Get the current tile's neighbours
//        for (PyObject *neighbour : vector_field_grid_bfs(current, self->height, self->width)) {
//            // Check if the neighbour is a wall or not
//            //PyObject *neighbour_tup = Py_BuildValue("(ii)", neighbour.x, neighbour.y);
//            PyObject *distance = PyDict_GetItem(self->distances, neighbour);
//            if (distance == Py_BuildValue("i", std::numeric_limits<int>::infinity())) {
//                continue;
//            }
//
//            // Test if the neighbour has already been reached or not. If it hasn't, add
//            // it to the queue and set its distance
//            if (distance == NULL) {
//                queue.push_back(neighbour);
//                PyDict_SetItem(self->distances, neighbour, 1 + PyDict_GetItem(self->distances, current));
//            }
//        }
//    }
//
//    PyObject_Print(self->distances, stdout, 0);
//    std::cout << "\n";
//
//    /* Return result and do cleanup */
//    return Py_BuildValue("");
//}


// ----- VECTORFIELD MAGIC METHODS ------------------------------
static PyObject *vector_field_repr(VectorField *self) {
    /* Build repr */
    std::string repr = "<VectorField (Width=" + std::to_string(WIDTH) + ") (Height=" + std::to_string(HEIGHT) + ")>";

    /* Return result */
    return Py_BuildValue("s", repr.c_str());
}


static int vector_field_init(VectorField *self, PyObject *args, PyObject *kwds) {
    /* Parse arguments */
    PyObject *walls;
    if (!PyArg_ParseTuple(args, "Oii", &walls, &HEIGHT, &WIDTH)) {
        return -1;
    }

    /* VectorField initialisation logic */
    // Convert each sprite's position in the spritelist to a grid position and store
    // it in walls_dict with infinity as the value
    self->vector_dict = PyDict_New();
    PyObject *sprite_list = PyObject_GetAttrString(walls, "sprite_list");
    Py_DECREF(walls);
    for (int i = 0; i < PyList_Size(sprite_list); i++) {
        PyObject *py_screen_position = PyObject_GetAttrString(PyList_GetItem(sprite_list, i), "position");
        float x = (float) PyFloat_AsDouble(PyTuple_GetItem(py_screen_position, 0));
        float y = (float) PyFloat_AsDouble(PyTuple_GetItem(py_screen_position, 1));
        WALLS_DICT[get_tile_pos_for_pixel({x, y})] = INT_INFINITY;
    }
    Py_DECREF(sprite_list);

    /* Return result */
    return 0;
}


// ----- VECTORFIELD DEFINITIONS ------------------------------
static PyMethodDef vector_field_methods[] = {
    /* Defines the methods which belong to the VectorField type */
    {"pixel_to_tile_pos", pixel_to_tile_pos, METH_VARARGS | METH_STATIC, pixel_to_tile_pos_docstring},
    //{"recalculate_map", recalculate_map, METH_VARARGS},
    {NULL},
};


static PyMemberDef vector_field_members[] = {
    /* Defines the members which belong to the VectorField type */
    {"vector_dict", T_OBJECT_EX, offsetof(VectorField, vector_dict), READONLY},
    {NULL},
};


static PyTypeObject VectorFieldType = {
    /* Describes the VectorField type and all of its slots */
    PyVarObject_HEAD_INIT(NULL, 0)
    "vector_field.VectorField",     // tp_name
    sizeof(VectorField),            // tp_basicsize
    0,                              // tp_itemsize
    0,                              // tp_dealloc
    0,                              // tp_vectorcall_offset
    0,                              // tp_getattr
    0,                              // tp_setattr
    0,                              // tp_as_async
    (reprfunc) vector_field_repr,   // tp_repr
    0,                              // tp_as_number
    0,                              // tp_as_sequence
    0,                              // tp_as_mapping
    0,                              // tp_hash
    0,                              // tp_call
    0,                              // tp_str
    0,                              // tp_getattro
    0,                              // tp_setattro
    0,                              // tp_as_buffer
    Py_TPFLAGS_DEFAULT,             // tp_flags
    vector_field_type_docstring,    // tp_doc
    0,                              // tp_traverse
    0,                              // tp_clear
    0,                              // tp_richcompare
    0,                              // tp_weaklistoffset
    0,                              // tp_iter
    0,                              // tp_iternext
    vector_field_methods,           // tp_methods
    vector_field_members,           // tp_members
    0,                              // tp_getset
    0,                              // tp_base
    0,                              // tp_dict
    0,                              // tp_descr_get
    0,                              // tp_descr_set
    0,                              // tp_dictoffset
    (initproc) vector_field_init,   // tp_init
    0,                              // tp_alloc
    PyType_GenericNew,              // tp_new
};


// ----- MODULE DEFINITION ------------------------------
static struct PyModuleDef vector_field_module = {
    /* Defines the metadata for this extension module */
    PyModuleDef_HEAD_INIT,
    "vector_field",
    vector_field_module_docstring,
    -1,
    NULL,
};


// ----- MODULE CREATION ------------------------------
PyMODINIT_FUNC PyInit_vector_field(void) {
    /* Initialises this module so Python can access it */
        // Initialise the C++ module and check if it's valid or not
    PyObject *module = PyModule_Create(&vector_field_module);
    if (module == NULL)
        return NULL;

    // Check if the VectorField object is ready to be initialised or not
    if (PyType_Ready(&VectorFieldType) < 0)
        return NULL;

    // Initialise the constants
    SPRITE_SIZE = (int) PyFloat_AsDouble(get_global_constant("hades.constants.game_object", {"SPRITE_SIZE"}));

    // Initialise the VectorField object
    Py_INCREF(&VectorFieldType);
    if (PyModule_AddObject(module, "VectorField", (PyObject*) &VectorFieldType) < 0) {
        Py_DECREF(&VectorFieldType);
        Py_DECREF(module);
        return NULL;
    }

    // Return the C++ initialised module
    return module;
}
