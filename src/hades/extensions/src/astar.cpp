// Definitions
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

// Includes
#include "hades_common.h"
#include <numpy/arrayobject.h>
#include <queue>
#include <iostream>


// ----- CONSTANTS ------------------------------
/* The ID of a TileType obstacle */
int OBSTACLE_ID;


// ----- DOCSTRINGS ------------------------------
/* Stores the docstring for the astar module */
PyDoc_STRVAR(
    astar_module_docstring,
    "Calculate the shortest path in a grid from one pair to another using the A* "
    "algorithm.\n\n"
);


/* Stores the docstring for the calculate_astar_path method */
PyDoc_STRVAR(
    astar_docstring,
    "Calculate the shortest path in a grid from one pair to another using the A* "
    "algorithm.\n\n"
    "Further reading which may be useful:\n"
    "`The A* algorithm <https://en.wikipedia.org/wiki/A*_search_algorithm>`_\n\n"
    "Parameters\n"
    "----------\n"
    "grid: np.ndarray\n"
    "   The 2D grid which represents the dungeon.\n"
    "start: Pair\n"
    "    The start pair for the algorithm.\n"
    "end: Pair\n"
    "    The end pair for the algorithm.\n\n"
    "Returns\n"
    "-------\n"
    "list[Pair]\n"
    "    A list of pairs mapping out the shortest path from start to end."
);


// ----- STRUCTURES ------------------------------
struct Neighbour {
    /* Represents a namedtuple describing a grid pair and its cost from the start
    position */
    int cost;
    IntPair pair;

    inline bool operator<(const Neighbour nghbr) const {
        // The priority_queue data structure gets the maximum priority, so we need to
        // override that functionality to get the minimum priority
        return cost > nghbr.cost;
    }
};


// ----- PYCFUNCTIONS ------------------------------
static PyObject *calculate_astar_path(PyObject *self, PyObject *args) {
    /* Parse arguments */
    PyArrayObject *grid;
    struct IntPair start, end;
    if (!PyArg_ParseTuple(args, "O(ii)(ii)", &grid, &start.x, &start.y, &end.x, &end.y)) {
        Py_RETURN_NONE;
    }

    // Do some validation checking on the given parameters
    if (PyArray_Check(grid) == 0) {
        Py_RETURN_NONE;
    }

    /* A* algorithm logic */
    // Set up a few variables needed for the pathfinding
    PyObject *result = PyList_New(0);
    std::priority_queue<Neighbour> queue;
    queue.push({0, start});
    std::unordered_map<IntPair, IntPair> came_from = {{start, start}};
    std::unordered_map<IntPair, int> distances = {{start, 0}};
    int height = (int) PyArray_DIM(grid, 0);
    int width = (int) PyArray_DIM(grid, 1);

    // Loop until the priority queue is empty
    while (!queue.empty()) {
        // Get the lowest cost pair from the priority queue
        IntPair current = queue.top().pair;
        queue.pop();

        // Check if we've reached our target
        if (current == end) {
            // Backtrack through came_from to get the path
            while (!(came_from[current] == current)) {
                // Add the current pair to the result list
                PyList_Append(result, Py_BuildValue("(ii)", current.x, current.y));

                // Get the next pair in the path
                current = came_from[current];
            }

            // Add the start pair and exit out of the loop
            PyList_Append(result, Py_BuildValue("(ii)", start.x, start.y));
            break;
        }

        // Add all the neighbours to the heap with their cost being f = g + h:
        //   f - The total cost of traversing the neighbour.
        //   g - The distance between the start pair and the neighbour pair.
        //   h - The estimated distance from the neighbour pair to the end pair. We're using the Manhattan distance for
        //       this.
        for (IntPair neighbour: grid_bfs(current, height, width)) {
            if (!came_from.count(neighbour)) {
                // Store the neighbour's parent and calculate its distance from the
                // start pair
                came_from[neighbour] = current;
                distances[neighbour] = distances[current] + 1;

                // Check if the neighbour is an obstacle
                int f_cost;
                if (*((int *) PyArray_GETPTR2(grid, neighbour.y, neighbour.x)) == OBSTACLE_ID) {
                    // Set the total cost for the obstacle to infinity
                    f_cost = INT_INFINITY;
                } else {
                    // Set the total cost for the neighbour to f = g + h
                    f_cost = distances[neighbour] + (abs(current.x - neighbour.x) + abs(current.y - neighbour.y));
                }

                // Add the neighbour to the priority queue
                queue.push({f_cost, neighbour});
            }
        }
    }

    /* Return result */
    return result;
}


// ----- METHOD DEFINITIONS ------------------------------
static PyMethodDef astar_methods[] = {
    /* Defines the metadata for methods accessible through Python */
    {"calculate_astar_path", calculate_astar_path, METH_VARARGS, astar_docstring},
    {NULL},
};


// ----- MODULE DEFINITION ------------------------------
static struct PyModuleDef astar_module = {
    /* Defines the metadata for this extension module */
    PyModuleDef_HEAD_INIT,
    "astar",
    astar_module_docstring,
    -1,
    astar_methods,
};


// ----- MODULE CREATION ------------------------------
PyMODINIT_FUNC PyInit_astar(void) {
    /* Initialises this module so Python can access it */
    // Initialise the C++ module and check if it's valid or not
    PyObject *module = PyModule_Create(&astar_module);
    if (module == NULL)
        return NULL;

    // Initialise the constants
    PyObject *temp_obstacle_id = get_global_constant("hades.constants.generation", {"TileType", "OBSTACLE", "value"});
    if (temp_obstacle_id == Py_None) {
        return NULL;
    } else {
        OBSTACLE_ID = (int) PyLong_AsLong(temp_obstacle_id);
    }
    Py_DECREF(temp_obstacle_id);

    // Initialise the Numpy C-API
    import_array();

    // Return the C++ initialised module
    return module;
}
