#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <limits>
#include <queue>
#include <unordered_map>
#include <vector>
#include "hades_common.h"


struct Neighbour {
    /* Represents a namedtuple describing a grid pair and its cost from the start
    position */
    int cost;
    Pair pair;

    inline bool operator<(const Neighbour nghbr) const {
        // The priority_queue data structure gets the maximum priority, so we need to
        // override that functionality to get the minimum priority
        return cost > nghbr.cost;
    }
};


inline int calculate_heuristic(Pair a, Pair b) {
    /* Calculates the heuristic used for the A* algorithm */
    return abs(a.x - b.x) + abs(a.y - b.y);
}


static PyObject *heuristic(PyObject *self, PyObject *args) {
    /* Parse arguments */
    struct Pair a, b;
    if (!PyArg_ParseTuple(args, "(ii)(ii)", &a.x, &a.y, &b.x, &b.y)) {
        return NULL;
    }

    /* Heuristic logic */
    int result = calculate_heuristic(a, b);

    /* Return result */
    return PyLong_FromLong(result);
}


static PyObject *calculate_astar_path(PyObject *self, PyObject *args) {
    /* Parse arguments */
    PyArrayObject *grid = NULL;
    struct Pair start, end;
    int obstacle_id = NULL;
    if (!PyArg_ParseTuple(args, "O(ii)(ii)i", &grid, &start.x, &start.y, &end.x, &end.y, &obstacle_id)) {
        return NULL;
    }

    /* A* algorithm logic */
    // Set up a few variables needed for the pathfinding
    int f_cost;
    PyObject *result = PyList_New(0);
    std::priority_queue<Neighbour> queue;
    queue.push({0, start});
    std::unordered_map<Pair, Pair> came_from = {{start, start}};
    std::unordered_map<Pair, int> distances = {{start, 0}};
    int height = (int)PyArray_DIM(grid, 0);
    int width = (int)PyArray_DIM(grid, 1);
    int* array_data_pointer = (int*)PyArray_DATA(grid);

    // Loop until the priority queue is empty
    while (!queue.empty()) {
        // Get the lowest cost pair from the priority queue
        Neighbour current_f = queue.top();
        queue.pop();
        Pair current = current_f.pair;

        // Check if we've reached our target
        if (current == end) {
            // Backtrack through came_from to get the path
            while (!(came_from.at(current) == current)) {
                // Add the current pair to the result list
                PyList_Append(result, Py_BuildValue("(ii)", current.x, current.y));

                // Get the next pair in the path
                current = came_from.at(current);
            }

            // Add the start pair and exit out of the loop
            PyList_Append(result, Py_BuildValue("(ii)", start.x, start.y));
            break;
        }

        // Add all the neighbours to the heap with their cost being f = g + h:
        //   f - The total cost of traversing the neighbour.
        //   g - The distance between the start pair and the neighbour pair.
        //   h - The estimated distance from the neighbour pair to the end pair.
        for (Pair neighbour : grid_bfs(current, height, width)) {
            if (!came_from.count(neighbour)) {
                // Store the neighbour's parent and calculate its distance from the
                // start pair
                came_from[neighbour] = current;
                distances[neighbour] = distances.at(came_from.at(neighbour)) + 1;

                // Check if the neighbour is an obstacle
                if (array_data_pointer[neighbour.y*width + neighbour.x] == obstacle_id) {
                    // Set the total cost for the obstacle to infinity
                    f_cost = std::numeric_limits<int>::max();
                } else {
                    // Set the total cost for the neighbour to f = g + h
                    f_cost = distances.at(neighbour) + calculate_heuristic(current, neighbour);
                }

                // Add the neighbour to the priority queue
                queue.push({f_cost, neighbour});
            }
        }
    }

    /* Return result */
    return result;
}


PyDoc_STRVAR(
    astar_module_docstring,
    "Calculate the shortest path in a grid from one pair to another using the A* "
    "algorithm.\n\n"
);


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


PyDoc_STRVAR(
    heuristic_docstring,
    "Calculate the Manhattan distance between two pairs.\n\n"
    "This preferable to the Euclidean distance since we can generate staircase-like "
    "paths instead of straight line paths.\n\n"
    "Further reading which may be useful:\n"
    "`Manhattan distance <https://en.wikipedia.org/wiki/Taxicab_geometry>`_\n"
    "`Euclidean distance <https://en.wikipedia.org/wiki/Euclidean_distance>`_\n\n"
    "Parameters\n"
    "----------\n"
    "a: Pair\n"
    "    The first pair.\n"
    "b: Pair\n"
    "    The second pair.\n\n"
    "Returns\n"
    "-------\n"
    "int\n"
    "    The heuristic distance."
);


static PyMethodDef astarmethods[] = {
    /* Defines the metadata for methods accessible through Python */
    {"calculate_astar_path", calculate_astar_path, METH_VARARGS, astar_docstring},
    {"heuristic", heuristic, METH_VARARGS, heuristic_docstring},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef astarmodule = {
    /* Defines the metadata for this extension module */
    PyModuleDef_HEAD_INIT,
    "astar",
    astar_module_docstring,
    -1,
    astarmethods,
};


PyMODINIT_FUNC PyInit_astar(void) {
    /* Initialises this module so Python can access it */
    PyObject *module = PyModule_Create(&astarmodule);
    import_array();
    return module;
}
