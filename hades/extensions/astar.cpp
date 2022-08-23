#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <iostream>
#include <list>
#include <unordered_map>
#include <queue>
#include <limits>
#include <Python.h>
#include <numpy/arrayobject.h>

using namespace std;

template <class T>
inline void hash_combine(size_t& seed, const T& v) {
    /* Allows multiple hashes to be combined for a struct */
    hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
}


struct Point {
    int x, y;

    bool operator==(const Point pnt) const {
        // If two hashes are the same, we need to check if the two structs are
        // the same
        return x == pnt.x && y == pnt.y;
    }
};


Point CARDINAL_OFFSETS[4] = {
        {0, -1},
        {-1, 0},
        {1, 0},
        {0, 1}
};


struct Neighbour {
    int cost;
    Point point;

    bool operator<(const Neighbour nghbr) const {
        // The priority_queue data structure gets the maximum priority, so we
        // need to override that functionality to get the minimum priority
        return cost > nghbr.cost;
    }
};


template<>
struct hash<Point> {
    /* Allows the Point struct to be hashed in a map */
    size_t operator()(const Point& pnt) const {
        size_t res = 0;
        hash_combine(res, pnt.x);
        hash_combine(res, pnt.y);
        return res;
    }
};


void print_numpy_array(PyArrayObject *arr) {
    int i, j;
    int* array_data_pointer = (int*)PyArray_DATA(arr);
    int height = (int)PyArray_DIM(arr, 0);
    int width = (int)PyArray_DIM(arr, 1);
    for(i = 0; i < 2; i++) {
        for(j = 0; j < 3; j++) {
            printf("arr pos i=%d, j=%d, val=%d\n", i, j, array_data_pointer[i * width + j]);
        }
    }
}


static int heuristic(Point a, Point b) {
    /* Calculates the heuristic used for the A* algorithm */
    return abs(a.x - b.x) + abs(a.y - b.y);
}


static list<Point> grid_bfs(Point target, int height, int width, Point* offsets) {
    /* Gets a target's neighbours in a grid */
    list<Point> result;
    for (int i = 0; i < 4; i++) {
        int x = target.x + offsets[i].x;
        int y = target.y + offsets[i].y;
        if ((x >= 0 && x < width) && (y >= 0 && y < height)) {
            result.push_back({x, y});
        }
    }
    return result;
}


static PyObject *calculate_astar_path(PyObject *self, PyObject *args) {
    /* Parse arguments */
    PyArrayObject *grid = NULL;
    struct Point start, end;
    int obstacle_id = NULL;
    if (!PyArg_ParseTuple(args, "O(ii)(ii)i", &grid, &start.x, &start.y, &end.x, &end.y, &obstacle_id)) {
        printf("Error while parsing args\n");
        return NULL;
    }

    /* A* algorithm logic */
    // Set up a few variables needed for the pathfinding
    PyObject *result = PyList_New(0);
    priority_queue<Neighbour> queue;
    queue.push({0, start});
    unordered_map<Point, Point> came_from = {{start, start}};
    unordered_map<Point, int> distances = {{start, 0}};
    unordered_map<Point, int> total_costs = {{start, 0}};
    int height = (int)PyArray_DIM(grid, 0);
    int width = (int)PyArray_DIM(grid, 1);
    int* array_data_pointer = (int*)PyArray_DATA(grid);

    // Loop until the priority queue is empty
    while (!queue.empty()) {
        // Get the lowest cost point from the priority queue
        Neighbour current_f = queue.top();
        queue.pop();
        Point current = current_f.point;

        // Check if we've reached our target
        if (current == end) {
            // Backtrack through came_from to get the path
            while (!(came_from.at(current) == current)) {
                // Add the current point to the result list
                PyList_Append(result, Py_BuildValue("(ii)", current.x, current.y));

                // Get the next point in the path
                current = came_from.at(current);
            }

            // Add the start point and exit out of the loop
            PyList_Append(result, Py_BuildValue("(ii)", start.x, start.y));
            break;
        }

        // Add all the neighbours to the heap with their cost being f = g + h:
        //   f - The total cost of traversing the neighbour.
        //   g - The distance between the start point and the neighbour point.
        //   h - The estimated distance from the neighbour point to the end
        //   point.
        for (Point neighbour : grid_bfs(current, height, width, CARDINAL_OFFSETS)) {
            if (!came_from.count(neighbour)) {
                // Store the neighbour's parent and calculate its distance from
                // the start point
                came_from[neighbour] = current;
                distances[neighbour] = distances.at(came_from.at(neighbour)) + 1;

                // Check if the neighbour is an obstacle
                if (array_data_pointer[neighbour.y*width + neighbour.x] == obstacle_id) {
                    // Set the total cost for the obstacle to infinity
                    total_costs[neighbour] = numeric_limits<int>::max();
                } else {
                    // Set the total cost for the neighbour to f = g + h
                    total_costs[neighbour] = distances.at(neighbour) + heuristic(current, neighbour);
                }

                // Add the neighbour to the priority queue
                queue.push({total_costs.at(neighbour), neighbour});
            }
        }
    }

    /* Return result */
    return result;
}


static PyMethodDef astarmethods[] = {
    {"calculate_astar_path", calculate_astar_path, METH_VARARGS, "Calculate the shortest path in a grid from one point to another using the A* algorithm."},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef astarmodule = {
    PyModuleDef_HEAD_INIT,
    "astar",
    "Calculate the shortest path in a grid from one point to another using the A* algorithm.",
    -1,
    astarmethods,
};

PyMODINIT_FUNC PyInit_astar(void) {
    PyObject *module = PyModule_Create(&astarmodule);
    import_array();
    return module;
}
