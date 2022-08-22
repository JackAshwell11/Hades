#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>


typedef struct Point {
    int x;
    int y;
} Point;

static void print_numpy_array(PyArrayObject *arr) {
    int i, j;
    int* array_data_pointer = (int*)PyArray_DATA(arr);
    int height = (int)PyArray_DIM(arr, 0);
    int width = (int)PyArray_DIM(arr, 1);
    for(i = 0; i < 2; i++) {
        for(j = 0; j < 3; j++) {
            printf("arr pos i=%d, j=%d, val=%d\n", i, j, array_data_pointer[i * width + j]);
        };
    };
}


static int heuristic(struct Point a, struct Point b) {
    return abs(a.x - b.x) + abs(a.y - b.y);
}


static PyObject *calculate_astar_path(PyObject *self, PyObject *args) {
    PyArrayObject *grid = NULL;
    Point start, end;
    if(!PyArg_ParseTuple(args, "O(ii)(ii)", &grid, &start.x, &start.y, &end.x, &end.y)) {
        printf("Error while parsing args\n");
        return NULL;
    }

    printf("%d %d\n", start.x, start.y);
    printf("%d %d\n", end.x, end.y);
    printf("%d\n", heuristic(start, end));
    print_numpy_array(grid);

    return PyLong_FromLong(0);
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
