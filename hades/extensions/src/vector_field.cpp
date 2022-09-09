#define PY_SSIZE_T_CLEAN
#include <Python.h>


PyDoc_STRVAR(
    vector_field_module_docstring,
    "Creates a vector field useful for navigating enemies around the game map."
);


static PyMethodDef vector_field_methods[] = {
    /* Defines the metadata for methods accessible through Python */
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef vector_field_module = {
    /* Defines the metadata for this extension module */
    PyModuleDef_HEAD_INIT,
    "vector_field",
    vector_field_module_docstring,
    -1,
    vector_field_methods,
};


PyMODINIT_FUNC PyInit_vector_field(void) {
    /* Initialises this module so Python can access it */
    return PyModule_Create(&vector_field_module);
}
