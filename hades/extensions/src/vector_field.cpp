#include "hades_common.h"
#include <math.h>
#include <structmember.h>
#include <string>
#include <iostream>
#include <ostream>

// The sprite size constant for use in the vector field
int SPRITE_SIZE;


/* Stores the docstring for the vector_field module */
PyDoc_STRVAR(
    vector_field_module_docstring,
    "Creates a vector field useful for navigating enemies around the game map."
);


/* Stores the docstring for the VectorField type */
PyDoc_STRVAR(
    vector_field_type_docstring,
    "test"
);


struct FloatPair {
    /* Represents a namedtuple describing a pair of floats */
    float x, y;
};


typedef struct {
    /* Represents a VectorField type and what it contains */
    PyObject base;
    int width, height;
    PyObject *walls_dict;
    PyObject *distances;
    PyObject *vector_dict;
} VectorField;


static PyObject *pixel_to_tile_pos(PyObject *self, PyObject *args) {
    /* Parse arguments */
    float x, y;
    if (!PyArg_ParseTuple(args, "ff", &x, &y)) {
        return Py_BuildValue("");
    }

    /* Converting logic */
    int a = (int) floor(x / SPRITE_SIZE);
    int b = (int) floor(y / SPRITE_SIZE);

    /* Return result */
    return Py_BuildValue("(ii)", a, b);
}


static PyObject *vector_field_repr(VectorField *self) {
    /* Build repr */
    std::string repr = "<VectorField (Width=" + std::to_string(self->width) + ") (Height=" + std::to_string(self->height) + ")>";

    /* Return result */
    return Py_BuildValue("s", repr.c_str());
}


static int vector_field_init(VectorField *self, PyObject *args, PyObject *kwds) {
    /* Parse arguments */
    PyObject *walls;
    int sprite_size, height, width;
    if (!PyArg_ParseTuple(args, "Oiii", &walls, &sprite_size, &height, &width)) {
        return -1;
    }

    /* VectorField initialisation logic */
    // Set the vector field sprite size, height and width constants
    if (sprite_size && height && width) {
        SPRITE_SIZE = sprite_size;
        self->height = height;
        self->width = width;
    } else {
        return -1;
    }

    // Convert each sprite's position in the spritelist to a grid position and store
    // it in walls_dict with infinity as the value
    self->walls_dict = PyDict_New();
    PyObject *sprite_list = PyObject_GetAttrString(walls, "sprite_list");
    PyObject *infinity = Py_BuildValue("i", std::numeric_limits<int>::max());
    for (int i = 0; i < PyList_Size(sprite_list); i++) {
        PyDict_SetItem(self->walls_dict, pixel_to_tile_pos(NULL, PyObject_GetAttrString(PyList_GetItem(sprite_list, i), "position")), infinity);
    }

    /* Return success */
    return 0;
}


static PyMethodDef vector_field_methods[] = {
    /* Defines the methods which belong to the VectorField type */
    {"pixel_to_tile_pos", pixel_to_tile_pos, METH_VARARGS | METH_STATIC},
    {NULL},
};


static PyMemberDef vector_field_members[] = {
    /* Defines the members which belong to the VectorField type */
    {"width", T_INT, offsetof(VectorField, width), READONLY},
    {"height", T_INT, offsetof(VectorField, height), READONLY},
    {"walls_dict", T_OBJECT_EX, offsetof(VectorField, walls_dict), READONLY},
    {"distances", T_OBJECT_EX, offsetof(VectorField, distances), READONLY},
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


static struct PyModuleDef vector_field_module = {
    /* Defines the metadata for this extension module */
    PyModuleDef_HEAD_INIT,
    "vector_field",
    vector_field_module_docstring,
    -1,
    NULL,
};


PyMODINIT_FUNC PyInit_vector_field(void) {
    /* Initialises this module so Python can access it */
    // Check if the VectorField object is ready to be initialised or not
    if (PyType_Ready(&VectorFieldType) < 0)
        return NULL;

    // Initialise the C++ module and check if it's valid or not
    PyObject *module = PyModule_Create(&vector_field_module);
    if (module == NULL)
        return NULL;

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
