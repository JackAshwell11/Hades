extern crate core;

use pyo3::prelude::*;
mod generation;

/// Generates the dungeon and places game objects in it.
#[pymodule]
fn hades_extensions_rust(_py: Python, module: &PyModule) -> PyResult<()> {
    // Add the entry function
    module.add_function(wrap_pyfunction!(generation::map::create_map, module)?)?;

    // Initialisation complete
    Ok(())
}
