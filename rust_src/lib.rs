use pyo3::prelude::*;
mod generation;

/// Generates the dungeon and places game objects in it.
#[pymodule]
fn hades_extensions(_py: Python, module: &PyModule) -> PyResult<()> {
    // Import the needed generations constants and store them
    // Python::with_gil(|py| {
    //     if let Ok(generation_constants) = PyModule::import(py, "hades.constants.generation") {
    //         println!("{:?}", generation_constants.dict().get_item("TileType")?.get_item("OBSTACLE")?.get_item("value"));
    //     } else {
    //         panic!("Error importing generation constants");
    //     }
    // });

    // Add all the classes
    module.add_class::<generation::primitives::Point>()?;
    module.add_class::<generation::primitives::Rect>()?;

    // Add all the methods
    module.add_function(wrap_pyfunction!(
        generation::astar::calculate_astar_path,
        module
    )?)?;

    // Initialisation complete
    Ok(())
}
