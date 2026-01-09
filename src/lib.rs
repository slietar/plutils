mod functions;

use polars::prelude::*;
use pyo3::types::{PyModule, PyModuleMethods as _};
use pyo3::{Bound, PyResult, Python, pyfunction, pymodule};
use pyo3_polars::PySeries;
use pyo3_polars::error::PyPolarsErr;
use pyo3_polars::derive::polars_expr;


#[pyfunction]
fn get_offsets(series: PySeries) -> PyResult<PySeries> {
    Ok(
        PySeries(
            functions::get_offsets(&series.into()).map_err(PyPolarsErr::from)?
        )
    )
}

#[pyfunction]
fn implode_like(
    target_series: PySeries,
    layout_series: PySeries,
) -> PyResult<PySeries> {
    Ok(
        PySeries(
            functions::implode_like(
                &target_series.into(),
                &layout_series.into(),
            ).map_err(PyPolarsErr::from)?
        )
    )
}


#[pymodule]
fn plutils(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(get_offsets, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(implode_like, m)?)?;
    Ok(())
}


fn implode_like_output(input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(input_fields[0].clone())
}

#[polars_expr(output_type_func=implode_like_output)]
fn implode_like_expr(inputs: &[Series]) -> PolarsResult<Series>{
    functions::implode_like(&inputs[0], &inputs[1])
}
