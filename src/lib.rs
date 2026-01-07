use polars::prelude::*;
use pyo3_polars::{derive::polars_expr, export::{polars_arrow::array::ListArray, polars_core::utils::align_chunks_binary}};
use std::fmt::Write;


fn pig_latin_str(value: &str, output: &mut String) {
    if let Some(first_char) = value.chars().next() {
        write!(output, "{}{}ay", &value[1..], first_char).unwrap()
    }
}

#[polars_expr(output_type=String)]
fn pig_latinnify(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out: StringChunked = ca.apply_into_string_amortized(pig_latin_str);
    Ok(out.into_series())
    // x
}

// #[pyo3::pyfunction]
#[polars_expr(output_type=Float32)]
fn implode_like(inputs: &[Series]) -> PolarsResult<Series>{
    let target_list = inputs[0].f32()?;
    let layout_list = inputs[1].list()?;

    let (lhs, rhs) = align_chunks_binary(target_list, layout_list);

    let offsets = layout_list.offsets()?;
    let offsets_buffer = offsets.buffer();

    let start = offsets[0] as usize;
    let len = offsets[offsets.len() - 1] as usize - start;

    let target_chunks = target_list.rechunk().into_owned();
    let target_array = &target_chunks.chunks()[0];

    let new_array = ListArray::new(
        // target_array.dtype().clone(),
        ArrowDataType::LargeList(
            Box::new(polars_arrow::datatypes::Field::new(
                "item".into(),
                ArrowDataType::Float32,
                true,
            ))
        ),
        offsets,
        target_array.clone(),
        None,
    );

    let new_chunked = unsafe { ListChunked::from_chunks("foo".into(), vec![Box::new(new_array)]) };
    let new_series = new_chunked.into_series();

    Ok(new_series)

    // ChunkedArray::<ListType>::from
    // let _ = unsafe {
    //     ListChunked::from_chunks("Foo", vec![
    //         Box::new(0)
    //     ]);
    // }

    // ChunkedArray::set_fast_explode_list(&mut self, value);

    // Create new series with data from inputs[0] and offsets from inputs[1]

    // align_chunks_binary(target_list, offsets)

    // let x = &lhs.chunks()[0];
    // x.off

            // .zip(rhs.iter())
            // .map(|(lhs, rhs)| match (lhs, rhs) {
            //     (Some(lhs), Some(rhs)) => f(&lhs, &rhs),
            //     _ => None,
            // });
}
