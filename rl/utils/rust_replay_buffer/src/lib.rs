// Cargo.toml dependencies you'll need:
// [dependencies]
// pyo3 = { version = "0.20", features = ["extension-module"] }
// numpy = "0.20"
// ndarray = { version = "0.15", features = ["serde"] }
// rayon = "1.8"

use ndarray::{Array3, Array2, s};
use numpy::{IntoPyArray, PyArray3};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use std::sync::Mutex;

#[pyclass]
pub struct MinimalImgReplayBuffer {
    #[pyo3(get)]
    height: usize,
    #[pyo3(get)]
    width: usize,
    x_min: i32,
    y_min: i32,
    buffer_size: usize,
    batch_size: usize,
    buffer_index: usize,
    full: bool,
    type_to_color_array: Vec<[u8; 3]>,
    state: Vec<Option<Array2<i32>>>,
    next_state: Vec<Option<Array2<i32>>>,
}

#[pymethods]
impl MinimalImgReplayBuffer {
    #[new]
    pub fn new(
        height: usize,
        width: usize,
        x_min: i32,
        y_min: i32,
        buffer_size: usize,
        batch_size: usize,
        type_to_color: &PyDict,
    ) -> Self {
        let mut max_type = 0;
        let mut type_to_color_array = vec![[0, 0, 0]; 256];

        for (key, value) in type_to_color.iter() {
            let key: usize = key.extract().unwrap();
            let color: Vec<u8> = value.extract().unwrap();
            type_to_color_array[key] = [color[0], color[1], color[2]];
            if key > max_type {
                max_type = key;
            }
        }
        type_to_color_array.truncate(max_type + 1);

        MinimalImgReplayBuffer {
            height,
            width,
            x_min,
            y_min,
            buffer_size,
            batch_size,
            buffer_index: 0,
            full: false,
            type_to_color_array,
            state: vec![None; buffer_size],
            next_state: vec![None; buffer_size],
        }
    }

    pub fn add(
        &mut self,
        state: Vec<(i32, i32, i32)>,
        next_state: Vec<(i32, i32, i32)>,
    ) {
        let to_array = |data: Vec<(i32, i32, i32)>| {
            Array2::from_shape_fn((data.len(), 3), |(i, j)| match j {
                0 => data[i].0,
                1 => data[i].1,
                _ => data[i].2,
            })
        };

        self.state[self.buffer_index] = Some(to_array(state));
        self.next_state[self.buffer_index] = Some(to_array(next_state));

        self.buffer_index = (self.buffer_index + 1) % self.buffer_size;
        if self.buffer_index == 0 {
            self.full = true;
        }
    }

    pub fn reconstruct_image<'py>(
        &self,
        py: Python<'py>,
        idx: usize,
    ) -> &'py PyArray3<u8> {
        let array = self.state[idx].as_ref().unwrap();

        let mut output = Array3::<u8>::zeros((3, self.height, self.width));

        array.axis_iter(ndarray::Axis(0)).for_each(|row| {
            let x = (row[0] - self.x_min) as usize;
            let y = (row[1] - self.y_min) as usize;
            let t = row[2] as usize;
            if x < self.height && y < self.width && t < self.type_to_color_array.len() {
                let [r, g, b] = self.type_to_color_array[t];
                output[[0, x, y]] = r;
                output[[1, x, y]] = g;
                output[[2, x, y]] = b;
            }
        });

        output.into_pyarray(py)
    }
}

#[pymodule]
fn rust_replay_buffer(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<MinimalImgReplayBuffer>()?;
    Ok(())
}
