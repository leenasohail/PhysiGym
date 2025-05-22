// Cargo.toml dependencies you'll need:
// [dependencies]
// pyo3 = { version = "0.20", features = ["extension-module"] }
// numpy = "0.20"
// ndarray = { version = "0.15", features = ["serde"] }
// rayon = "1.8"
// https://chatgpt.com/c/682edcb0-ecb4-8007-a5fa-595b441283b9 rust replay buffer in rust optimized 
use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

#[pyclass]
pub struct ReplayBuffer {
    buffer_size: usize,
    index: usize,
    full: bool,
    state: Vec<Option<Vec<[i32; 3]>>>,
    next_state: Vec<Option<Vec<[i32; 3]>>>,
    action: Vec<[f32; 4]>,
    reward: Vec<f32>,
    done: Vec<bool>,
    height: usize,
    width: usize,
    x_min: i32,
    y_min: i32,
    type_to_color: Vec<[u8; 3]>,
    image_gray: bool,
}

#[pymethods]
impl ReplayBuffer {
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        buffer_size: usize,
        height: usize,
        width: usize,
        x_min: i32,
        y_min: i32,
        type_to_color: Vec<(usize, [u8; 3])>,
        image_gray: bool,
    ) -> Self {
        let mut color_array = vec![[0, 0, 0]; 256];
        for (idx, color) in type_to_color {
            if idx < color_array.len() {
                color_array[idx] = color;
            }
        }

        Self {
            buffer_size,
            index: 0,
            full: false,
            state: vec![None; buffer_size],
            next_state: vec![None; buffer_size],
            action: vec![[0.0; 4]; buffer_size],
            reward: vec![0.0; buffer_size],
            done: vec![false; buffer_size],
            height,
            width,
            x_min,
            y_min,
            type_to_color: color_array,
            image_gray,
        }
    }

    pub fn add(
        &mut self,
        state: Vec<[i32; 3]>,
        action: [f32; 4],
        reward: f32,
        next_state: Vec<[i32; 3]>,
        done: bool,
    ) {
        self.state[self.index] = Some(state);
        self.next_state[self.index] = Some(next_state);
        self.action[self.index] = action;
        self.reward[self.index] = reward;
        self.done[self.index] = done;
        self.index = (self.index + 1) % self.buffer_size;
        if self.index == 0 {
            self.full = true;
        }
    }

    pub fn len(&self) -> usize {
        if self.full {
            self.buffer_size
        } else {
            self.index
        }
    }

    pub fn sample(&self, py: Python, batch_size: usize) -> PyResult<PyObject> {
        use rand::seq::SliceRandom;
        use rand::thread_rng;

        let mut rng = thread_rng();
        let range = if self.full {
            0..self.buffer_size
        } else {
            0..self.index
        };

        let indices: Vec<usize> = range.clone().collect::<Vec<_>>().choose_multiple(&mut rng, batch_size).cloned().collect();

        let state_imgs: Vec<Vec<u8>> = indices
            .par_iter()
            .map(|&i| {
                let state = self.state[i].as_ref().unwrap();
                self.render_image(state)
            })
            .collect();

        let next_state_imgs: Vec<Vec<u8>> = indices
            .par_iter()
            .map(|&i| {
                let next_state = self.next_state[i].as_ref().unwrap();
                self.render_image(next_state)
            })
            .collect();

        let action: Vec<[f32; 4]> = indices.iter().map(|&i| self.action[i]).collect();
        let reward: Vec<f32> = indices.iter().map(|&i| self.reward[i]).collect();
        let done: Vec<bool> = indices.iter().map(|&i| self.done[i]).collect();

        let py_dict = PyDict::new(py);
        py_dict.set_item("state", state_imgs)?;
        py_dict.set_item("next_state", next_state_imgs)?;
        py_dict.set_item("action", action)?;
        py_dict.set_item("reward", reward)?;
        py_dict.set_item("done", done)?;

        Ok(py_dict.into())
    }
}

impl ReplayBuffer {
    fn render_image(&self, data: &[[i32; 3]]) -> Vec<u8> {
        let mut image = if self.image_gray {
            vec![0u8; self.height * self.width]
        } else {
            vec![0u8; 3 * self.height * self.width]
        };

        for [x, y, t] in data {
            let x = x - self.x_min;
            let y = y - self.y_min;
            if *x >= 0 && *x < self.height as i32 && *y >= 0 && *y < self.width as i32 {
                let idx = (*x as usize) * self.width + (*y as usize);
                let color = self.type_to_color[*t as usize];
                if self.image_gray {
                    let gray = (0.2989 * color[0] as f64
                        + 0.5870 * color[1] as f64
                        + 0.1140 * color[2] as f64) as u8;
                    image[idx] = gray;
                } else {
                    image[idx] = color[0];
                    image[idx + self.height * self.width] = color[1];
                    image[idx + 2 * self.height * self.width] = color[2];
                }
            }
        }

        image
    }
}

#[pymodule]
fn rust_replay_buffer(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ReplayBuffer>()?;
    Ok(())
}
