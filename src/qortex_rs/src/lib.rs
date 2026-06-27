//! qortex_rs — Rust-accelerated real-time EEG windowing primitives.
//!
//! Provides a lock-free ring buffer for single-producer / single-consumer
//! real-time EEG windowing, plus a batch windowing helper.
//!
//! # Building
//!
//! ```bash
//! cd src/qortex_rs
//! maturin develop --release
//! ```
//!
//! # Python usage
//!
//! ```python
//! from qortex_rs import RingBuffer, batch_window
//! import numpy as np
//!
//! buf = RingBuffer(n_channels=64, capacity=8192, window_size=512, step_size=128)
//! chunk = np.random.randn(64, 256).astype(np.float32)
//! buf.push(chunk)
//! win = buf.pop_window()   # ndarray[64, 512] or None
//! ```

use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Lock-free ring buffer for real-time EEG windowing.
///
/// Thread-safe for single-producer, single-consumer (SPSC) use.
/// The buffer accumulates incoming samples and yields windows of fixed size
/// with configurable step (hop) size.
#[pyclass]
struct RingBuffer {
    /// Sample data: data[channel][sample_index % capacity]
    data: Vec<Vec<f32>>,
    n_channels: usize,
    capacity: usize,
    write_pos: AtomicUsize,
    window_size: usize,
    step_size: usize,
    n_buffered: usize,
}

#[pymethods]
impl RingBuffer {
    /// Create a new ring buffer.
    ///
    /// Parameters
    /// ----------
    /// n_channels : int
    ///     Number of signal channels.
    /// capacity : int
    ///     Ring buffer capacity in samples (must be >= window_size).
    /// window_size : int
    ///     Samples per output window.
    /// step_size : int
    ///     Hop size between windows (step_size == window_size → non-overlapping).
    #[new]
    fn new(n_channels: usize, capacity: usize, window_size: usize, step_size: usize) -> PyResult<Self> {
        if capacity < window_size {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("capacity ({}) must be >= window_size ({})", capacity, window_size)
            ));
        }
        if step_size == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "step_size must be > 0"
            ));
        }
        Ok(RingBuffer {
            data: vec![vec![0.0_f32; capacity]; n_channels],
            n_channels,
            capacity,
            write_pos: AtomicUsize::new(0),
            window_size,
            step_size,
            n_buffered: 0,
        })
    }

    /// Push a chunk of samples into the buffer.
    ///
    /// Parameters
    /// ----------
    /// samples : np.ndarray[float32]
    ///     Array of shape ``[n_channels, n_samples]``.
    fn push(&mut self, samples: PyReadonlyArray2<f32>) -> PyResult<()> {
        let arr = samples.as_array();
        let shape = arr.shape();
        if shape[0] != self.n_channels {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Expected {} channels, got {}",
                self.n_channels, shape[0]
            )));
        }
        let n_samples = shape[1];

        for s in 0..n_samples {
            let pos = self.write_pos.load(Ordering::Relaxed) % self.capacity;
            for ch in 0..self.n_channels {
                self.data[ch][pos] = arr[[ch, s]];
            }
            self.write_pos.fetch_add(1, Ordering::Release);
        }
        self.n_buffered += n_samples;
        Ok(())
    }

    /// Try to pop one window from the buffer.
    ///
    /// Returns ``None`` if fewer than ``window_size`` samples are buffered.
    ///
    /// Returns
    /// -------
    /// np.ndarray[float32] or None
    ///     Shape ``[n_channels, window_size]``, or ``None``.
    fn pop_window<'py>(&mut self, py: Python<'py>) -> PyResult<Option<&'py PyArray2<f32>>> {
        if self.n_buffered < self.window_size {
            return Ok(None);
        }

        let write = self.write_pos.load(Ordering::Acquire);
        // Oldest unconsumed sample sits at (write - n_buffered) in absolute space
        let read = write - self.n_buffered;

        let mut flat = vec![0.0_f32; self.n_channels * self.window_size];
        for s in 0..self.window_size {
            let pos = (read + s) % self.capacity;
            for ch in 0..self.n_channels {
                flat[ch * self.window_size + s] = self.data[ch][pos];
            }
        }

        self.n_buffered = self.n_buffered.saturating_sub(self.step_size);

        let arr = numpy::ndarray::Array2::from_shape_vec(
            (self.n_channels, self.window_size),
            flat,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Ok(Some(arr.into_pyarray(py)))
    }

    /// Number of unconsumed samples currently in the buffer.
    fn n_available(&self) -> usize {
        self.n_buffered
    }

    /// Clear the buffer and reset all cursors.
    fn reset(&mut self) {
        self.write_pos.store(0, Ordering::Release);
        self.n_buffered = 0;
        for ch in &mut self.data {
            ch.fill(0.0);
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RingBuffer(n_channels={}, capacity={}, window_size={}, step_size={}, n_buffered={})",
            self.n_channels, self.capacity, self.window_size, self.step_size, self.n_buffered,
        )
    }
}

/// Split a ``[n_channels, n_samples]`` array into overlapping windows.
///
/// Parameters
/// ----------
/// data : np.ndarray[float32]
///     Input array of shape ``[n_channels, n_samples]``.
/// window_size : int
///     Samples per window.
/// step_size : int
///     Hop size between windows.
///
/// Returns
/// -------
/// list of np.ndarray[float32]
///     Each element has shape ``[n_channels, window_size]``.
#[pyfunction]
fn batch_window<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<f32>,
    window_size: usize,
    step_size: usize,
) -> PyResult<Vec<&'py PyArray2<f32>>> {
    if step_size == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("step_size must be > 0"));
    }
    let arr = data.as_array();
    let n_channels = arr.shape()[0];
    let n_samples = arr.shape()[1];

    let mut windows = Vec::new();
    let mut start = 0usize;

    while start + window_size <= n_samples {
        let mut flat = vec![0.0_f32; n_channels * window_size];
        for ch in 0..n_channels {
            for s in 0..window_size {
                flat[ch * window_size + s] = arr[[ch, start + s]];
            }
        }
        let win_arr = numpy::ndarray::Array2::from_shape_vec((n_channels, window_size), flat)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        windows.push(win_arr.into_pyarray(py));
        start += step_size;
    }

    Ok(windows)
}

/// Register the Python module.
#[pymodule]
fn qortex_rs(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<RingBuffer>()?;
    m.add_function(wrap_pyfunction!(batch_window, m)?)?;
    Ok(())
}
