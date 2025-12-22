use std::fmt;
use serde::{Deserialize, Serialize};

// Simple Linear Congruential Generator for deterministic randomness without dependencies
pub struct Lcg {
    state: u64,
}

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_f32(&mut self) -> f32 {
        // Constants from Numerical Recipes
        self.state = self.state.wrapping_mul(1664525).wrapping_add(1013904223);
        // Normalize to [0, 1]
        (self.state as f32) / (u64::MAX as f32)
    }

    // Gaussian approximation using Box-Muller or Central Limit Thm (sum of 12 uniforms - 6)
    pub fn next_gaussian(&mut self) -> f32 {
        let mut sum = 0.0;
        for _ in 0..12 {
            sum += self.next_f32();
        }
        sum - 6.0
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Vector {
    pub data: Vec<f32>,
}

impl Vector {
    pub fn new(size: usize) -> Self {
        Self { data: vec![0.0; size] }
    }

    pub fn from_vec(data: Vec<f32>) -> Self {
        Self { data }
    }

    pub fn zeros(size: usize) -> Self {
        Self { data: vec![0.0; size] }
    }

    pub fn random(size: usize, rng: &mut Lcg) -> Self {
        let data = (0..size).map(|_| rng.next_gaussian() * 0.1).collect();
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn add(&self, other: &Vector) -> Vector {
        assert_eq!(self.len(), other.len());
        let data = self.data.iter().zip(&other.data).map(|(a, b)| a + b).collect();
        Vector { data }
    }

    pub fn sub(&self, other: &Vector) -> Vector {
        assert_eq!(self.len(), other.len());
        let data = self.data.iter().zip(&other.data).map(|(a, b)| a - b).collect();
        Vector { data }
    }

    pub fn scale(&self, scalar: f32) -> Vector {
        let data = self.data.iter().map(|a| a * scalar).collect();
        Vector { data }
    }

    pub fn dot(&self, other: &Vector) -> f32 {
        assert_eq!(self.len(), other.len());
        self.data.iter().zip(&other.data).map(|(a, b)| a * b).sum()
    }

    pub fn norm(&self) -> f32 {
        self.dot(self).sqrt()
    }

    pub fn cosine_similarity(&self, other: &Vector) -> f32 {
        let n1 = self.norm();
        let n2 = other.norm();
        if n1 < 1e-6 || n2 < 1e-6 {
            return 0.0;
        }
        self.dot(other) / (n1 * n2)
    }

    pub fn mse(&self, other: &Vector) -> f32 {
        assert_eq!(self.len(), other.len());
        let sum_sq: f32 = self.data.iter().zip(&other.data).map(|(a, b)| (a - b).powi(2)).sum();
        sum_sq / (self.len() as f32)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>, // Row-major
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn random(rows: usize, cols: usize, rng: &mut Lcg) -> Self {
        // Xavier/Glorot initialization-ish
        let scale = (2.0 / (rows + cols) as f32).sqrt();
        let data = (0..rows * cols)
            .map(|_| rng.next_gaussian() * scale)
            .collect();
        Self { rows, cols, data }
    }

    pub fn matmul(&self, vec: &Vector) -> Vector {
        assert_eq!(self.cols, vec.len());
        let mut result = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            let mut sum = 0.0;
            for j in 0..self.cols {
                sum += self.data[i * self.cols + j] * vec.data[j];
            }
            result.push(sum);
        }
        Vector::from_vec(result)
    }

    // Outer product of two vectors (u * v^T) -> Matrix
    pub fn outer(u: &Vector, v: &Vector) -> Matrix {
        let rows = u.len();
        let cols = v.len();
        let mut data = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            for j in 0..cols {
                data.push(u.data[i] * v.data[j]);
            }
        }
        Matrix { rows, cols, data }
    }

    pub fn add_scaled(&mut self, other: &Matrix, scale: f32) {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        for i in 0..self.data.len() {
            self.data[i] += other.data[i] * scale;
        }
    }
}

impl fmt::Display for Vector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, v) in self.data.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{:.4}", v)?;
        }
        write!(f, "]")
    }
}
