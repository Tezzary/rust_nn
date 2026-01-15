use core::f32;

use rand::prelude::*;

#[derive(Clone, Debug)]
pub struct Matrix {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Matrix {
        let mut matrix = Matrix {
            data: Vec::new(),
            rows: rows,
            cols: cols,
        };

        for _y in 0..rows {
            for _x in 0..cols {
                matrix.data.push(0.0);
            }
        }

        matrix
    }

    //fill array random floats between [-1, 1]
    pub fn populate_random(rows: usize, cols: usize, min: f32, max: f32) -> Matrix {
        let mut matrix = Matrix {
            data: Vec::new(),
            rows: rows,
            cols: cols,
        };

        let mut rng = rand::rng();

        let range = max - min;

        for _y in 0..rows {
            for _x in 0..cols {
                matrix.data.push(rng.random::<f32>() * range + min);
            }
        }

        matrix
    }

    pub fn get_rows(&self) -> usize {
        self.rows
    }

    pub fn get_cols(&self) -> usize {
        self.cols
    }

    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.cols + col]
    }

    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        assert!(row < self.rows);
        assert!(col < self.cols);

        self.data[row * self.cols + col] = value;
    }

    pub fn add(&self, rhs: &Matrix) -> Matrix {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);
        
        let mut result = Matrix::zeros(self.rows, self.cols);

        for i in 0..self.data.len() {
            result.data[i] = self.data[i] + rhs.data[i];
        }

        result
    }

    pub fn subtract(&self, rhs: &Matrix) -> Matrix {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);
        
        let mut result = Matrix::zeros(self.rows, self.cols);

        for i in 0..self.data.len() {
            result.data[i] = self.data[i] - rhs.data[i];
        }

        result
    }

    pub fn multiply(&self, rhs: &Matrix) -> Matrix {
        assert_eq!(self.cols, rhs.rows);

        let mut result = Matrix::zeros(self.rows, rhs.cols);

        for y in 0..result.get_rows() {
            for x in 0..result.get_cols() {
                let mut value: f32 = 0.0;
                for counter in 0..self.cols {
                    value += self.get(y, counter) * rhs.get(counter, x);
                }
                result.set(y, x, value)
            }
        }

        result
    }

    pub fn pointwise_multiply(&self, rhs: &Matrix) -> Matrix {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);

        let mut result = Matrix::zeros(self.rows, self.cols);

        for y in 0..result.get_rows() {
            for x in 0..result.get_cols() {
                let value = self.get(y, x) * rhs.get(y, x);
                result.set(y, x, value);
            }
        }

        result
    }

    pub fn scalar_multiply(&self, scale: f32) -> Matrix {
        let mut result = Matrix::zeros(self.rows, self.cols);
         for y in 0..result.get_rows() {
            for x in 0..result.get_cols() {
                let value = self.get(y, x);
                result.set(y, x, value * scale);
            }
        }
        result
    }

    pub fn relu(&mut self) {
        for item in &mut self.data {
            if *item < 0.0 {
                *item = 0.0;
            }
        }
    }

    pub fn relu_derivative(&self) -> Matrix {
        let mut derivatives = Matrix::zeros(self.get_rows(), self.get_cols());
        
        for y in 0..self.get_rows() {
            for x in 0..self.get_cols() {
                let mut derivative = 0.0;
                if self.get(y, x) > 0.0 {
                    derivative = 1.0;
                }
                derivatives.set(y, x, derivative);
            }
        }

        derivatives
    }

    pub fn transpose(&self) -> Matrix {
        let mut t = Matrix::zeros(self.get_cols(), self.get_rows());

        for y in 0..self.get_rows() {
            for x in 0..self.get_cols() {
                let value = self.get(y, x);
                t.set(x, y, value);
            }
        }

        t
    }
    
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let matrix = Matrix::zeros(2, 3);

        assert_eq!(matrix.get_rows(), 2);
        assert_eq!(matrix.get_cols(), 3);

        for y in 0..matrix.get_rows() {
            for x in 0..matrix.get_cols() {
                assert_eq!(matrix.get(y, x), 0.0);
            }
        }
    }

    #[test]
    fn test_populate_random() {
        let matrix = Matrix::populate_random(2, 3, -1.0, 1.0);

        assert_eq!(matrix.get_rows(), 2);
        assert_eq!(matrix.get_cols(), 3);

        for y in 0..matrix.get_rows() {
            for x in 0..matrix.get_cols() {
                assert!(-1.0 <= matrix.get(y, x) && 1.0 >= matrix.get(y, x));
            }
        }
        
    }

    #[test]
    fn test_set() {
        let mut matrix = Matrix::zeros(2, 3);
        matrix.set(1, 2, 5.0);
        assert_eq!(matrix.get(1, 2), 5.0);
    }

    #[test]
    fn test_multiply() {
        let mut matrix = Matrix::zeros(2, 3);
        matrix.set(0, 0, 1.0);
        matrix.set(0, 1, 2.0);

        let mut matrix_2 = Matrix::zeros(3, 4);
        matrix_2.set(0, 2, 3.0);
        matrix_2.set(1, 2, 4.0);

        let result_matrix = matrix.multiply(&matrix_2);
        
        assert_eq!(result_matrix.get(0, 2), 11.0); //test multiply worked
        assert_eq!(result_matrix.get(0, 0), 0.0); //check other still 0

        assert_eq!(result_matrix.get_rows(), matrix.get_rows());
        assert_eq!(result_matrix.get_cols(), matrix_2.get_cols()); 
    }
}