extern crate rand;
use std;

// addition subtraction multiplication division
// + +=     - -=        * *=           / /=
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Sub;
use std::ops::SubAssign;
use std::ops::Mul;
use std::ops::MulAssign;
use std::ops::Div;
use std::ops::DivAssign;
// matrix multiplication
// ^ <<= >>=
use std::ops::BitXor;
use std::ops::ShlAssign;
use std::ops::ShrAssign;
// transpose
// !
use std::ops::Not;
// get parallel and perpendicular
//     | |=         % %=
use std::ops::BitOr;
use std::ops::BitOrAssign;
use std::ops::Rem;
use std::ops::RemAssign;
use r_matrix::rand::Rng;

// helper macro
// fn(&mut A) ==> fn(&A) -> A
macro_rules! impl_fn_i_0 {
    ($method:ident from $imethod:ident for $type:ty) => {
        impl $type {
            #[inline]
            pub fn $method(&self) -> $type {
                let mut tmp = self.clone();
                tmp.$imethod();
                tmp
            }
        }
    }
}

// fn(&mut A) -> A ==> fn(&A) -> (A, B)
macro_rules! impl_fn_i_1 {
    ($method:ident from $imethod:ident for $type:ty) => {
        impl $type {
            #[inline]
            pub fn $method(&self) -> ($type, $type) {
                let mut tmp = self.clone();
                let a = tmp.$imethod();
                (a, tmp)
            }
        }
    }
}

// fn(&mut A) -> (A, C) ==> fn(&A) -> (A, B, C)
macro_rules! impl_fn_i_2 {
    ($method:ident from $imethod:ident for $type:ty) => {
        impl $type {
            #[inline]
            pub fn $method(&self) -> ($type, $type, $type) {
                let mut tmp = self.clone();
                let (a, c) = tmp.$imethod();
                (a, tmp, c)
            }
        }
    }
}

// real(float64) matrix
pub struct RMatrix {
    // x: rows
    x: usize,
    // y: cols
    y: usize,
    data: Vec<f64>,
}

impl RMatrix {
    // get row data
    // in place
    fn row(&self, num: usize) -> &[f64] {
        &self.data[(num * self.y)..((num + 1) * self.y)]
    }

    // get row data
    pub fn get_row(&self, num:usize) -> Vec<f64> {
        let mut ret: Vec<f64> = vec![0.0; self.y];
        ret.copy_from_slice(&self.data[(num * self.y)..((num + 1) * self.y)]);
        ret
    }

    // get a vector of row data
    pub fn get_all_rows(&self) -> Vec<Vec<f64>> {
        let mut ret: Vec<Vec<f64>> = vec![vec![0.0; self.y]; self.x];
        for i in 0..self.x {
            ret[i].copy_from_slice(&self.data[(i * self.y)..((i + 1) * self.y)]);
        }
        ret
    }

    // get column data
    pub fn get_col(&self, num:usize) -> Vec<f64> {
        let mut ret: Vec<f64> = vec![0.0; self.x];
        for i in 0..self.x {
            ret[i] = self.data[i * self.y + num];
        }
        ret
    }

    // get a vector of column data
    pub fn get_all_cols(&self) -> Vec<Vec<f64>> {
        let mut ret: Vec<Vec<f64>> = vec![vec![0.0; self.x]; self.y];
        for i in 0..self.x {
            for j in 0..self.y {
                ret[j][i] = self.data[i * self.y + j];
            }
        }
        ret
    }

    // get diagonal data
    pub fn get_diag(&self) -> Vec<f64> {
        let num = if self.x > self.y {
            self.y
        } else {
            self.x
        };
        let mut ret: Vec<f64> = vec![0.0; num];
        for i in 0..num {
            ret[i] = self.data[i * self.y + i];
        }
        ret
    }

    // get data
    pub fn get_data(&self) -> Vec<f64> {
        self.data.clone()
    }

    // print all data to the screen, with info
    pub fn print(&self) {
        println!("x: {}", self.x);
        println!("y: {}", self.y);
        for i in 0..self.x {
            println!("{:?}", self.row(i));
        }
    }

    // print row & column number of the matrix
    pub fn print_info(&self) {
        println!("x: {}", self.x);
        println!("y: {}", self.y);
    }

    // generate a vector from a Vec
    pub fn gen_vec(vec: Vec<f64>) -> RMatrix {
        RMatrix {
            x: vec.len(),
            y: 1,
            data: vec.clone()
        }
    }

    // generate a matrix from a Vec
    pub fn gen_matrix(m: usize, n: usize, data: Vec<f64>) -> RMatrix {
        assert_eq!(data.len(), m * n, "RMatrix::gen_matrix(): vector size doesn't match");
        RMatrix {
            x: m,
            y: n,
            data: data.clone()
        }
    }

    // generate a matrix from a vector, as diagonal element
    pub fn gen_diag(m: usize, n: usize, diag: Vec<f64>) -> RMatrix {
        let min = m.min(n);
        assert_eq!(diag.len(), min, "RMatrix::gen_diag(): vector size doesn't match");
        let mut ret_data: Vec<f64> = vec![0.0; m * n];
        for i in 0..min {
            ret_data[i * (n + 1)] = diag[i];
        }
        RMatrix {
            x: m,
            y: n,
            data: ret_data
        }
    }

    // generate an eye matrix
    pub fn gen_eye(m: usize, n: usize) -> RMatrix {
        let min = m.min(n);
        let mut ret_data: Vec<f64> = vec![0.0; m * n];
        for i in 0..min {
            ret_data[i * (n + 1)] = 1.0;
        }
        RMatrix {
            x: m,
            y: n,
            data: ret_data
        }
    }

    // generate a matrix, all elements are 0
    pub fn gen_zeros(m: usize, n: usize) -> RMatrix {
        let ret_data: Vec<f64> = vec![0.0; m * n];
        RMatrix {
            x: m,
            y: n,
            data: ret_data
        }
    }

    // generate a matrix, all elements are 1
    pub fn gen_ones(m: usize, n: usize) -> RMatrix {
        let ret_data: Vec<f64> = vec![1.0; m * n];
        RMatrix {
            x: m,
            y: n,
            data: ret_data
        }
    }

    // generate a matrix, all elements are random number between 0 and 1
    pub fn gen_rand(m: usize, n: usize) -> RMatrix {
        let mut ret_data: Vec<f64> = vec![0.0; m * n];
        for i in 0..(m * n) {
            ret_data[i] = rand::thread_rng().gen_range(0.0, 1.0);
        }
        RMatrix {
            x: m,
            y: n,
            data: ret_data
        }
    }

    // generate a tridiagonal matrix, all elements are random number between 0 and 1
    pub fn gen_rand_tri(m: usize, n: usize) -> RMatrix {
        let mut ret_data: Vec<f64> = vec![0.0; m * n];
        let min = m.min(n);
        for i in 0..(min - 1) {
            ret_data[i * n + (i + 1)] = rand::thread_rng().gen_range(0.0, 1.0);
            ret_data[(i + 1) * n + i] = rand::thread_rng().gen_range(0.0, 1.0);
            ret_data[i * n + i] = rand::thread_rng().gen_range(0.0, 1.0);
        }
        ret_data[(min - 1) * (n + 1)] = rand::thread_rng().gen_range(0.0, 1.0);
        RMatrix {
            x: m,
            y: n,
            data: ret_data
        }
    }

    // generate a matrix from given eigenvalues
    // B = R1 ^ A ^ R2, R1 & R2: n Givens
    pub fn gen_rand_eig(n: usize, eig: Vec<f64>) -> RMatrix {
        assert_eq!(eig.len(), n, "RMatrix::gen_rand_eig(): vector size doesn't match");
        let mut ret_data: Vec<f64> = vec![0.0; n * n];
        let mut x: f64;
        let mut y: f64;
        let mut c: f64;
        let mut s: f64;
        for i in 0..n {
            ret_data[i * n + i] = eig[i];
        }
        for k in 0..(n - 1) {
            c = rand::thread_rng().gen_range(0.0, 1.0);
            s = (1.0 - c * c).sqrt();
            for i in 0..n {
                x = ret_data[i * n + k];
                y = ret_data[i * n + k + 1];
                ret_data[i * n + k] = x * c + y * s;
                ret_data[i * n + k + 1] = -x * s + y * c;
            }
            c = rand::thread_rng().gen_range(0.0, 1.0);
            s = (1.0 - c * c).sqrt();
            for j in 0..n {
                x = ret_data[k * n + j];
                y = ret_data[(k + 1) * n + j];
                ret_data[k * n + j] = x * c + y * s;
                ret_data[(k + 1) * n + j] = -x * s + y * c;
            }
        }
        RMatrix {
            x: n,
            y: n,
            data: ret_data
        }
    }

    // generate an up bidiagonal matrix, all elements are random number between 0 and 1
    pub fn gen_rand_ubi(n: usize) -> RMatrix {
        let mut ret_data: Vec<f64> = vec![0.0; n * n];
        for i in 0..(n - 1) {
            ret_data[i * n + (i + 1)] = rand::thread_rng().gen_range(0.0, 1.0);
            ret_data[i * n + i] = rand::thread_rng().gen_range(0.0, 1.0);
        }
        ret_data[n * n - 1] = rand::thread_rng().gen_range(0.0, 1.0);
        RMatrix {
            x: n,
            y: n,
            data: ret_data
        }
    }

    // generate a diagonal matrix, all elements are random number between 0 and 1
    pub fn gen_rand_diag(m: usize, n: usize) -> RMatrix {
        let mut ret_data: Vec<f64> = vec![0.0; m * n];
        let min = m.min(n);
        for i in 0..min {
            ret_data[i * n + i] = rand::thread_rng().gen_range(0.0, 1.0);
        }
        RMatrix {
            x: m,
            y: n,
            data: ret_data
        }
    }

    // generate a symmetric matrix, all elements are random number between 0 and 1
    pub fn gen_rand_sym(n: usize) -> RMatrix {
        let mut ret_data: Vec<f64> = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..i {
                ret_data[i * n + j] = rand::thread_rng().gen_range(0.0, 1.0);
                ret_data[j * n + i] = ret_data[i * n + j];
            }
            ret_data[i * n + i] = rand::thread_rng().gen_range(0.0, 1.0);
        }
        RMatrix {
            x: n,
            y: n,
            data: ret_data
        }
    }

    // generate a symmetric tridiagonal matrix, all elements are random number between 0 and 1
    pub fn gen_rand_sym_tri(n: usize) -> RMatrix {
        let mut ret_data: Vec<f64> = vec![0.0; n * n];
        for i in 0..(n - 1) {
            ret_data[i * n + (i + 1)] = rand::thread_rng().gen_range(0.0, 1.0);
            ret_data[(i + 1) * n + i] = ret_data[i * n + (i + 1)];
            ret_data[i * n + i] = rand::thread_rng().gen_range(0.0, 1.0);
        }
        ret_data[n * n - 1] = rand::thread_rng().gen_range(0.0, 1.0);
        RMatrix {
            x: n,
            y: n,
            data: ret_data
        }
    }

    // generate a symmetric matrix from given eigenvalues
    // B = R ^ A ^ !R, R: n Givens
    pub fn gen_rand_sym_eig(n: usize, eig: Vec<f64>) -> RMatrix {
        assert_eq!(eig.len(), n, "RMatrix::gen_rand_sym_eig(): vector size doesn't match");
        let mut ret_data: Vec<f64> = vec![0.0; n * n];
        let mut x: f64;
        let mut y: f64;
        let mut c: f64;
        let mut s: f64;
        for i in 0..n {
            ret_data[i * n + i] = eig[i];
        }
        for k in 0..(n - 1) {
            c = rand::thread_rng().gen_range(0.0, 1.0);
            s = (1.0 - c * c).sqrt();
            for i in 0..n {
                x = ret_data[i * n + k];
                y = ret_data[i * n + k + 1];
                ret_data[i * n + k] = x * c + y * s;
                ret_data[i * n + k + 1] = -x * s + y * c;
            }
            for j in 0..n {
                x = ret_data[k * n + j];
                y = ret_data[(k + 1) * n + j];
                ret_data[k * n + j] = x * c + y * s;
                ret_data[(k + 1) * n + j] = -x * s + y * c;
            }
        }
        // symmetric
        for i in 0..(n - 1) {
            for j in 0..i {
                ret_data[i * n + j] = (ret_data[i * n + j] + ret_data[j * n + i]) / 2.0;
                ret_data[j * n + i] = ret_data[i * n + j];
            }
        }
        RMatrix {
            x: n,
            y: n,
            data: ret_data
        }
    }

    // generate a skew symmetric matrix, all elements are random number between 0 and 1
    pub fn gen_rand_ssym(n: usize) -> RMatrix {
        let mut ret_data: Vec<f64> = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..i {
                ret_data[i * n + j] = rand::thread_rng().gen_range(0.0, 1.0);
                ret_data[j * n + i] = -1.0 * ret_data[i * n + j];
            }
            ret_data[i * n + i] = 0.0;
        }
        RMatrix {
            x: n,
            y: n,
            data: ret_data
        }
    }

    // generate a skew symmetric tridiagonal matrix, all elements are random number between 0 and 1
    pub fn gen_rand_ssym_tri(n: usize) -> RMatrix {
        let mut ret_data: Vec<f64> = vec![0.0; n * n];
        for i in 0..(n - 1) {
            ret_data[i * n + (i + 1)] = rand::thread_rng().gen_range(0.0, 1.0);
            ret_data[(i + 1) * n + i] = -1.0 * ret_data[i * n + (i + 1)];
        }
        RMatrix {
            x: n,
            y: n,
            data: ret_data
        }
    }

    // 1-norm
    pub fn norm_1(&self) -> f64 {
        let mut sum: f64;
        let mut ret: f64 = 0.0;
        for j in 0..self.y {
            sum = 0.0;
            for i in 0..self.x {
                sum += self.data[i * self.y + j].abs();
            }
            ret = ret.max(sum);
        }
        ret
    }

    // 2-norm
    pub fn norm_2(&self) -> f64 {
        if self.y != 1 {
            self.norm_2_mat();
        }
        let mut ret: f64 = 0.0;
        for i in 0..self.x {
            ret += self.data[i] * self.data[i];
        }
        ret.sqrt()
    }

    // 2-norm of a matrix, SVD
    fn norm_2_mat(&self) -> f64 {
        let mut tmp = self.clone();
        tmp.isv_qr();
        tmp.data[0]
    }

    // frobenius-norm of a matrix
    pub fn norm_f(&self) -> f64 {
        let mut ret: f64 = 0.0;
        for i in 0..(self.x * self.y) {
            ret += self.data[i] * self.data[i];
        }
        ret.sqrt()
    }

    // infinity-norm
    pub fn norm_i(&self) -> f64 {
        let mut sum: f64;
        let mut ret: f64 = 0.0;
        for i in 0..self.x {
            sum = 0.0;
            for j in 0..self.y {
                sum += self.data[i * self.y + j].abs();
            }
            ret = ret.max(sum);
        }
        ret
    }

    // condition number of a matrix, SVD
    pub fn cond(&self) -> f64 {
        let mut tmp = self.clone();
        tmp.isv_qr();
        let min = tmp.x.min(tmp.y);
        tmp.data[0] / tmp.data[(min - 1) * (tmp.y + 1)]
    }

    // get the maximum element of a matrix
    pub fn max(&self) -> f64 {
        let mut ret: f64 = self.data[0];
        for i in 1..(self.x * self.y) {
            ret = ret.max(self.data[i]);
        }
        ret
    }

    // get the minimum element of a matrix
    pub fn min(&self) -> f64 {
        let mut ret: f64 = self.data[0];
        for i in 1..(self.x * self.y) {
            ret = ret.min(self.data[i]);
        }
        ret
    }

    // get the sum of all element in the matrix
    pub fn sum(&self) -> f64 {
        let mut ret: f64 = 0.0;
        for i in 0..(self.x * self.y) {
            ret += self.data[i];
        }
        ret
    }

    // calculate the square of A
    // B = A ^ !A
    pub fn square(&self) -> RMatrix {
        let mut ret_data: Vec<f64> = vec![0.0; self.x * self.x];
        let mut sum: f64;
        for i in 0..self.x {
            for j in i..self.x {
                sum = 0.0;
                for k in 0..self.y {
                    sum += self.data[i * self.y + k] * self.data[j * self.y + k];
                }
                ret_data[i * self.x + j] = sum;
                ret_data[j * self.x + i] = sum;
            }
        }
        RMatrix {
            x: self.x,
            y: self.x,
            data: ret_data
        }
    }

    // calculate the square of A
    // A = A ^ !A
    // in place
    // A must be square
    pub fn isquare(&mut self) {
        assert_eq!(self.x, self.y, "RMatrix.isquare(): square matrix only");
        let mut tmp: Vec<f64>;
        let mut sum: f64;
        for i in 0..self.x {
            tmp = self.get_row(i);
            for j in i..self.x {
                sum = 0.0;
                for k in 0..self.y {
                    sum += tmp[k] * self.data[j * self.y + k];
                }
                self.data[i * self.x + j] = sum;
            }
            for j in 0..i {
                self.data[i * self.x + j] = self.data[j * self.x + i];
            }
        }
    }

    // Cholesky decomposition
    // A must be symmetric, positive-define
    // in place
    // A = L ^ !L
    pub fn ichol(&mut self) {
        assert_eq!(self.x, self.y, "RMatrix.chol(): square matrix only");
        for i in 0..self.x {
            for j in (i + 1)..self.y {
                self.data[i * self.y + j] = 0.0;
            }
        }
        for i in 0..self.x {
            for j in 0..i {
                for k in 0..j {
                    self.data[i * self.y + j] -= self.data[i * self.y + k] * self.data[j * self.y + k];
                }
                self.data[i * self.y + j] /= self.data[j * self.y + j];
                self.data[i * self.y + i] -= self.data[i * self.y + j] * self.data[i * self.y + j];
            }
            self.data[i * self.y + i] = self.data[i * self.y + i].sqrt();
        }
    }

    // Cholesky decomposition for tridiagonal matrix
    // A must be symmetric, positive-define
    // in place
    // A = L ^ !L
    pub fn ichol_tri(&mut self) {
        assert_eq!(self.x, self.y, "RMatrix.chol_tri(): square matrix only");
        for i in 1..self.x {
            self.data[(i - 1) * self.y + i] = 0.0;
        }
        self.data[0] = self.data[0].sqrt();
        for i in 1..self.x {
            self.data[i * self.y + i - 1] /= self.data[(i - 1) * (self.y + 1)];
            self.data[i * self.y + i] -= self.data[i * self.y + i - 1] * self.data[i * self.y + i - 1];
            self.data[i * self.y + i] = self.data[i * self.y + i].sqrt();
        }
    }

    // LU decomposition
    // in place
    // A = L ^ U
    pub fn ilu(&mut self) -> RMatrix {
        assert_eq!(self.x, self.y, "RMatrix.lu(): square matrix only");
        let mut ret_l_data: Vec<f64> = vec![0.0; self.x * self.y];
        for j in 0..self.y {
            for i in (j + 1)..self.x {
                ret_l_data[i * self.y + j] = self.data[i * self.y + j] / self.data[j * self.y + j];
                for k in j..self.y {
                    self.data[i * self.y + k] -= self.data[j * self.y + k] * ret_l_data[i * self.y + j];
                }
            }
        }
        for i in 0..self.x {
            ret_l_data[i * self.y + i] = 1.0;
        }
        RMatrix {
            x: self.x,
            y: self.y,
            data: ret_l_data
        }
    }

    // LU decomposition for tridiagonal matrix
    // in place
    // A = L ^ U
    pub fn ilu_tri(&mut self) -> RMatrix {
        assert_eq!(self.x, self.y, "RMatrix.lu_tri(): square matrix only");
        let mut ret_l_data: Vec<f64> = vec![0.0; self.x * self.y];
        ret_l_data[0] = 1.0;
        for j in 1..self.y {
            ret_l_data[j * self.y + j - 1] = self.data[j * self.y + j - 1] / self.data[(j - 1) * (self.y + 1)];
            ret_l_data[j * self.y + j] = 1.0;
            self.data[j * self.y + j - 1] = 0.0;
            self.data[j * self.y + j] -= self.data[(j - 1) * self.y + j] * ret_l_data[j * self.y + j - 1];
        }
        RMatrix {
            x: self.x,
            y: self.y,
            data: ret_l_data
        }
    }

    // LU decomposition
    // P ^ A = L ^ U
    pub fn plu(&self) -> (RMatrix, RMatrix, RMatrix) {
        assert_eq!(self.x, self.y, "RMatrix.plu(): square matrix only");
        let mut ret_p_data: Vec<f64> = vec![0.0; self.x * self.y];
        let mut ret_l_data: Vec<f64> = vec![0.0; self.x * self.y];
        let mut ret_u_data: Vec<f64> = vec![0.0; self.x * self.y];
        {
            let mut p_rank: Vec<usize> = vec![0; self.x];
            let mut l_data: Vec<f64> = vec![0.0; self.x * self.y];
            let mut u_data: Vec<f64> = self.data.clone();
            for i in 0..self.x {
                p_rank[i] = i;
            }
            let mut max_d: f64;
            let mut max_r: usize;
            let mut tmp: usize;
            for j in 0..self.y {
                max_d = u_data[p_rank[j] * self.y + j];
                max_r = j;
                for i in (j + 1)..self.x {
                    if u_data[p_rank[i] * self.y + j].abs() > max_d.abs() {
                        max_d = u_data[p_rank[i] * self.y + j];
                        max_r = i;
                    }
                }
                tmp = p_rank[max_r];
                p_rank[max_r] = p_rank[j];
                p_rank[j] = tmp;
                for i in (j + 1)..self.x {
                    l_data[p_rank[i] * self.y + j] = u_data[p_rank[i] * self.y + j] / max_d;
                    for k in j..self.y {
                        u_data[p_rank[i] * self.y + k] -= u_data[p_rank[j] * self.y + k] * l_data[p_rank[i] * self.y + j];
                    }
                }
            }
            // write to P
            for i in 0..self.x {
                ret_p_data[i * self.y + p_rank[i]] = 1.0;
                for j in 0..self.y {
                    ret_l_data[i * self.y + j] = l_data[p_rank[i] * self.y + j];
                    ret_u_data[i * self.y + j] = u_data[p_rank[i] * self.y + j];
                }
                ret_l_data[i * self.y + i] = 1.0;
            }
        }
        (RMatrix {
            x: self.x,
            y: self.y,
            data: ret_p_data
        },
        RMatrix {
            x: self.x,
            y: self.y,
            data: ret_l_data
        },
        RMatrix {
            x: self.x,
            y: self.y,
            data: ret_u_data
        })
    }

    // LU decomposition
    // A ^ Q = L ^ U
    pub fn qlu(&self) -> (RMatrix, RMatrix, RMatrix) {
        assert_eq!(self.x, self.y, "RMatrix.qlu(): square matrix only");
        let mut ret_q_data: Vec<f64> = vec![0.0; self.x * self.y];
        let mut ret_l_data: Vec<f64> = vec![0.0; self.x * self.y];
        let mut ret_u_data: Vec<f64> = vec![0.0; self.x * self.y];
        {
            let mut q_rank: Vec<usize> = vec![0; self.y];
            let mut l_data: Vec<f64> = vec![0.0; self.x * self.y];
            let mut u_data: Vec<f64> = self.data.clone();
            for i in 0..self.y {
                q_rank[i] = i;
            }
            let mut max_d: f64;
            let mut max_r: usize;
            let mut tmp: usize;
            for j in 0..self.y {
                max_d = u_data[j * self.y + q_rank[j]];
                max_r = j;
                for i in (j + 1)..self.y {
                    if u_data[j * self.y + q_rank[i]].abs() > max_d.abs() {
                        max_d = u_data[j * self.y + q_rank[i]];
                        max_r = i;
                    }
                }
                tmp = q_rank[max_r];
                q_rank[max_r] = q_rank[j];
                q_rank[j] = tmp;
                for i in (j + 1)..self.x {
                    l_data[i * self.y + q_rank[j]] = u_data[i * self.y + q_rank[j]] / max_d;
                    for k in j..self.y {
                        u_data[i * self.y + q_rank[k]] -= u_data[j * self.y + q_rank[k]] * l_data[i * self.y + q_rank[j]];
                    }
                }
            }
            // write to Q
            for i in 0..self.x {
                ret_q_data[q_rank[i] * self.y + i] = 1.0;
                for j in 0..self.y {
                    ret_l_data[i * self.y + j] = l_data[i * self.y + q_rank[j]];
                    ret_u_data[i * self.y + j] = u_data[i * self.y + q_rank[j]];
                }
                ret_l_data[i * self.y + i] = 1.0;
            }
        }
        (RMatrix {
            x: self.x,
            y: self.y,
            data: ret_q_data
        },
        RMatrix {
            x: self.x,
            y: self.y,
            data: ret_l_data
        },
        RMatrix {
            x: self.x,
            y: self.y,
            data: ret_u_data
        })
    }

    // LU decomposition
    // P ^ A ^ !P = L ^ U
    pub fn pplu(&self) -> (RMatrix, RMatrix, RMatrix) {
        assert_eq!(self.x, self.y, "RMatrix.pplu(): square matrix only");
        let mut ret_p_data: Vec<f64> = vec![0.0; self.x * self.y];
        let mut ret_l_data: Vec<f64> = vec![0.0; self.x * self.y];
        let mut ret_u_data: Vec<f64> = vec![0.0; self.x * self.y];
        {
            let mut p_rank: Vec<usize> = vec![0; self.x];
            let mut l_data: Vec<f64> = vec![0.0; self.x * self.y];
            let mut u_data: Vec<f64> = self.data.clone();
            for i in 0..self.x {
                p_rank[i] = i;
            }
            let mut max_d: f64;
            let mut max_r: usize;
            let mut tmp: usize;
            for j in 0..self.y {
                max_d = u_data[p_rank[j] * self.y + p_rank[j]];
                max_r = j;
                for i in (j + 1)..self.x {
                    if u_data[p_rank[i] * self.y + p_rank[i]].abs() > max_d.abs() {
                        max_d = u_data[p_rank[i] * self.y + p_rank[i]];
                        max_r = i;
                    }
                }
                tmp = p_rank[max_r];
                p_rank[max_r] = p_rank[j];
                p_rank[j] = tmp;
                for i in (j + 1)..self.x {
                    l_data[p_rank[i] * self.y + p_rank[j]] = u_data[p_rank[i] * self.y + p_rank[j]] / max_d;
                    for k in j..self.y {
                        u_data[p_rank[i] * self.y + p_rank[k]] -= u_data[p_rank[j] * self.y + p_rank[k]] * l_data[p_rank[i] * self.y + p_rank[j]];
                    }
                }
            }
            // write to P
            for i in 0..self.x {
                ret_p_data[i * self.y + p_rank[i]] = 1.0;
                for j in 0..self.y {
                    ret_l_data[i * self.y + j] = l_data[p_rank[i] * self.y + p_rank[j]];
                    ret_u_data[i * self.y + j] = u_data[p_rank[i] * self.y + p_rank[j]];
                }
                ret_l_data[i * self.y + i] = 1.0;
            }
        }
        (RMatrix {
            x: self.x,
            y: self.y,
            data: ret_p_data
        },
        RMatrix {
            x: self.x,
            y: self.y,
            data: ret_l_data
        },
        RMatrix {
            x: self.x,
            y: self.y,
            data: ret_u_data
        })
    }

    // LU decomposition
    // P ^ A ^ Q = L ^ U
    pub fn pqlu(&self) -> (RMatrix, RMatrix, RMatrix, RMatrix) {
        assert_eq!(self.x, self.y, "RMatrix.pplu(): square matrix only");
        let mut ret_p_data: Vec<f64> = vec![0.0; self.x * self.y];
        let mut ret_q_data: Vec<f64> = vec![0.0; self.x * self.y];
        let mut ret_l_data: Vec<f64> = vec![0.0; self.x * self.y];
        let mut ret_u_data: Vec<f64> = vec![0.0; self.x * self.y];
        {
            let mut p_rank: Vec<usize> = vec![0; self.x];
            let mut q_rank: Vec<usize> = vec![0; self.y];
            let mut l_data: Vec<f64> = vec![0.0; self.x * self.y];
            let mut u_data: Vec<f64> = self.data.clone();
            for i in 0..self.x {
                p_rank[i] = i;
            }
            for i in 0..self.y {
                q_rank[i] = i;
            }
            let mut max_d: f64;
            let mut max_p: usize;
            let mut max_q: usize;
            let mut tmp: usize;
            for j in 0..self.y {
                max_d = u_data[p_rank[j] * self.y + q_rank[j]];
                max_p = j;
                max_q = j;
                for i in j..self.x {
                    for k in j..self.y {
                        if u_data[p_rank[i] * self.y + q_rank[k]].abs() > max_d.abs() {
                            max_d = u_data[p_rank[i] * self.y + q_rank[k]];
                            max_p = i;
                            max_q = k;
                        }
                    }
                }
                tmp = p_rank[max_p];
                p_rank[max_p] = p_rank[j];
                p_rank[j] = tmp;
                tmp = q_rank[max_q];
                q_rank[max_q] = q_rank[j];
                q_rank[j] = tmp;
                for i in (j + 1)..self.x {
                    l_data[p_rank[i] * self.y + q_rank[j]] = u_data[p_rank[i] * self.y + q_rank[j]] / max_d;
                    for k in j..self.y {
                        u_data[p_rank[i] * self.y + q_rank[k]] -= u_data[p_rank[j] * self.y + q_rank[k]] * l_data[p_rank[i] * self.y + q_rank[j]];
                    }
                }
            }
            // write to P and Q
            for i in 0..self.x {
                ret_p_data[i * self.y + p_rank[i]] = 1.0;
                ret_q_data[q_rank[i] * self.y + i] = 1.0;
                for j in 0..self.y {
                    ret_l_data[i * self.y + j] = l_data[p_rank[i] * self.y + q_rank[j]];
                    ret_u_data[i * self.y + j] = u_data[p_rank[i] * self.y + q_rank[j]];
                }
                ret_l_data[i * self.y + i] = 1.0;
            }
        }
        (RMatrix {
            x: self.x,
            y: self.y,
            data: ret_p_data
        },
        RMatrix {
            x: self.x,
            y: self.y,
            data: ret_q_data
        },
        RMatrix {
            x: self.x,
            y: self.y,
            data: ret_l_data
        },
        RMatrix {
            x: self.x,
            y: self.y,
            data: ret_u_data
        })
    }

    // compact QR decomposition, classical Gram-Schmidt
    // A = Q ^ R
    pub fn cqr_cgs(&self) -> (RMatrix, RMatrix) {
        assert!(self.x >= self.y, "RMatrix.cqr_cgs(): x >= y only");
        let mut ret_q_data: Vec<f64> = vec![0.0; self.x * self.y];
        let mut ret_r_data: Vec<f64> = vec![0.0; self.y * self.y];
        let mut tmp: Vec<f64> = vec![0.0; self.x];
        let mut n2: f64;
        for j in 0..self.y {
            for i in 0..self.x {
                tmp[i] = self.data[i * self.y + j];
            }
            for k in 0..j {
                for i in 0..self.x {
                    ret_r_data[k * self.y + j] += ret_q_data[i * self.y + k] * self.data[i * self.y + j];
                }
                for i in 0..self.x {
                    tmp[i] -= ret_r_data[k * self.y + j] * ret_q_data[i * self.y + k];
                }
            }
            n2 = 0.0;
            for i in 0..self.x {
                n2 += tmp[i] * tmp[i];
            }
            assert_ne!(n2, 0.0, "RMatrix.cqr_cgs(): break!");
            n2 = n2.sqrt();
            ret_r_data[j * self.y + j] = n2;
            for i in 0..self.x {
                ret_q_data[i * self.y + j] = tmp[i] / n2;
            }
        }
        (RMatrix {
            x: self.x,
            y: self.y,
            data: ret_q_data
        },
        RMatrix {
            x: self.y,
            y: self.y,
            data: ret_r_data
        })
    }

    // compact QR decomposition, modified Gram-Schmidt
    // A = Q ^ R
    pub fn cqr_mgs(&self) -> (RMatrix, RMatrix) {
        assert!(self.x >= self.y, "RMatrix.cqr_mgs(): x >= y only");
        let mut ret_q_data: Vec<f64> = vec![0.0; self.x * self.y];
        let mut ret_r_data: Vec<f64> = vec![0.0; self.y * self.y];
        let mut tmp: Vec<f64> = vec![0.0; self.x];
        let mut n2: f64;
        for j in 0..self.y {
            for i in 0..self.x {
                tmp[i] = self.data[i * self.y + j];
            }
            for k in 0..j {
                for i in 0..self.x {
                    ret_r_data[k * self.y + j] += ret_q_data[i * self.y + k] * tmp[i];
                }
                for i in 0..self.x {
                    tmp[i] -= ret_r_data[k * self.y + j] * ret_q_data[i * self.y + k];
                }
            }
            n2 = 0.0;
            for i in 0..self.x {
                n2 += tmp[i] * tmp[i];
            }
            assert_ne!(n2, 0.0, "RMatrix.cqr_mgs(): break!");
            n2 = n2.sqrt();
            ret_r_data[j * self.y + j] = n2;
            for i in 0..self.x {
                ret_q_data[i * self.y + j] = tmp[i] / n2;
            }
        }
        (RMatrix {
            x: self.x,
            y: self.y,
            data: ret_q_data
        },
        RMatrix {
            x: self.y,
            y: self.y,
            data: ret_r_data
        })
    }

    // QR decomposition, Householder
    // in place
    // A = Q ^ R
    pub fn iqr_hh(&mut self) -> RMatrix {
        let mut ret_q_data: Vec<f64> = vec![0.0; self.x * self.x];
        let mut tmp: Vec<f64> = vec![0.0; self.x];
        let mut n2: f64;
        let min: usize = self.x.min(self.y);
        for i in 0..self.x {
            ret_q_data[i * self.x + i] = 1.0;
        }
        for j in 0..min {
            // Householder
            for i in 0..j {
                tmp[i] = 0.0;
            }
            for i in j..self.x {
                tmp[i] = self.data[i * self.y + j];
            }
            n2 = 0.0;
            for i in j..self.x {
                n2 += tmp[i] * tmp[i];
            }
            if n2 == tmp[j] * tmp[j] {
                continue;
            }
            if tmp[j] > 0.0 {
                tmp[j] += n2.sqrt();
                n2 = (2.0 * n2.sqrt() * tmp[j]).sqrt();
            } else {
                tmp[j] -= n2.sqrt();
                n2 = (-2.0 * n2.sqrt() * tmp[j]).sqrt();
            }
            for i in j..self.x {
                tmp[i] /= n2;
            }
            // apply to A
            for k in j..self.y {
                n2 = 0.0;
                for i in j..self.x {
                    n2 += self.data[i * self.y + k] * tmp[i];
                }
                for i in j..self.x {
                    self.data[i * self.y + k] -= 2.0 * n2 * tmp[i];
                }
            }
            for i in (j + 1)..self.x {
                self.data[i * self.y + j] = 0.0;
            }
            // get Q
            for k in 0..self.x {
                n2 = 0.0;
                for i in j..self.x {
                    n2 += ret_q_data[k * self.x + i] * tmp[i];
                }
                for i in j..self.x {
                    ret_q_data[k * self.x + i] -= 2.0 * n2 * tmp[i];
                }
            }
        }
        RMatrix {
            x: self.x,
            y: self.x,
            data: ret_q_data
        }
    }

    // QR decomposition, Householder
    // in place
    // without Q
    pub fn ir_hh(&mut self) {
        let mut tmp: Vec<f64> = vec![0.0; self.x];
        let mut n2: f64;
        let min: usize = self.x.min(self.y);
        for j in 0..min {
            // Householder
            for i in 0..j {
                tmp[i] = 0.0;
            }
            for i in j..self.x {
                tmp[i] = self.data[i * self.y + j];
            }
            n2 = 0.0;
            for i in j..self.x {
                n2 += tmp[i] * tmp[i];
            }
            if n2 == tmp[j] * tmp[j] {
                continue;
            }
            if tmp[j] > 0.0 {
                tmp[j] += n2.sqrt();
                n2 = (2.0 * n2.sqrt() * tmp[j]).sqrt();
            } else {
                tmp[j] -= n2.sqrt();
                n2 = (-2.0 * n2.sqrt() * tmp[j]).sqrt();
            }
            for i in j..self.x {
                tmp[i] /= n2;
            }
            // apply to A
            for k in j..self.y {
                n2 = 0.0;
                for i in j..self.x {
                    n2 += self.data[i * self.y + k] * tmp[i];
                }
                for i in j..self.x {
                    self.data[i * self.y + k] -= 2.0 * n2 * tmp[i];
                }
            }
            for i in (j + 1)..self.x {
                self.data[i * self.y + j] = 0.0;
            }
        }
    }

    // QR decomposition, Givens
    // in place
    // A = Q ^ R
    pub fn iqr_givens(&mut self) -> RMatrix {
        let mut ret_q_data: Vec<f64> = vec![0.0; self.x * self.x];
        let mut x: f64;
        let mut y: f64;
        let mut c: f64;
        let mut s: f64;
        for i in 0..self.x {
            ret_q_data[i * self.x + i] = 1.0;
        }
        for i in 0..self.x {
            for j in 0..i.min(self.y) {
                if self.data[i * self.y + j] == 0.0 {
                    continue;
                }
                // Givens
                x = self.data[j * self.y + j];
                y = self.data[i * self.y + j];
                c = x / (x * x + y * y).sqrt();
                s = y / (x * x + y * y).sqrt();
                for k in j..self.y {
                    x = self.data[j * self.y + k];
                    y = self.data[i * self.y + k];
                    self.data[j * self.y + k] =  x * c + y * s;
                    self.data[i * self.y + k] = -x * s + y * c;
                }
                self.data[i * self.y + j] = 0.0;
                // get Q
                for k in 0..(i + 1) {
                    x = ret_q_data[k * self.x + j];
                    y = ret_q_data[k * self.x + i];
                    ret_q_data[k * self.x + j] =  x * c + y * s;
                    ret_q_data[k * self.x + i] = -x * s + y * c;
                }
            }
        }
        RMatrix {
            x: self.x,
            y: self.x,
            data: ret_q_data
        }
    }

    // QR decomposition, Givens
    // in place
    // without Q
    pub fn ir_givens(&mut self) {
        let mut x: f64;
        let mut y: f64;
        let mut c: f64;
        let mut s: f64;
        for i in 0..self.x {
            for j in 0..i.min(self.y) {
                if self.data[i * self.y + j] == 0.0 {
                    continue;
                }
                // Givens
                x = self.data[j * self.y + j];
                y = self.data[i * self.y + j];
                c = x / (x * x + y * y).sqrt();
                s = y / (x * x + y * y).sqrt();
                for k in j..self.y {
                    x = self.data[j * self.y + k];
                    y = self.data[i * self.y + k];
                    self.data[j * self.y + k] =  x * c + y * s;
                    self.data[i * self.y + k] = -x * s + y * c;
                }
                self.data[i * self.y + j] = 0.0;
            }
        }
    }

    // Hessenberg matrix decomposition, Householder
    // in place
    // A = Q ^ H ^ !Q
    pub fn iqhq_hh(&mut self) -> RMatrix {
        assert_eq!(self.x, self.y, "RMatrix.qhq_hh(): square matrix only");
        if self.x < 3 {
            return RMatrix::gen_eye(self.x, self.y);
        }
        let mut ret_q_data: Vec<f64> = vec![0.0; self.x * self.x];
        let mut tmp: Vec<f64> = vec![0.0; self.x];
        let mut n2: f64;
        for i in 0..self.x {
            ret_q_data[i * self.x + i] = 1.0;
        }
        for n in 0..(self.y - 1) {
            // Householder
            for i in 0..(n + 1) {
                tmp[i] = 0.0;
            }
            for i in (n + 1)..self.x {
                tmp[i] = self.data[i * self.y + n];
            }
            n2 = 0.0;
            for i in (n + 1)..self.x {
                n2 += tmp[i] * tmp[i];
            }
            if n2 == tmp[n + 1] * tmp[n + 1] {
                continue;
            }
            if tmp[n + 1] > 0.0 {
                tmp[n + 1] += n2.sqrt();
                n2 = (2.0 * n2.sqrt() * tmp[n + 1]).sqrt();
            } else {
                tmp[n + 1] -= n2.sqrt();
                n2 = (-2.0 * n2.sqrt() * tmp[n + 1]).sqrt();
            }
            for i in (n + 1)..self.x {
                tmp[i] /= n2;
            }
            // apply to A
            for k in n..self.y {
                n2 = 0.0;
                for i in (n + 1)..self.x {
                    n2 += self.data[i * self.y + k] * tmp[i];
                }
                for i in (n + 1)..self.x {
                    self.data[i * self.y + k] -= 2.0 * n2 * tmp[i];
                }
            }
            for i in (n + 2)..self.x {
                self.data[i * self.y + n] = 0.0;
            }
            // get Q
            for k in 0..self.x {
                n2 = 0.0;
                for i in (n + 1)..self.x {
                    n2 += ret_q_data[k * self.x + i] * tmp[i];
                }
                for i in (n + 1)..self.x {
                    ret_q_data[k * self.x + i] -= 2.0 * n2 * tmp[i];
                }
            }
            // apply to A
            for k in 0..self.x {
                n2 = 0.0;
                for i in (n + 1)..self.y {
                    n2 += self.data[k * self.y + i] * tmp[i];
                }
                for i in (n + 1)..self.y {
                    self.data[k * self.y + i] -= 2.0 * n2 * tmp[i];
                }
            }
        }
        RMatrix {
            x: self.x,
            y: self.x,
            data: ret_q_data
        }
    }

    // Hessenberg matrix decomposition, Givens
    // in place
    // A = Q ^ H ^ !Q
    pub fn iqhq_givens(&mut self) -> RMatrix {
        assert_eq!(self.x, self.y, "RMatrix.qhq_givens(): square matrix only");
        if self.x < 3 {
            return RMatrix::gen_eye(self.x, self.y);
        }
        let mut ret_q_data: Vec<f64> = vec![0.0; self.x * self.x];
        let mut x: f64;
        let mut y: f64;
        let mut c: f64;
        let mut s: f64;
        for i in 0..self.x {
            ret_q_data[i * self.x + i] = 1.0;
        }
        for n in 0..(self.y - 1) {
            for i in (n + 2)..self.x {
                if self.data[i * self.y + n] == 0.0 {
                    continue;
                }
                // Givens
                x = self.data[(n + 1) * self.y + n];
                y = self.data[i * self.y + n];
                c = x / (x * x + y * y).sqrt();
                s = y / (x * x + y * y).sqrt();
                for k in n..self.y {
                    x = self.data[(n + 1) * self.y + k];
                    y = self.data[i * self.y + k];
                    self.data[(n + 1) * self.y + k] =  x * c + y * s;
                    self.data[i * self.y + k] = -x * s + y * c;
                }
                self.data[i * self.y + n] = 0.0;
                // get Q
                for k in 0..(i + 1) {
                    x = ret_q_data[k * self.x + n + 1];
                    y = ret_q_data[k * self.x + i];
                    ret_q_data[k * self.x + n + 1] =  x * c + y * s;
                    ret_q_data[k * self.x + i] = -x * s + y * c;
                }
                for k in 0..self.x {
                    x = self.data[k * self.y + n + 1];
                    y = self.data[k * self.y + i];
                    self.data[k * self.y + n + 1] =  x * c + y * s;
                    self.data[k * self.y + i] = -x * s + y * c;
                }
            }
        }
        RMatrix {
            x: self.x,
            y: self.x,
            data: ret_q_data
        }
    }

    // Hessenberg matrix decomposition, Arnoldi
    // A = Q ^ H ^ !Q
    pub fn qhq_arnoldi(&self) -> (RMatrix, RMatrix) {
        assert_eq!(self.x, self.y, "RMatrix.qhq_arnoldi(): square matrix only");
        if self.x < 3 {
            return (RMatrix::gen_eye(self.x, self.y), self.clone());
        }
        let mut ret_q_data: Vec<f64> = vec![0.0; self.x * self.x];
        let mut ret_h_data: Vec<f64> = vec![0.0; self.x * self.x];
        let mut q: RMatrix;
        let mut r = RMatrix::gen_rand(self.x, 1);
        let mut b: f64 = r.norm_2();
        for n in 0..self.y {
            q = &r / b;
            r = self ^ &q;
            for i in 0..n {
                for k in 0..self.x {
                    ret_h_data[i * self.x + n] += ret_q_data[k * self.y + i] * r.data[k];
                }
                for k in 0..self.x {
                    r.data[k] -= ret_h_data[i * self.y + n] * ret_q_data[k * self.y + i];
                }
            }
            for k in 0..self.x {
                ret_h_data[n * self.x + n] += q.data[k] * r.data[k];
            }
            for k in 0..self.x {
                r.data[k] -= ret_h_data[n * self.y + n] * q.data[k];
            }
            b = r.norm_2();
            assert_ne!(b, 0.0, "RMatrix.qhq_arnoldi(): break!");
            // for Q
            for i in 0..self.x {
                ret_q_data[i * self.y + n] = q.data[i];
            }
            if (n + 1) < self.y {
                ret_h_data[(n + 1) * self.y + n] = b;
            }
        }
        (RMatrix {
            x: self.x,
            y: self.x,
            data: ret_q_data
        },
        RMatrix {
            x: self.x,
            y: self.x,
            data: ret_h_data
        })
    }

    // tridiagonal matrix decomposition, Householder
    // A must be (skew-)symmetric
    // in place
    // A = Q ^ T ^ !Q
    pub fn iqtq_hh(&mut self) -> RMatrix {
        assert_eq!(self.x, self.y, "RMatrix.qtq_hh(): square matrix only");
        if self.x < 3 {
            return RMatrix::gen_eye(self.x, self.y);
        }
        let mut ret_q_data: Vec<f64> = vec![0.0; self.x * self.x];
        let mut tmp: Vec<f64> = vec![0.0; self.x];
        let mut n2: f64;
        for i in 0..self.x {
            ret_q_data[i * self.x + i] = 1.0;
        }
        for n in 0..(self.y - 1) {
            // Householder
            for i in 0..(n + 1) {
                tmp[i] = 0.0;
            }
            for i in (n + 1)..self.x {
                tmp[i] = self.data[i * self.y + n];
            }
            n2 = 0.0;
            for i in (n + 1)..self.x {
                n2 += tmp[i] * tmp[i];
            }
            if n2 == tmp[n + 1] * tmp[n + 1] {
                continue;
            }
            if tmp[n + 1] > 0.0 {
                tmp[n + 1] += n2.sqrt();
                n2 = (2.0 * n2.sqrt() * tmp[n + 1]).sqrt();
            } else {
                tmp[n + 1] -= n2.sqrt();
                n2 = (-2.0 * n2.sqrt() * tmp[n + 1]).sqrt();
            }
            for i in (n + 1)..self.x {
                tmp[i] /= n2;
            }
            // apply to A
            for k in n..self.y {
                n2 = 0.0;
                for i in (n + 1)..self.x {
                    n2 += self.data[i * self.y + k] * tmp[i];
                }
                for i in (n + 1)..self.x {
                    self.data[i * self.y + k] -= 2.0 * n2 * tmp[i];
                }
            }
            for i in (n + 2)..self.x {
                self.data[i * self.y + n] = 0.0;
            }
            // get Q
            for k in 0..self.x {
                n2 = 0.0;
                for i in (n + 1)..self.x {
                    n2 += ret_q_data[k * self.x + i] * tmp[i];
                }
                for i in (n + 1)..self.x {
                    ret_q_data[k * self.x + i] -= 2.0 * n2 * tmp[i];
                }
            }
            // apply to A
            for k in n..self.x {
                n2 = 0.0;
                for i in (n + 1)..self.y {
                    n2 += self.data[k * self.y + i] * tmp[i];
                }
                for i in (n + 1)..self.y {
                    self.data[k * self.y + i] -= 2.0 * n2 * tmp[i];
                }
            }
            for i in (n + 2)..self.y {
                self.data[n * self.y + i] = 0.0;
            }
        }
        if self.x > 1 {
            if self.data[1] * self.data[self.y] < 0.0 {
                // skew symmetric
                for n in 0..(self.y - 1) {
                    self.data[n * self.y + n + 1] = (self.data[n * self.y + n + 1] - self.data[(n + 1) * self.y + n]) / 2.0;
                    self.data[(n + 1) * self.y + n] = -1.0 * self.data[n * self.y + n + 1];
                    self.data[(n + 1) * self.y + n + 1] = 0.0;
                }
            } else if self.data[1] * self.data[self.y] > 0.0 {
                // symmetric
                for n in 0..(self.y - 1) {
                    self.data[n * self.y + n + 1] = (self.data[n * self.y + n + 1] + self.data[(n + 1) * self.y + n]) / 2.0;
                    self.data[(n + 1) * self.y + n] = self.data[n * self.y + n + 1];
                }
            }
        }
        RMatrix {
            x: self.x,
            y: self.x,
            data: ret_q_data
        }
    }

    // tridiagonal matrix decomposition, Givens
    // A must be (skew-)symmetric
    // in place
    // A = Q ^ T ^ !Q
    pub fn iqtq_givens(&mut self) -> RMatrix {
        assert_eq!(self.x, self.y, "RMatrix.qtq_givens(): square matrix only");
        if self.x < 3 {
            return RMatrix::gen_eye(self.x, self.y);
        }
        let mut ret_q_data: Vec<f64> = vec![0.0; self.x * self.x];
        let mut x: f64;
        let mut y: f64;
        let mut c: f64;
        let mut s: f64;
        for i in 0..self.x {
            ret_q_data[i * self.x + i] = 1.0;
        }
        for n in 0..(self.y - 1) {
            for i in (n + 2)..self.x {
                if self.data[i * self.y + n] == 0.0 {
                    continue;
                }
                // Givens
                x = self.data[(n + 1) * self.y + n];
                y = self.data[i * self.y + n];
                c = x / (x * x + y * y).sqrt();
                s = y / (x * x + y * y).sqrt();
                for k in n..self.y {
                    x = self.data[(n + 1) * self.y + k];
                    y = self.data[i * self.y + k];
                    self.data[(n + 1) * self.y + k] =  x * c + y * s;
                    self.data[i * self.y + k] = -x * s + y * c;
                }
                self.data[i * self.y + n] = 0.0;
                // get Q
                for k in 0..(i + 1) {
                    x = ret_q_data[k * self.x + n + 1];
                    y = ret_q_data[k * self.x + i];
                    ret_q_data[k * self.x + n + 1] =  x * c + y * s;
                    ret_q_data[k * self.x + i] = -x * s + y * c;
                }
                for k in n..self.x {
                    x = self.data[k * self.y + n + 1];
                    y = self.data[k * self.y + i];
                    self.data[k * self.y + n + 1] =  x * c + y * s;
                    self.data[k * self.y + i] = -x * s + y * c;
                }
                self.data[n * self.y + i] = 0.0;
            }
        }
        if self.x > 1 {
            if self.data[1] * self.data[self.y] < 0.0 {
                // skew symmetric
                for n in 0..(self.y - 1) {
                    self.data[n * self.y + n + 1] = (self.data[n * self.y + n + 1] - self.data[(n + 1) * self.y + n]) / 2.0;
                    self.data[(n + 1) * self.y + n] = -1.0 * self.data[n * self.y + n + 1];
                    self.data[(n + 1) * self.y + n + 1] = 0.0;
                }
            } else if self.data[1] * self.data[self.y] > 0.0 {
                // symmetric
                for n in 0..(self.y - 1) {
                    self.data[n * self.y + n + 1] = (self.data[n * self.y + n + 1] + self.data[(n + 1) * self.y + n]) / 2.0;
                    self.data[(n + 1) * self.y + n] = self.data[n * self.y + n + 1];
                }
            }
        }
        RMatrix {
            x: self.x,
            y: self.x,
            data: ret_q_data
        }
    }

    // tridiagonal matrix decomposition, Lanczos
    // A must be (skew-)symmetric
    // A = Q ^ T ^ !Q
    pub fn qtq_lanczos(&self) -> (RMatrix, RMatrix) {
        assert_eq!(self.x, self.y, "RMatrix.qtq_lanczos(): square matrix only");
        if self.x < 3 {
            return (RMatrix::gen_eye(self.x, self.y), self.clone());
        }
        let mut ret_q_data: Vec<f64> = vec![0.0; self.x * self.x];
        let mut ret_t_data: Vec<f64> = vec![0.0; self.x * self.x];
        let mut q = RMatrix::gen_zeros(self.x, 1);
        let mut r = RMatrix::gen_rand(self.x, 1);
        let mut a: f64;
        let mut b: f64 = r.norm_2();
        let mut bq: RMatrix;
        if self.data[1] * self.data[self.y] < 0.0 {
            // skew symmetric
            for n in 0..self.y {
                bq = b * &q;
                q = &r / b;
                r = (self ^ &q) + &bq;
                b = r.norm_2();
                assert_ne!(b, 0.0, "RMatrix.qtq_lanczos(): break!");
                // for Q
                for i in 0..self.x {
                    ret_q_data[i * self.y + n] = q.data[i];
                }
                ret_t_data[n * self.y + n] = 0.0;
                if (n + 1) < self.y {
                    ret_t_data[n * self.y + (n + 1)] =-b;
                    ret_t_data[(n + 1) * self.y + n] = b;
                }
            }
        } else if self.data[1] * self.data[self.y] > 0.0 {
            // symmetric
            for n in 0..self.y {
                bq = b * &q;
                q = &r / b;
                r = (self ^ &q) - &bq;
                a = 0.0;
                for i in 0..self.x {
                    a += q.data[i] * r.data[i];
                }
                for i in 0..self.x {
                    r.data[i] -= a * q.data[i];
                }
                b = r.norm_2();
                assert_ne!(b, 0.0, "RMatrix.qtq_lanczos(): break!");
                // for Q
                for i in 0..self.x {
                    ret_q_data[i * self.y + n] = q.data[i];
                }
                ret_t_data[n * self.y + n] = a;
                if (n + 1) < self.y {
                    ret_t_data[n * self.y + (n + 1)] = b;
                    ret_t_data[(n + 1) * self.y + n] = b;
                }
            }
        }
        (RMatrix {
            x: self.x,
            y: self.x,
            data: ret_q_data
        },
        RMatrix {
            x: self.x,
            y: self.x,
            data: ret_t_data
        })
    }

    // up/dn bidiagonal matrix decomposition, Householder
    // in place
    // A = P ^ B ^ Q
    pub fn ipbq_hh(&mut self) -> (RMatrix, RMatrix) {
        let mut ret_p_data: Vec<f64> = vec![0.0; self.x * self.x];
        let mut ret_q_data: Vec<f64> = vec![0.0; self.y * self.y];
        let mut tmp_x: Vec<f64> = vec![0.0; self.x];
        let mut tmp_y: Vec<f64> = vec![0.0; self.y];
        let mut n2: f64;
        let mut n: usize = 0;
        for i in 0..self.x {
            ret_p_data[i * self.x + i] = 1.0;
        }
        for i in 0..self.y {
            ret_q_data[i * self.y + i] = 1.0;
        }
        if self.x < self.y {
            loop {
                // Householder
                for i in 0..n {
                    tmp_y[i] = 0.0;
                }
                for i in n..self.y {
                    tmp_y[i] = self.data[n * self.y + i];
                }
                n2 = 0.0;
                for i in n..self.y {
                    n2 += tmp_y[i] * tmp_y[i];
                }
                if n2 != tmp_y[n] * tmp_y[n] {
                    if tmp_y[n] > 0.0 {
                        tmp_y[n] += n2.sqrt();
                        n2 = (2.0 * n2.sqrt() * tmp_y[n]).sqrt();
                    } else {
                        tmp_y[n] -= n2.sqrt();
                        n2 = (-2.0 * n2.sqrt() * tmp_y[n]).sqrt();
                    }
                    for i in n..self.y {
                        tmp_y[i] /= n2;
                    }
                    // apply to A
                    for k in n..self.x {
                        n2 = 0.0;
                        for i in n..self.y {
                            n2 += self.data[k * self.y + i] * tmp_y[i];
                        }
                        for i in n..self.y {
                            self.data[k * self.y + i] -= 2.0 * n2 * tmp_y[i];
                        }
                    }
                    for i in (n + 1)..self.y {
                        self.data[n * self.y + i] = 0.0;
                    }
                    // get Q
                    for k in 0..self.y {
                        n2 = 0.0;
                        for i in n..self.y {
                            n2 += ret_q_data[i * self.y + k] * tmp_y[i];
                        }
                        for i in n..self.y {
                            ret_q_data[i * self.y + k] -= 2.0 * n2 * tmp_y[i];
                        }
                    }
                }

                if (n + 1) == self.x {
                    break;
                }

                // Householder
                for i in 0..(n + 1) {
                    tmp_x[i] = 0.0;
                }
                for i in (n + 1)..self.x {
                    tmp_x[i] = self.data[i * self.y + n];
                }
                n2 = 0.0;
                for i in (n + 1)..self.x {
                    n2 += tmp_x[i] * tmp_x[i];
                }
                if n2 != tmp_x[n + 1] * tmp_x[n + 1] {
                    if tmp_x[n + 1] > 0.0 {
                        tmp_x[n + 1] += n2.sqrt();
                        n2 = (2.0 * n2.sqrt() * tmp_x[n + 1]).sqrt();
                    } else {
                        tmp_x[n + 1] -= n2.sqrt();
                        n2 = (-2.0 * n2.sqrt() * tmp_x[n + 1]).sqrt();
                    }
                    for i in (n + 1)..self.x {
                        tmp_x[i] /= n2;
                    }
                    // apply to A
                    for k in n..self.y {
                        n2 = 0.0;
                        for i in (n + 1)..self.x {
                            n2 += self.data[i * self.y + k] * tmp_x[i];
                        }
                        for i in (n + 1)..self.x {
                            self.data[i * self.y + k] -= 2.0 * n2 * tmp_x[i];
                        }
                    }
                    for i in (n + 2)..self.x {
                        self.data[i * self.y + n] = 0.0;
                    }
                    // get P
                    for k in 0..self.x {
                        n2 = 0.0;
                        for i in (n + 1)..self.x {
                            n2 += ret_p_data[k * self.x + i] * tmp_x[i];
                        }
                        for i in (n + 1)..self.x {
                            ret_p_data[k * self.x + i] -= 2.0 * n2 * tmp_x[i];
                        }
                    }
                }

                n += 1;
            }
        } else {
            loop {
                // Householder
                for i in 0..n {
                    tmp_x[i] = 0.0;
                }
                for i in n..self.x {
                    tmp_x[i] = self.data[i * self.y + n];
                }
                n2 = 0.0;
                for i in n..self.x {
                    n2 += tmp_x[i] * tmp_x[i];
                }
                if n2 != tmp_x[n] * tmp_x[n] {
                    if tmp_x[n] > 0.0 {
                        tmp_x[n] += n2.sqrt();
                        n2 = (2.0 * n2.sqrt() * tmp_x[n]).sqrt();
                    } else {
                        tmp_x[n] -= n2.sqrt();
                        n2 = (-2.0 * n2.sqrt() * tmp_x[n]).sqrt();
                    }
                    for i in n..self.x {
                        tmp_x[i] /= n2;
                    }
                    // apply to A
                    for k in n..self.y {
                        n2 = 0.0;
                        for i in n..self.x {
                            n2 += self.data[i * self.y + k] * tmp_x[i];
                        }
                        for i in n..self.x {
                            self.data[i * self.y + k] -= 2.0 * n2 * tmp_x[i];
                        }
                    }
                    for i in (n + 1)..self.x {
                        self.data[i * self.y + n] = 0.0;
                    }
                    // get P
                    for k in 0..self.x {
                        n2 = 0.0;
                        for i in n..self.x {
                            n2 += ret_p_data[k * self.x + i] * tmp_x[i];
                        }
                        for i in n..self.x {
                            ret_p_data[k * self.x + i] -= 2.0 * n2 * tmp_x[i];
                        }
                    }
                }

                if (n + 1) == self.y {
                    break;
                }

                // Householder
                for i in 0..(n + 1) {
                    tmp_y[i] = 0.0;
                }
                for i in (n + 1)..self.y {
                    tmp_y[i] = self.data[n * self.y + i];
                }
                n2 = 0.0;
                for i in (n + 1)..self.y {
                    n2 += tmp_y[i] * tmp_y[i];
                }
                if n2 != tmp_y[n + 1] * tmp_y[n + 1] {
                    if tmp_y[n + 1] > 0.0 {
                        tmp_y[n + 1] += n2.sqrt();
                        n2 = (2.0 * n2.sqrt() * tmp_y[n + 1]).sqrt();
                    } else {
                        tmp_y[n + 1] -= n2.sqrt();
                        n2 = (-2.0 * n2.sqrt() * tmp_y[n + 1]).sqrt();
                    }
                    for i in (n + 1)..self.y {
                        tmp_y[i] /= n2;
                    }
                    // apply to A
                    for k in n..self.x {
                        n2 = 0.0;
                        for i in (n + 1)..self.y {
                            n2 += self.data[k * self.y + i] * tmp_y[i];
                        }
                        for i in (n + 1)..self.y {
                            self.data[k * self.y + i] -= 2.0 * n2 * tmp_y[i];
                        }
                    }
                    for i in (n + 2)..self.y {
                        self.data[n * self.y + i] = 0.0;
                    }
                    // get Q
                    for k in 0..self.y {
                        n2 = 0.0;
                        for i in (n + 1)..self.y {
                            n2 += ret_q_data[i * self.y + k] * tmp_y[i];
                        }
                        for i in (n + 1)..self.y {
                            ret_q_data[i * self.y + k] -= 2.0 * n2 * tmp_y[i];
                        }
                    }
                }

                n += 1;
            }
        }
        (RMatrix {
            x: self.x,
            y: self.x,
            data: ret_p_data
        },
        RMatrix {
            x: self.y,
            y: self.y,
            data: ret_q_data
        })
    }

    // up/dn bidiagonal matrix decomposition, Givens
    // in place
    // A = P ^ B ^ Q
    pub fn ipbq_givens(&mut self) -> (RMatrix, RMatrix) {
        let mut ret_p_data: Vec<f64> = vec![0.0; self.x * self.x];
        let mut ret_q_data: Vec<f64> = vec![0.0; self.y * self.y];
        let mut x: f64;
        let mut y: f64;
        let mut c: f64;
        let mut s: f64;
        for i in 0..self.x {
            ret_p_data[i * self.x + i] = 1.0;
        }
        for i in 0..self.y {
            ret_q_data[i * self.y + i] = 1.0;
        }
        if self.x < self.y {
            for n in 0..self.x {
                for j in (n + 1)..self.y {
                    if self.data[n * self.y + j] == 0.0 {
                        continue;
                    }
                    // Givens
                    x = self.data[n * self.y + n];
                    y = self.data[n * self.y + j];
                    c = x / (x * x + y * y).sqrt();
                    s = y / (x * x + y * y).sqrt();
                    for k in n..self.x {
                        x = self.data[k * self.y + n];
                        y = self.data[k * self.y + j];
                        self.data[k * self.y + n] =  x * c + y * s;
                        self.data[k * self.y + j] = -x * s + y * c;
                    }
                    self.data[n * self.y + j] = 0.0;
                    // get Q
                    for k in 0..(j + 1) {
                        x = ret_q_data[n * self.y + k];
                        y = ret_q_data[j * self.y + k];
                        ret_q_data[n * self.y + k] =  x * c + y * s;
                        ret_q_data[j * self.y + k] = -x * s + y * c;
                    }
                }
                for i in (n + 2)..self.x {
                    if self.data[i * self.y + n] == 0.0 {
                        continue;
                    }
                    // Givens
                    x = self.data[(n + 1) * self.y + n];
                    y = self.data[i * self.y + n];
                    c = x / (x * x + y * y).sqrt();
                    s = y / (x * x + y * y).sqrt();
                    for k in n..self.y {
                        x = self.data[(n + 1) * self.y + k];
                        y = self.data[i * self.y + k];
                        self.data[(n + 1) * self.y + k] =  x * c + y * s;
                        self.data[i * self.y + k] = -x * s + y * c;
                    }
                    self.data[i * self.y + n] = 0.0;
                    // get P
                    for k in 0..(i + 1) {
                        x = ret_p_data[k * self.x + n + 1];
                        y = ret_p_data[k * self.x + i];
                        ret_p_data[k * self.x + n + 1] =  x * c + y * s;
                        ret_p_data[k * self.x + i] = -x * s + y * c;
                    }
                }
            }
        } else {
            for n in 0..self.y {
                for i in (n + 1)..self.x {
                    if self.data[i * self.y + n] == 0.0 {
                        continue;
                    }
                    // Givens
                    x = self.data[n * self.y + n];
                    y = self.data[i * self.y + n];
                    c = x / (x * x + y * y).sqrt();
                    s = y / (x * x + y * y).sqrt();
                    for k in n..self.y {
                        x = self.data[n * self.y + k];
                        y = self.data[i * self.y + k];
                        self.data[n * self.y + k] =  x * c + y * s;
                        self.data[i * self.y + k] = -x * s + y * c;
                    }
                    self.data[i * self.y + n] = 0.0;
                    // get P
                    for k in 0..(i + 1) {
                        x = ret_p_data[k * self.x + n];
                        y = ret_p_data[k * self.x + i];
                        ret_p_data[k * self.x + n] =  x * c + y * s;
                        ret_p_data[k * self.x + i] = -x * s + y * c;
                    }
                }
                for j in (n + 2)..self.y {
                    if self.data[n * self.y + j] == 0.0 {
                        continue;
                    }
                    // Givens
                    x = self.data[n * self.y + n + 1];
                    y = self.data[n * self.y + j];
                    c = x / (x * x + y * y).sqrt();
                    s = y / (x * x + y * y).sqrt();
                    for k in n..self.x {
                        x = self.data[k * self.y + n + 1];
                        y = self.data[k * self.y + j];
                        self.data[k * self.y + n + 1] =  x * c + y * s;
                        self.data[k * self.y + j] = -x * s + y * c;
                    }
                    self.data[n * self.y + j] = 0.0;
                    // get Q
                    for k in 0..(j + 1) {
                        x = ret_q_data[(n + 1) * self.y + k];
                        y = ret_q_data[j * self.y + k];
                        ret_q_data[(n + 1) * self.y + k] =  x * c + y * s;
                        ret_q_data[j * self.y + k] = -x * s + y * c;
                    }
                }
            }
        }
        (RMatrix {
            x: self.x,
            y: self.x,
            data: ret_p_data
        },
        RMatrix {
            x: self.y,
            y: self.y,
            data: ret_q_data
        })
    }

    // QR decomposition for up bidiagonal matrix, Givens
    // in place
    // B = R ^ Q
    // for SVD
    fn ibqr_givens(&mut self, q: &mut RMatrix) {
        let mut x: f64;
        let mut y: f64;
        let mut c: f64;
        let mut s: f64;
        let n: usize = self.x.min(self.y);
        for i in 0..(n - 1) {
            if self.data[i * self.y + i + 1] == 0.0 {
                continue;
            }
            // Givens
            x = self.data[i * self.y + i];
            y = self.data[i * self.y + i + 1];
            c = x / (x * x + y * y).sqrt();
            s = y / (x * x + y * y).sqrt();
            self.data[i * self.y + i] =  x * c + y * s;
            self.data[i * self.y + i + 1] = 0.0;
            x = self.data[(i + 1) * self.y + i];
            y = self.data[(i + 1) * self.y + i + 1];
            self.data[(i + 1) * self.y + i] =  x * c + y * s;
            self.data[(i + 1) * self.y + i + 1] = -x * s + y * c;
            // apply to Q
            for k in 0..self.y {
                x = q.data[i * self.y + k];
                y = q.data[(i + 1) * self.y + k];
                q.data[i * self.y + k] =  x * c + y * s;
                q.data[(i + 1) * self.y + k] = -x * s + y * c;
            }
        }
    }

    // QR decomposition for down bidiagonal matrix, Givens
    // in place
    // B = P ^ R
    // for SVD
    fn ibpr_givens(&mut self, p: &mut RMatrix) {
        let mut x: f64;
        let mut y: f64;
        let mut c: f64;
        let mut s: f64;
        let n: usize = self.x.min(self.y);
        for i in 0..(n - 1) {
            if self.data[(i + 1) * self.y + i] == 0.0 {
                continue;
            }
            // Givens
            x = self.data[i * self.y + i];
            y = self.data[(i + 1) * self.y + i];
            c = x / (x * x + y * y).sqrt();
            s = y / (x * x + y * y).sqrt();
            self.data[i * self.y + i] =  x * c + y * s;
            self.data[(i + 1) * self.y + i] = 0.0;
            x = self.data[i * self.y + i + 1];
            y = self.data[(i + 1) * self.y + i + 1];
            self.data[i * self.y + i + 1] =  x * c + y * s;
            self.data[(i + 1) * self.y + i + 1] = -x * s + y * c;
            // apply to P
            for k in 0..self.x {
                x = p.data[k * self.x + i];
                y = p.data[k * self.x + i + 1];
                p.data[k * self.x + i] =  x * c + y * s;
                p.data[k * self.x + i + 1] = -x * s + y * c;
            }
        }
    }

    // two QR decompositions for bidiagonal matrix, Givens
    // in place
    // B = P ^ R ^ Q
    // for SVD
    fn ibpqr_givens(&mut self, p: &mut RMatrix, q: &mut RMatrix) {
        if self.x < self.y {
            self.ibpr_givens(p);
            self.ibqr_givens(q);
        } else {
            self.ibqr_givens(q);
            self.ibpr_givens(p);
        }
    }

    // Hessenberg matrix decomposition, Householder
    // in place
    // without Q
    pub fn ih_hh(&mut self) {
        assert_eq!(self.x, self.y, "RMatrix.h_hh(): square matrix only");
        if self.x < 3 {
            return;
        }
        let mut tmp: Vec<f64> = vec![0.0; self.x];
        let mut n2: f64;
        for n in 0..(self.y - 1) {
            // Householder
            for i in 0..(n + 1) {
                tmp[i] = 0.0;
            }
            for i in (n + 1)..self.x {
                tmp[i] = self.data[i * self.y + n];
            }
            n2 = 0.0;
            for i in (n + 1)..self.x {
                n2 += tmp[i] * tmp[i];
            }
            if n2 == tmp[n + 1] * tmp[n + 1] {
                continue;
            }
            if tmp[n + 1] > 0.0 {
                tmp[n + 1] += n2.sqrt();
                n2 = (2.0 * n2.sqrt() * tmp[n + 1]).sqrt();
            } else {
                tmp[n + 1] -= n2.sqrt();
                n2 = (-2.0 * n2.sqrt() * tmp[n + 1]).sqrt();
            }
            for i in (n + 1)..self.x {
                tmp[i] /= n2;
            }
            // apply to A
            for k in n..self.y {
                n2 = 0.0;
                for i in (n + 1)..self.x {
                    n2 += self.data[i * self.y + k] * tmp[i];
                }
                for i in (n + 1)..self.x {
                    self.data[i * self.y + k] -= 2.0 * n2 * tmp[i];
                }
            }
            for i in (n + 2)..self.x {
                self.data[i * self.y + n] = 0.0;
            }
            // apply to A
            for k in 0..self.x {
                n2 = 0.0;
                for i in (n + 1)..self.y {
                    n2 += self.data[k * self.y + i] * tmp[i];
                }
                for i in (n + 1)..self.y {
                    self.data[k * self.y + i] -= 2.0 * n2 * tmp[i];
                }
            }
        }
    }

    // Hessenberg matrix decomposition, Givens
    // in place
    // without Q
    pub fn ih_givens(&mut self) {
        assert_eq!(self.x, self.y, "RMatrix.h_givens(): square matrix only");
        if self.x < 3 {
            return;
        }
        let mut x: f64;
        let mut y: f64;
        let mut c: f64;
        let mut s: f64;
        for n in 0..(self.y - 1) {
            for i in (n + 2)..self.x {
                if self.data[i * self.y + n] == 0.0 {
                    continue;
                }
                // Givens
                x = self.data[(n + 1) * self.y + n];
                y = self.data[i * self.y + n];
                c = x / (x * x + y * y).sqrt();
                s = y / (x * x + y * y).sqrt();
                for k in n..self.y {
                    x = self.data[(n + 1) * self.y + k];
                    y = self.data[i * self.y + k];
                    self.data[(n + 1) * self.y + k] =  x * c + y * s;
                    self.data[i * self.y + k] = -x * s + y * c;
                }
                self.data[i * self.y + n] = 0.0;
                for k in 0..self.x {
                    x = self.data[k * self.y + n + 1];
                    y = self.data[k * self.y + i];
                    self.data[k * self.y + n + 1] =  x * c + y * s;
                    self.data[k * self.y + i] = -x * s + y * c;
                }
            }
        }
    }

    // tridiagonal matrix decomposition, Householder
    // A must be (skew-)symmetric
    // in place
    // without Q
    pub fn it_hh(&mut self) {
        assert_eq!(self.x, self.y, "RMatrix.t_hh(): square matrix only");
        if self.x < 3 {
            return;
        }
        let mut tmp: Vec<f64> = vec![0.0; self.x];
        let mut n2: f64;
        for n in 0..(self.y - 1) {
            // Householder
            for i in 0..(n + 1) {
                tmp[i] = 0.0;
            }
            for i in (n + 1)..self.x {
                tmp[i] = self.data[i * self.y + n];
            }
            n2 = 0.0;
            for i in (n + 1)..self.x {
                n2 += tmp[i] * tmp[i];
            }
            if n2 == tmp[n + 1] * tmp[n + 1] {
                continue;
            }
            if tmp[n + 1] > 0.0 {
                tmp[n + 1] += n2.sqrt();
                n2 = (2.0 * n2.sqrt() * tmp[n + 1]).sqrt();
            } else {
                tmp[n + 1] -= n2.sqrt();
                n2 = (-2.0 * n2.sqrt() * tmp[n + 1]).sqrt();
            }
            for i in (n + 1)..self.x {
                tmp[i] /= n2;
            }
            // apply to A
            for k in n..self.y {
                n2 = 0.0;
                for i in (n + 1)..self.x {
                    n2 += self.data[i * self.y + k] * tmp[i];
                }
                for i in (n + 1)..self.x {
                    self.data[i * self.y + k] -= 2.0 * n2 * tmp[i];
                }
            }
            for i in (n + 2)..self.x {
                self.data[i * self.y + n] = 0.0;
            }
            // apply to A
            for k in n..self.x {
                n2 = 0.0;
                for i in (n + 1)..self.y {
                    n2 += self.data[k * self.y + i] * tmp[i];
                }
                for i in (n + 1)..self.y {
                    self.data[k * self.y + i] -= 2.0 * n2 * tmp[i];
                }
            }
            for i in (n + 2)..self.y {
                self.data[n * self.y + i] = 0.0;
            }
        }
        if self.x > 1 {
            if self.data[1] * self.data[self.y] < 0.0 {
                // skew symmetric
                for n in 0..(self.y - 1) {
                    self.data[n * self.y + n + 1] = (self.data[n * self.y + n + 1] - self.data[(n + 1) * self.y + n]) / 2.0;
                    self.data[(n + 1) * self.y + n] = -1.0 * self.data[n * self.y + n + 1];
                    self.data[(n + 1) * self.y + n + 1] = 0.0;
                }
            } else if self.data[1] * self.data[self.y] > 0.0 {
                // symmetric
                for n in 0..(self.y - 1) {
                    self.data[n * self.y + n + 1] = (self.data[n * self.y + n + 1] + self.data[(n + 1) * self.y + n]) / 2.0;
                    self.data[(n + 1) * self.y + n] = self.data[n * self.y + n + 1];
                }
            }
        }
    }

    // tridiagonal matrix decomposition, Givens
    // A must be (skew-)symmetric
    // in place
    // without Q
    pub fn it_givens(&mut self) {
        assert_eq!(self.x, self.y, "RMatrix.t_givens(): square matrix only");
        if self.x < 3 {
            return;
        }
        let mut x: f64;
        let mut y: f64;
        let mut c: f64;
        let mut s: f64;
        for n in 0..(self.y - 1) {
            for i in (n + 2)..self.x {
                if self.data[i * self.y + n] == 0.0 {
                    continue;
                }
                // Givens
                x = self.data[(n + 1) * self.y + n];
                y = self.data[i * self.y + n];
                c = x / (x * x + y * y).sqrt();
                s = y / (x * x + y * y).sqrt();
                for k in n..self.y {
                    x = self.data[(n + 1) * self.y + k];
                    y = self.data[i * self.y + k];
                    self.data[(n + 1) * self.y + k] =  x * c + y * s;
                    self.data[i * self.y + k] = -x * s + y * c;
                }
                self.data[i * self.y + n] = 0.0;
                for k in n..self.x {
                    x = self.data[k * self.y + n + 1];
                    y = self.data[k * self.y + i];
                    self.data[k * self.y + n + 1] =  x * c + y * s;
                    self.data[k * self.y + i] = -x * s + y * c;
                }
                self.data[n * self.y + i] = 0.0;
            }
        }
        if self.x > 1 {
            if self.data[1] * self.data[self.y] < 0.0 {
                // skew symmetric
                for n in 0..(self.y - 1) {
                    self.data[n * self.y + n + 1] = (self.data[n * self.y + n + 1] - self.data[(n + 1) * self.y + n]) / 2.0;
                    self.data[(n + 1) * self.y + n] = -1.0 * self.data[n * self.y + n + 1];
                    self.data[(n + 1) * self.y + n + 1] = 0.0;
                }
            } else if self.data[1] * self.data[self.y] > 0.0 {
                // symmetric
                for n in 0..(self.y - 1) {
                    self.data[n * self.y + n + 1] = (self.data[n * self.y + n + 1] + self.data[(n + 1) * self.y + n]) / 2.0;
                    self.data[(n + 1) * self.y + n] = self.data[n * self.y + n + 1];
                }
            }
        }
    }

    // up/dn bidiagonal matrix decomposition, Householder
    // in place
    // without P or Q
    pub fn ib_hh(&mut self) {
        let mut tmp_x: Vec<f64> = vec![0.0; self.x];
        let mut tmp_y: Vec<f64> = vec![0.0; self.y];
        let mut n2: f64;
        let mut n: usize = 0;
        if self.x < self.y {
            loop {
                // Householder
                for i in 0..n {
                    tmp_y[i] = 0.0;
                }
                for i in n..self.y {
                    tmp_y[i] = self.data[n * self.y + i];
                }
                n2 = 0.0;
                for i in n..self.y {
                    n2 += tmp_y[i] * tmp_y[i];
                }
                if n2 != tmp_y[n] * tmp_y[n] {
                    if tmp_y[n] > 0.0 {
                        tmp_y[n] += n2.sqrt();
                        n2 = (2.0 * n2.sqrt() * tmp_y[n]).sqrt();
                    } else {
                        tmp_y[n] -= n2.sqrt();
                        n2 = (-2.0 * n2.sqrt() * tmp_y[n]).sqrt();
                    }
                    for i in n..self.y {
                        tmp_y[i] /= n2;
                    }
                    // apply to A
                    for k in n..self.x {
                        n2 = 0.0;
                        for i in n..self.y {
                            n2 += self.data[k * self.y + i] * tmp_y[i];
                        }
                        for i in n..self.y {
                            self.data[k * self.y + i] -= 2.0 * n2 * tmp_y[i];
                        }
                    }
                    for i in (n + 1)..self.y {
                        self.data[n * self.y + i] = 0.0;
                    }
                }

                if (n + 1) == self.x {
                    break;
                }

                // Householder
                for i in 0..(n + 1) {
                    tmp_x[i] = 0.0;
                }
                for i in (n + 1)..self.x {
                    tmp_x[i] = self.data[i * self.y + n];
                }
                n2 = 0.0;
                for i in (n + 1)..self.x {
                    n2 += tmp_x[i] * tmp_x[i];
                }
                if n2 != tmp_x[n + 1] * tmp_x[n + 1] {
                    if tmp_x[n + 1] > 0.0 {
                        tmp_x[n + 1] += n2.sqrt();
                        n2 = (2.0 * n2.sqrt() * tmp_x[n + 1]).sqrt();
                    } else {
                        tmp_x[n + 1] -= n2.sqrt();
                        n2 = (-2.0 * n2.sqrt() * tmp_x[n + 1]).sqrt();
                    }
                    for i in (n + 1)..self.x {
                        tmp_x[i] /= n2;
                    }
                    // apply to A
                    for k in n..self.y {
                        n2 = 0.0;
                        for i in (n + 1)..self.x {
                            n2 += self.data[i * self.y + k] * tmp_x[i];
                        }
                        for i in (n + 1)..self.x {
                            self.data[i * self.y + k] -= 2.0 * n2 * tmp_x[i];
                        }
                    }
                    for i in (n + 2)..self.x {
                        self.data[i * self.y + n] = 0.0;
                    }
                }

                n += 1;
            }
        } else {
            loop {
                // Householder
                for i in 0..n {
                    tmp_x[i] = 0.0;
                }
                for i in n..self.x {
                    tmp_x[i] = self.data[i * self.y + n];
                }
                n2 = 0.0;
                for i in n..self.x {
                    n2 += tmp_x[i] * tmp_x[i];
                }
                if n2 != tmp_x[n] * tmp_x[n] {
                    if tmp_x[n] > 0.0 {
                        tmp_x[n] += n2.sqrt();
                        n2 = (2.0 * n2.sqrt() * tmp_x[n]).sqrt();
                    } else {
                        tmp_x[n] -= n2.sqrt();
                        n2 = (-2.0 * n2.sqrt() * tmp_x[n]).sqrt();
                    }
                    for i in n..self.x {
                        tmp_x[i] /= n2;
                    }
                    // apply to A
                    for k in n..self.y {
                        n2 = 0.0;
                        for i in n..self.x {
                            n2 += self.data[i * self.y + k] * tmp_x[i];
                        }
                        for i in n..self.x {
                            self.data[i * self.y + k] -= 2.0 * n2 * tmp_x[i];
                        }
                    }
                    for i in (n + 1)..self.x {
                        self.data[i * self.y + n] = 0.0;
                    }
                }

                if (n + 1) == self.y {
                    break;
                }

                // Householder
                for i in 0..(n + 1) {
                    tmp_y[i] = 0.0;
                }
                for i in (n + 1)..self.y {
                    tmp_y[i] = self.data[n * self.y + i];
                }
                n2 = 0.0;
                for i in (n + 1)..self.y {
                    n2 += tmp_y[i] * tmp_y[i];
                }
                if n2 != tmp_y[n + 1] * tmp_y[n + 1] {
                    if tmp_y[n + 1] > 0.0 {
                        tmp_y[n + 1] += n2.sqrt();
                        n2 = (2.0 * n2.sqrt() * tmp_y[n + 1]).sqrt();
                    } else {
                        tmp_y[n + 1] -= n2.sqrt();
                        n2 = (-2.0 * n2.sqrt() * tmp_y[n + 1]).sqrt();
                    }
                    for i in (n + 1)..self.y {
                        tmp_y[i] /= n2;
                    }
                    // apply to A
                    for k in n..self.x {
                        n2 = 0.0;
                        for i in (n + 1)..self.y {
                            n2 += self.data[k * self.y + i] * tmp_y[i];
                        }
                        for i in (n + 1)..self.y {
                            self.data[k * self.y + i] -= 2.0 * n2 * tmp_y[i];
                        }
                    }
                    for i in (n + 2)..self.y {
                        self.data[n * self.y + i] = 0.0;
                    }
                }

                n += 1;
            }
        }
    }

    // up/dn bidiagonal matrix decomposition, Givens
    // in place
    // without P or Q
    pub fn ib_givens(&mut self) {
        let mut x: f64;
        let mut y: f64;
        let mut c: f64;
        let mut s: f64;
        if self.x < self.y {
            for n in 0..self.x {
                for j in (n + 1)..self.y {
                    if self.data[n * self.y + j] == 0.0 {
                        continue;
                    }
                    // Givens
                    x = self.data[n * self.y + n];
                    y = self.data[n * self.y + j];
                    c = x / (x * x + y * y).sqrt();
                    s = y / (x * x + y * y).sqrt();
                    for k in n..self.x {
                        x = self.data[k * self.y + n];
                        y = self.data[k * self.y + j];
                        self.data[k * self.y + n] =  x * c + y * s;
                        self.data[k * self.y + j] = -x * s + y * c;
                    }
                    self.data[n * self.y + j] = 0.0;
                }
                for i in (n + 2)..self.x {
                    if self.data[i * self.y + n] == 0.0 {
                        continue;
                    }
                    // Givens
                    x = self.data[(n + 1) * self.y + n];
                    y = self.data[i * self.y + n];
                    c = x / (x * x + y * y).sqrt();
                    s = y / (x * x + y * y).sqrt();
                    for k in n..self.y {
                        x = self.data[(n + 1) * self.y + k];
                        y = self.data[i * self.y + k];
                        self.data[(n + 1) * self.y + k] =  x * c + y * s;
                        self.data[i * self.y + k] = -x * s + y * c;
                    }
                    self.data[i * self.y + n] = 0.0;
                }
            }
        } else {
            for n in 0..self.y {
                for i in (n + 1)..self.x {
                    if self.data[i * self.y + n] == 0.0 {
                        continue;
                    }
                    // Givens
                    x = self.data[n * self.y + n];
                    y = self.data[i * self.y + n];
                    c = x / (x * x + y * y).sqrt();
                    s = y / (x * x + y * y).sqrt();
                    for k in n..self.y {
                        x = self.data[n * self.y + k];
                        y = self.data[i * self.y + k];
                        self.data[n * self.y + k] =  x * c + y * s;
                        self.data[i * self.y + k] = -x * s + y * c;
                    }
                    self.data[i * self.y + n] = 0.0;
                }
                for j in (n + 2)..self.y {
                    if self.data[n * self.y + j] == 0.0 {
                        continue;
                    }
                    // Givens
                    x = self.data[n * self.y + n + 1];
                    y = self.data[n * self.y + j];
                    c = x / (x * x + y * y).sqrt();
                    s = y / (x * x + y * y).sqrt();
                    for k in n..self.x {
                        x = self.data[k * self.y + n + 1];
                        y = self.data[k * self.y + j];
                        self.data[k * self.y + n + 1] =  x * c + y * s;
                        self.data[k * self.y + j] = -x * s + y * c;
                    }
                    self.data[n * self.y + j] = 0.0;
                }
            }
        }
    }

    // QR decomposition for up bidiagonal matrix, Givens
    // in place
    // without Q
    // for SVD
    fn iubr_givens(&mut self) {
        let mut x: f64;
        let mut y: f64;
        let mut c: f64;
        let mut s: f64;
        let n: usize = self.x.min(self.y);
        for i in 0..(n - 1) {
            if self.data[i * self.y + i + 1] == 0.0 {
                continue;
            }
            // Givens
            x = self.data[i * self.y + i];
            y = self.data[i * self.y + i + 1];
            c = x / (x * x + y * y).sqrt();
            s = y / (x * x + y * y).sqrt();
            self.data[i * self.y + i] =  x * c + y * s;
            self.data[i * self.y + i + 1] = 0.0;
            x = self.data[(i + 1) * self.y + i];
            y = self.data[(i + 1) * self.y + i + 1];
            self.data[(i + 1) * self.y + i] =  x * c + y * s;
            self.data[(i + 1) * self.y + i + 1] = -x * s + y * c;
        }
    }

    // QR decomposition for down bidiagonal matrix, Givens
    // in place
    // without P
    // for SVD
    fn idbr_givens(&mut self) {
        let mut x: f64;
        let mut y: f64;
        let mut c: f64;
        let mut s: f64;
        let n: usize = self.x.min(self.y);
        for i in 0..(n - 1) {
            if self.data[(i + 1) * self.y + i] == 0.0 {
                continue;
            }
            // Givens
            x = self.data[i * self.y + i];
            y = self.data[(i + 1) * self.y + i];
            c = x / (x * x + y * y).sqrt();
            s = y / (x * x + y * y).sqrt();
            self.data[i * self.y + i] =  x * c + y * s;
            self.data[(i + 1) * self.y + i] = 0.0;
            x = self.data[i * self.y + i + 1];
            y = self.data[(i + 1) * self.y + i + 1];
            self.data[i * self.y + i + 1] =  x * c + y * s;
            self.data[(i + 1) * self.y + i + 1] = -x * s + y * c;
        }
    }

    // two QR decompositions for bidiagonal matrix, Givens
    // in place
    // without P or Q
    // for SVD
    fn ibr_givens(&mut self) {
        if self.x < self.y {
            self.idbr_givens();
            self.iubr_givens();
        } else {
            self.iubr_givens();
            self.idbr_givens();
        }
    }

    // singular value decomposition(SVD)
    // in place
    // A = U ^ S ^ V
    pub fn isvd_qr(&mut self) -> (RMatrix, RMatrix) {
        let (mut u, mut v) = self.ipbq_hh();
        let n: usize = self.x.min(self.y);
        let mut n1: f64;
        let mut n2: f64;
        let mut delta: f64 = std::f64::MAX;
        while delta > 0.000000000000000000001 {
            self.ibpqr_givens(&mut u, &mut v);
            n1 = 0.0;
            n2 = self.data[0] * self.data[0];
            for i in 1..n {
                n1 += self.data[i * self.y + i - 1] * self.data[i * self.y + i - 1];
                n1 += self.data[(i - 1) * self.y + i] * self.data[(i - 1) * self.y + i];
                n2 += self.data[i * self.y + i] * self.data[i * self.y + i];
            }
            delta = (n1 / n2).sqrt();
        }
        // clean up
        for i in 1..n {
            self.data[i * self.y + (i - 1)] = 0.0;
            self.data[(i - 1) * self.y + i] = 0.0;
        }
        // caused by householder
        if self.x < self.y {
            for k in 0..self.x {
                if self.data[k * self.y + k] < 0.0 {
                    self.data[k * self.y + k] *= -1.0;
                    for i in 0..self.x {
                        u.data[i * self.x + k] *= -1.0;
                    }
                }
            }
        } else {
            for k in 0..self.y {
                if self.data[k * self.y + k] < 0.0 {
                    self.data[k * self.y + k] *= -1.0;
                    for j in 0..self.y {
                        v.data[k * self.y + j] *= -1.0;
                    }
                }
            }
        }
        (u, v)
    }

    // singular value decomposition(SVD)
    // in place
    // without U or V
    pub fn isv_qr(&mut self) {
        self.ib_hh();
        let n: usize = self.x.min(self.y);
        let mut n1: f64;
        let mut n2: f64;
        let mut delta: f64 = std::f64::MAX;
        while delta > 0.000000000000000000001 {
            self.ibr_givens();
            n1 = 0.0;
            n2 = self.data[0] * self.data[0];
            for i in 1..n {
                n1 += self.data[i * self.y + i - 1] * self.data[i * self.y + i - 1];
                n1 += self.data[(i - 1) * self.y + i] * self.data[(i - 1) * self.y + i];
                n2 += self.data[i * self.y + i] * self.data[i * self.y + i];
            }
            delta = (n1 / n2).sqrt();
        }
        // clean up
        for i in 1..n {
            self.data[i * self.y + (i - 1)] = 0.0;
            self.data[(i - 1) * self.y + i] = 0.0;
        }
        // caused by householder
        for k in 0..self.x.min(self.y) {
            if self.data[k * self.y + k] < 0.0 {
                self.data[k * self.y + k] *= -1.0;
            }
        }
    }

    // solve a linear system, Cholesky
    // A must be symmetric, positive-define
    // A ^ x = b
    pub fn solve_chol(&self, b: &RMatrix) -> RMatrix {
        assert_eq!(self.x, self.y, "RMatrix.solve_chol(&RMatrix): square matrix only");
        assert_eq!(1, b.y, "RMatrix.solve_chol(&RMatrix): b must be vector");
        assert_eq!(self.x, b.x, "RMatrix.solve_chol(&RMatrix): matrix size mismatch");
        let l = self.chol();
        let mut y = b.clone();
        for i in 0..self.x {
            y.data[i] /= l.data[i * self.y + i];
            for j in (i + 1)..self.x {
                y.data[j] -= y.data[i] * l.data[j * self.y + i];
            }
        }
        let mut x = y;
        for i in (0..self.x).rev() {
            x.data[i] /= l.data[i * self.y + i];
            for j in 0..i {
                x.data[j] -= x.data[i] * l.data[i * self.y + j];
            }
        }
        x
    }

    // solve a tridiagonal linear system, Cholesky
    // A must be symmetric, positive-define
    // A ^ x = b
    pub fn solve_tri_chol(&self, b: &RMatrix) -> RMatrix {
        assert_eq!(self.x, self.y, "RMatrix.solve_tri_chol(&RMatrix): square matrix only");
        assert_eq!(1, b.y, "RMatrix.solve_tri_chol(&RMatrix): b must be vector");
        assert_eq!(self.x, b.x, "RMatrix.solve_tri_chol(&RMatrix): matrix size mismatch");
        let l = self.chol_tri();
        let mut y = b.clone();
        for i in 0..(self.x - 1) {
            y.data[i] /= l.data[i * self.y + i];
            y.data[i + 1] -= y.data[i] * l.data[(i + 1) * self.y + i];
        }
        y.data[self.x - 1] /= l.data[(self.x - 1) * (self.y + 1)];
        let mut x = y;
        for i in (1..self.x).rev() {
            x.data[i] /= l.data[i * self.y + i];
            x.data[i - 1] -= x.data[i] * l.data[i * self.y + (i - 1)];
        }
        x.data[0] /= l.data[0];
        x
    }

    // solve a linear system, LU
    // A ^ x = b
    pub fn solve_lu(&self, b: &RMatrix) -> RMatrix {
        assert_eq!(self.x, self.y, "RMatrix.solve_lu(&RMatrix): square matrix only");
        assert_eq!(1, b.y, "RMatrix.solve_lu(&RMatrix): b must be vector");
        assert_eq!(self.x, b.x, "RMatrix.solve_lu(&RMatrix): matrix size mismatch");
        let (l, u) = self.lu();
        let mut y = b.clone();
        for i in 0..self.x {
            y.data[i] /= l.data[i * self.y + i];
            for j in (i + 1)..self.x {
                y.data[j] -= y.data[i] * l.data[j * self.y + i];
            }
        }
        let mut x = y;
        for i in (0..self.x).rev() {
            x.data[i] /= u.data[i * self.y + i];
            for j in 0..i {
                x.data[j] -= x.data[i] * u.data[j * self.y + i];
            }
        }
        x
    }

    // solve a linear system, LU
    // A ^ x = b
    pub fn solve_tri_lu(&self, b: &RMatrix) -> RMatrix {
        assert_eq!(self.x, self.y, "RMatrix.solve_tri_lu(&RMatrix): square matrix only");
        assert_eq!(1, b.y, "RMatrix.solve_tri_lu(&RMatrix): b must be vector");
        assert_eq!(self.x, b.x, "RMatrix.solve_tri_lu(&RMatrix): matrix size mismatch");
        let (l, u) = self.lu_tri();
        let mut y = b.clone();
        for i in 0..(self.x - 1) {
            y.data[i] /= l.data[i * self.y + i];
            y.data[i + 1] -= y.data[i] * l.data[(i + 1) * self.y + i];
        }
        y.data[self.x - 1] /= l.data[(self.x - 1) * (self.y + 1)];
        let mut x = y;
        for i in (1..self.x).rev() {
            x.data[i] /= u.data[i * self.y + i];
            x.data[i - 1] -= x.data[i] * u.data[(i - 1) * self.y + i];
        }
        x.data[0] /= u.data[0];
        x
    }

    // solve a linear system, gradient descent
    // A must be symmetric, positive-define
    // A ^ x = b
    pub fn solve_gd(&self, b: &RMatrix) -> RMatrix {
        assert_eq!(self.x, self.y, "RMatrix.solve_gd(&RMatrix): square matrix only");
        assert_eq!(1, b.y, "RMatrix.solve_gd(&RMatrix): b must be vector");
        assert_eq!(self.x, b.x, "RMatrix.solve_gd(&RMatrix): matrix size mismatch");
        let mut x = RMatrix::gen_zeros(b.x, 1);
        let mut r = b.clone();
        let mut delta: f64 = r.norm_2();
        let mut ar: RMatrix;
        let mut a: f64;
        let mut s1: f64;
        let mut s2: f64;
        while delta > 1.0 {
            ar = self ^ &r;
            s1 = 0.0;
            s2 = 0.0;
            for i in 0..self.x {
                s1 += r.data[i] * r.data[i];
                s2 += ar.data[i] * r.data[i];
            }
            a = s1 / s2;
            x += a * &r;
            r -= a * &ar;
            delta = r.norm_2();
        }
        x
    }

    // solve a linear system, gradient descent, normal equations
    // A ^ x = b, or min 2-norm of |A ^ x - b|
    pub fn solve_gdnr(&self, b: &RMatrix) -> RMatrix {
        assert_eq!(1, b.y, "RMatrix.solve_gdnr(&RMatrix): b must be vector");
        assert_eq!(self.x, b.x, "RMatrix.solve_gdnr(&RMatrix): matrix size mismatch");
        let mut x = RMatrix::gen_zeros(self.y, 1);
        let mut r = !(!b ^ self);
        let mut delta: f64 = r.norm_2();
        let mut ar: RMatrix;
        let mut a: f64;
        let mut s1: f64;
        let mut s2: f64;
        while delta > 10.0 {
            ar = self ^ &r;
            s1 = 0.0;
            s2 = 0.0;
            for i in 0..self.y {
                s1 += r.data[i] * r.data[i];
            }
            for i in 0..self.x {
                s2 += ar.data[i] * ar.data[i];
            }
            a = s1 / s2;
            x += a * &r;
            r -= a * !&mut (!&mut ar ^ self);
            delta = r.norm_2();
        }
        x
    }

    // solve a linear system, conjugate gradient
    // A must be symmetric, positive-define
    // A ^ x = b
    pub fn solve_cg(&self, b: &RMatrix) -> RMatrix {
        assert_eq!(self.x, self.y, "RMatrix.solve_cg(&RMatrix): square matrix only");
        assert_eq!(1, b.y, "RMatrix.solve_cg(&RMatrix): b must be vector");
        assert_eq!(self.x, b.x, "RMatrix.solve_cg(&RMatrix): matrix size mismatch");
        let mut x = RMatrix::gen_zeros(b.x, 1);
        let mut r = b.clone();
        let mut p = r.clone();
        let mut delta: f64 = r.norm_2();
        let mut ap: RMatrix;
        let mut a: f64;
        let mut b: f64;
        let mut s1: f64 = 0.0;
        let mut s2: f64;
        let mut den: f64;
        for i in 0..self.x {
            s1 += r.data[i] * r.data[i];
        }
        while delta > 0.00000000000001 {
            ap = self ^ &p;
            den = 0.0;
            for i in 0..self.x {
                den += p.data[i] * ap.data[i];
            }
            if den == 0.0 {
                break;
            }
            a = s1 / den;
            x += a * &p;
            r -= a * &ap;
            s2 = s1;
            s1 = 0.0;
            for i in 0..self.x {
                s1 += r.data[i] * r.data[i];
            }
            b = s1 / s2;
            p = &r + b * &p;
            delta = r.norm_2();
        }
        x
    }

    // solve a linear system, conjugate gradient, normal equations
    // A ^ x = b, or min 2-norm of |A ^ x - b|
    pub fn solve_cgnr(&self, b: &RMatrix) -> RMatrix {
        assert_eq!(1, b.y, "RMatrix.solve_cgnr(&RMatrix): b must be vector");
        assert_eq!(self.x, b.x, "RMatrix.solve_cgnr(&RMatrix): matrix size mismatch");
        let mut x = RMatrix::gen_zeros(self.y, 1);
        let mut r = !(!b ^ self);
        let mut p = r.clone();
        let mut delta: f64 = r.norm_2();
        let mut ap: RMatrix;
        let mut a: f64;
        let mut b: f64;
        let mut s1: f64 = 0.0;
        let mut s2: f64;
        let mut den: f64;
        for i in 0..self.y {
            s1 += r.data[i] * r.data[i];
        }
        while delta > 0.00000000000001 {
            ap = self ^ &p;
            den = 0.0;
            for i in 0..self.x {
                den += ap.data[i] * ap.data[i];
            }
            if den == 0.0 {
                break;
            }
            a = s1 / den;
            x += a * &p;
            r -= a * !&mut (!&mut ap ^ self);
            s2 = s1;
            s1 = 0.0;
            for i in 0..self.y {
                s1 += r.data[i] * r.data[i];
            }
            b = s1 / s2;
            p = &r + b * &p;
            delta = r.norm_2();
        }
        x
    }

    // solve a linear system, preconditioned conjugate gradient, Jacobi
    // A must be symmetric, positive-define
    // A ^ x = b
    pub fn solve_pcg1(&self, b: &RMatrix) -> RMatrix {
        assert_eq!(self.x, self.y, "RMatrix.solve_pcg1(&RMatrix): square matrix only");
        assert_eq!(1, b.y, "RMatrix.solve_pcg1(&RMatrix): b must be vector");
        assert_eq!(self.x, b.x, "RMatrix.solve_pcg1(&RMatrix): matrix size mismatch");
        let m = RMatrix::gen_vec(self.get_diag());
        let mut x = RMatrix::gen_zeros(b.x, 1);
        let mut r = b.clone();
        let mut z = &r / &m;
        let mut p = z.clone();
        let mut delta: f64 = r.norm_2();
        let mut ap: RMatrix;
        let mut a: f64;
        let mut b: f64;
        let mut s1: f64 = 0.0;
        let mut s2: f64;
        let mut den: f64;
        for i in 0..self.x {
            s1 += r.data[i] * z.data[i];
        }
        while delta > 0.00000000000001 {
            ap = self ^ &p;
            den = 0.0;
            for i in 0..self.x {
                den += p.data[i] * ap.data[i];
            }
            a = s1 / den;
            if den == 0.0 {
                break;
            }
            x += a * &p;
            r -= a * &ap;
            z = &r / &m;
            s2 = s1;
            s1 = 0.0;
            for i in 0..self.x {
                s1 += r.data[i] * z.data[i];
            }
            b = s1 / s2;
            p = &z + b * &p;
            delta = r.norm_2();
        }
        x
    }

    // solve a linear system, preconditioned conjugate gradient, A3
    // A must be symmetric, positive-define
    // A ^ x = b
    pub fn solve_pcg3(&self, b: &RMatrix) -> RMatrix {
        assert_eq!(self.x, self.y, "RMatrix.solve_pcg3(&RMatrix): square matrix only");
        assert_eq!(1, b.y, "RMatrix.solve_pcg3(&RMatrix): b must be vector");
        assert_eq!(self.x, b.x, "RMatrix.solve_pcg3(&RMatrix): matrix size mismatch");
        let mut m = self.clone();
        for i in 0..self.x {
            for j in (i + 2)..self.y {
                m.data[i * self.y + j] = 0.0;
                m.data[j * self.y + i] = 0.0;
            }
        }
        let mut x = RMatrix::gen_zeros(b.x, 1);
        let mut r = b.clone();
        let mut z = m.solve_lu(&r);
        let mut p = z.clone();
        let mut delta: f64 = r.norm_2();
        let mut ap: RMatrix;
        let mut a: f64;
        let mut b: f64;
        let mut s1: f64 = 0.0;
        let mut s2: f64;
        let mut den: f64;
        for i in 0..self.x {
            s1 += r.data[i] * z.data[i];
        }
        while delta > 0.00000000000001 {
            ap = self ^ &p;
            den = 0.0;
            for i in 0..self.x {
                den += p.data[i] * ap.data[i];
            }
            a = s1 / den;
            if den == 0.0 {
                break;
            }
            x += a * &p;
            r -= a * &ap;
            z = m.solve_lu(&r);
            s2 = s1;
            s1 = 0.0;
            for i in 0..self.x {
                s1 += r.data[i] * z.data[i];
            }
            b = s1 / s2;
            p = &z + b * &p;
            delta = r.norm_2();
        }
        x
    }

    // solve a linear system, preconditioned conjugate gradient, A5
    // A must be symmetric, positive-define
    // A ^ x = b
    pub fn solve_pcg5(&self, b: &RMatrix) -> RMatrix {
        assert_eq!(self.x, self.y, "RMatrix.solve_pcg5(&RMatrix): square matrix only");
        assert_eq!(1, b.y, "RMatrix.solve_pcg5(&RMatrix): b must be vector");
        assert_eq!(self.x, b.x, "RMatrix.solve_pcg5(&RMatrix): matrix size mismatch");
        let mut m = self.clone();
        for i in 0..self.x {
            for j in (i + 3)..self.y {
                m.data[i * self.y + j] = 0.0;
                m.data[j * self.y + i] = 0.0;
            }
        }
        let mut x = RMatrix::gen_zeros(b.x, 1);
        let mut r = b.clone();
        let mut z = m.solve_lu(&r);
        let mut p = z.clone();
        let mut delta: f64 = r.norm_2();
        let mut ap: RMatrix;
        let mut a: f64;
        let mut b: f64;
        let mut s1: f64 = 0.0;
        let mut s2: f64;
        let mut den: f64;
        for i in 0..self.x {
            s1 += r.data[i] * z.data[i];
        }
        while delta > 0.00000000000001 {
            ap = self ^ &p;
            den = 0.0;
            for i in 0..self.x {
                den += p.data[i] * ap.data[i];
            }
            a = s1 / den;
            if den == 0.0 {
                break;
            }
            x += a * &p;
            r -= a * &ap;
            z = m.solve_lu(&r);
            s2 = s1;
            s1 = 0.0;
            for i in 0..self.x {
                s1 += r.data[i] * z.data[i];
            }
            b = s1 / s2;
            p = &z + b * &p;
            delta = r.norm_2();
        }
        x
    }

    // solve a linear system, preconditioned conjugate gradient, Ab
    // A must be symmetric, positive-define
    // A ^ x = b
    pub fn solve_pcgb(&self, b: &RMatrix) -> RMatrix {
        assert_eq!(self.x, self.y, "RMatrix.solve_pcgb(&RMatrix): square matrix only");
        assert_eq!(1, b.y, "RMatrix.solve_pcgb(&RMatrix): b must be vector");
        assert_eq!(self.x, b.x, "RMatrix.solve_pcgb(&RMatrix): matrix size mismatch");
        let mut m = self.clone();
        let mut i: usize = 0;
        while i < self.x {
            let mut rand: usize = rand::thread_rng().gen_range(3, 5);
            if (i + rand + 1) > self.x {
                rand = self.x - i;
            }
            for k in i..(i + rand) {
                for j in (i + rand)..self.y {
                    m.data[k * self.y + j] = 0.0;
                    m.data[j * self.y + k] = 0.0;
                }
            }
            i += rand;
        }
        let mut x = RMatrix::gen_zeros(b.x, 1);
        let mut r = b.clone();
        let mut z = m.solve_lu(&r);
        let mut p = z.clone();
        let mut delta: f64 = r.norm_2();
        let mut ap: RMatrix;
        let mut a: f64;
        let mut b: f64;
        let mut s1: f64 = 0.0;
        let mut s2: f64;
        let mut den: f64;
        for i in 0..self.x {
            s1 += r.data[i] * z.data[i];
        }
        while delta > 0.00000000000001 {
            ap = self ^ &p;
            den = 0.0;
            for i in 0..self.x {
                den += p.data[i] * ap.data[i];
            }
            a = s1 / den;
            if den == 0.0 {
                break;
            }
            x += a * &p;
            r -= a * &ap;
            z = m.solve_lu(&r);
            s2 = s1;
            s1 = 0.0;
            for i in 0..self.x {
                s1 += r.data[i] * z.data[i];
            }
            b = s1 / s2;
            p = &z + b * &p;
            delta = r.norm_2();
        }
        x
    }

    // solve a linear system, Lanczos
    // A must be symmetric
    // A ^ x = b
    pub fn solve_lanczos(&self, b: &RMatrix) -> RMatrix {
        assert_eq!(self.x, self.y, "RMatrix.solve_lanczos(&RMatrix): square matrix only");
        assert_eq!(1, b.y, "RMatrix.solve_lanczos(&RMatrix): b must be vector");
        assert_eq!(self.x, b.x, "RMatrix.solve_lanczos(&RMatrix): matrix size mismatch");
        let mut q_data: Vec<f64> = vec![0.0; self.x * self.x];
        let mut ll_data: Vec<f64> = vec![0.0; self.x];
        let mut uu_data: Vec<f64> = vec![0.0; self.x];
        let mut ud_data: Vec<f64> = vec![0.0; self.x + 1];
        let mut y_data: Vec<f64> = vec![0.0; self.x + 1];
        let mut q = RMatrix::gen_zeros(self.x, 1);
        let mut r = b.clone();
        let mut a: f64;
        let mut b: f64 = r.norm_2();
        let mut eps: f64;
        y_data[0] = b;
        let mut bq: RMatrix;
        for n in 0..self.y {
            bq = b * &q;
            q = &r / b;
            r = (self ^ &q) - &bq;
            a = 0.0;
            for i in 0..self.x {
                a += q.data[i] * r.data[i];
            }
            for i in 0..self.x {
                r.data[i] -= a * q.data[i];
            }
            b = r.norm_2();
            assert_ne!(b, 0.0, "RMatrix.solve_lanczos(): break!");
            // for Q
            for i in 0..self.x {
                q_data[n * self.x + i] = q.data[i];
            }
            // submat LU decomposition
            ud_data[n] += a;
            ll_data[n] = b / ud_data[n];
            uu_data[n] = b;
            ud_data[n + 1] = -b * ll_data[n];
            // solve submat(LU, l only)
            y_data[n + 1] -= y_data[n] * ll_data[n];
            eps = (b * y_data[n] / ud_data[n]).abs();
            if eps < 0.00000000000001 {
                // solve submat(LU, u)
                for i in (1..n).rev() {
                    y_data[i] /= ud_data[i];
                    y_data[i - 1] -= y_data[i] * uu_data[i - 1];
                }
                y_data[0] /= ud_data[0];
                // clean
                y_data[n + 1] = 0.0;
                break;
            }
        }
        let y = RMatrix{
            x: 1,
            y: self.x,
            data: y_data
        };
        let q = RMatrix {
            x: self.x,
            y: self.x,
            data: q_data
        };
        !(y ^ q)
    }

    // solve a linear system, minimal residual
    // A must be symmetric
    // A ^ x = b
    pub fn solve_minres(&self, b: &RMatrix) -> RMatrix {
        assert_eq!(self.x, self.y, "RMatrix.solve_minres(&RMatrix): square matrix only");
        assert_eq!(1, b.y, "RMatrix.solve_minres(&RMatrix): b must be vector");
        assert_eq!(self.x, b.x, "RMatrix.solve_minres(&RMatrix): matrix size mismatch");
        let mut q_data: Vec<f64> = vec![0.0; (self.x + 1) * self.x];
        let mut q1_data: Vec<f64> = vec![0.0; self.x + 1];
        let mut rd_data: Vec<f64> = vec![0.0; self.x];
        let mut r1_data: Vec<f64> = vec![0.0; self.x];
        let mut r2_data: Vec<f64> = vec![0.0; self.x];
        let mut y_data: Vec<f64> = vec![0.0; self.x + 1];
        let mut q = RMatrix::gen_zeros(self.x, 1);
        let mut r = b.clone();
        let mut a: f64;
        let mut b: f64 = r.norm_2();
        let mut eps: f64;
        let mut x: f64;
        let mut y: f64;
        let mut c: f64 = 0.0;
        let mut s: f64 = 0.0;
        y_data[0] = b;
        q1_data[0] = 1.0;
        let mut bq = b * &q;
        q = &r / b;
        for i in 0..self.x {
            q_data[i] = q.data[i];
        }
        for n in 0..self.y {
            r = (self ^ &q) - &bq;
            a = 0.0;
            for i in 0..self.x {
                a += q.data[i] * r.data[i];
            }
            for i in 0..self.x {
                r.data[i] -= a * q.data[i];
            }
            b = r.norm_2();
            assert_ne!(b, 0.0, "RMatrix.solve_minres(): break!");
            bq = b * &q;
            q = &r / b;
            // for Q
            for i in 0..self.x {
                q_data[(n + 1) * self.x + i] = q.data[i];
            }
            // submat QR decomposition, Givens
            // R
            if n == 0 {
                rd_data[n] = a;
                r1_data[n] = b;
            } else {
                x = r1_data[n - 1];
                // y = a
                r1_data[n - 1] =  x * c + a * s;
                rd_data[n] = -x * s + a * c;
                // x = 0.0
                // y = b
                r2_data[n - 1] = b * s;
                r1_data[n] = b * c;
            }
            x = rd_data[n];
            y = b;
            c = x / (x * x + y * y).sqrt();
            s = y / (x * x + y * y).sqrt();
            rd_data[n] =  x * c + y * s;
            // Q
            x = q1_data[n];
            // y = 0.0
            q1_data[n] =  x * c;
            q1_data[n + 1] = -x * s;
            // solve submat(LU, l only)
            eps = (q1_data[n + 1]).abs();
            if eps < 0.00000000000001 {
                // solve submat(QR, r)
                for i in (0..n).rev() {
                    y_data[i] = q1_data[i] * y_data[0];
                }
                for i in (2..n).rev() {
                    y_data[i] /= rd_data[i];
                    y_data[i - 1] -= y_data[i] * r1_data[i - 1];
                    y_data[i - 2] -= y_data[i] * r2_data[i - 2];
                }
                y_data[1] /= rd_data[1];
                y_data[0] -= y_data[1] * r1_data[0];
                y_data[0] /= rd_data[0];
                break;
            }
        }
        let y = RMatrix{
            x: 1,
            y: self.x + 1,
            data: y_data
        };
        let q = RMatrix {
            x: self.x + 1,
            y: self.x,
            data: q_data
        };
        !(y ^ q)
    }
}

// Clone
impl Clone for RMatrix {
    fn clone(&self) -> RMatrix {
        let ret_data: Vec<f64> = self.data.clone();
        RMatrix {
            x: self.x,
            y: self.y,
            data: ret_data
        }
    }

    fn clone_from(&mut self, mat: &RMatrix) {
        let ret_data: Vec<f64> = mat.data.clone();
        self.x = mat.x;
        self.y = mat.y;
        self.data = ret_data;
    }
}

impl_fn_i_0!{chol from ichol for RMatrix}
impl_fn_i_0!{chol_tri from ichol_tri for RMatrix}

impl_fn_i_1!{lu from ilu for RMatrix}
impl_fn_i_1!{lu_tri from ilu_tri for RMatrix}

impl_fn_i_1!{qr_hh from iqr_hh for RMatrix}
impl_fn_i_0!{r_hh from ir_hh for RMatrix}
impl_fn_i_1!{qhq_hh from iqhq_hh for RMatrix}
impl_fn_i_0!{h_hh from ih_hh for RMatrix}
impl_fn_i_1!{qtq_hh from iqtq_hh for RMatrix}
impl_fn_i_0!{t_hh from it_hh for RMatrix}

impl_fn_i_1!{qr_givens from iqr_givens for RMatrix}
impl_fn_i_0!{r_givens from ir_givens for RMatrix}
impl_fn_i_1!{qhq_givens from iqhq_givens for RMatrix}
impl_fn_i_0!{h_givens from ih_givens for RMatrix}
impl_fn_i_1!{qtq_givens from iqtq_givens for RMatrix}
impl_fn_i_0!{t_givens from it_givens for RMatrix}

impl_fn_i_2!{pbq_hh from ipbq_hh for RMatrix}
impl_fn_i_0!{b_hh from ib_hh for RMatrix}

impl_fn_i_2!{pbq_givens from ipbq_givens for RMatrix}
impl_fn_i_0!{b_givens from ib_givens for RMatrix}

impl_fn_i_2!{svd_qr from isvd_qr for RMatrix}
impl_fn_i_0!{sv_qr from isv_qr for RMatrix}

// operators
// helper macro
// op &A -> op A
macro_rules! impl_op_1_i {
    (impl $imp:ident, $method:ident for $it:ty, $ot:ty) => {
        impl $imp for $it {
            type Output = $ot;

            #[inline]
            fn $method(self) -> $ot {
                $imp::$method(&self)
            }
        }
    }
}

// op &A -> op &mut A
macro_rules! impl_op_1_m {
    (impl $imp:ident, $method:ident for $it:ty, $ot:ty) => {
        impl_op_1_i!(impl $imp, $method for $it, $ot);

        impl<'a> $imp for &'a mut $it {
            type Output = $ot;

            #[inline]
            fn $method(self) -> $ot {
                $imp::$method(&*self)
            }
        }
    }
}

// &A op &B -> &A op B, A op &B, A op B
macro_rules! impl_op_2_ii {
    (impl $imp:ident, $method:ident for $lt:ty, $rt:ty, $ot:ty) => {
        impl<'a> $imp<$rt> for &'a $lt {
            type Output = $ot;

            #[inline]
            fn $method(self, rhs: $rt) -> $ot {
                $imp::$method(self, &rhs)
            }
        }

        impl<'b> $imp<&'b $rt> for $lt {
            type Output = $ot;

            #[inline]
            fn $method(self, rhs: &'b $rt) -> $ot {
                $imp::$method(&self, rhs)
            }
        }

        impl $imp<$rt> for $lt {
            type Output = $ot;

            #[inline]
            fn $method(self, rhs: $rt) -> $ot {
                $imp::$method(&self, &rhs)
            }
        }
    }
}

// &A op &B -> &mut A op B, &mut A op &B
macro_rules! impl_op_2_mi {
    (impl $imp:ident, $method:ident for $lt:ty, $rt:ty, $ot:ty) => {
        impl_op_2_ii!(impl $imp, $method for $lt, $rt, $ot);

        impl_op_2_mi!(impl $imp, $method for mut $lt, $rt, $ot);
    };
    (impl $imp:ident, $method:ident for mut $lt:ty, $rt:ty, $ot:ty) => {
        impl<'a> $imp<$rt> for &'a mut $lt {
            type Output = $ot;

            #[inline]
            fn $method(self, rhs: $rt) -> $ot {
                $imp::$method(&*self, &rhs)
            }
        }

        impl<'a, 'b> $imp<&'b $rt> for &'a mut $lt {
            type Output = $ot;

            #[inline]
            fn $method(self, rhs: &'b $rt) -> $ot {
                $imp::$method(&*self, rhs)
            }
        }
    }
}

// &A op &B -> A op &mut B, &A op &mut B
macro_rules! impl_op_2_im {
    (impl $imp:ident, $method:ident for $lt:ty, $rt:ty, $ot:ty) => {
        impl_op_2_ii!(impl $imp, $method for $lt, $rt, $ot);

        impl_op_2_im!(impl $imp, $method for $lt, mut $rt, $ot);
    };
    (impl $imp:ident, $method:ident for $lt:ty, mut $rt:ty, $ot:ty) => {
        impl<'b> $imp<&'b mut $rt> for $lt {
            type Output = $ot;

            #[inline]
            fn $method(self, rhs: &'b mut $rt) -> $ot {
                $imp::$method(&self, &*rhs)
            }
        }

        impl<'a, 'b> $imp<&'b mut $rt> for &'a $lt {
            type Output = $ot;

            #[inline]
            fn $method(self, rhs: &'b mut $rt) -> $ot {
                $imp::$method(self, &*rhs)
            }
        }
    }
}

// &A op &B -> &mut A op &mut B
macro_rules! impl_op_2_mm {
    (impl $imp:ident, $method:ident for $lt:ty, $rt:ty, $ot:ty) => {
        impl_op_2_ii!(impl $imp, $method for $lt, $rt, $ot);

        impl_op_2_mi!(impl $imp, $method for mut $lt, $rt, $ot);

        impl_op_2_im!(impl $imp, $method for $lt, mut $rt, $ot);

        impl_op_2_mm!(impl $imp, $method for mut $lt, mut $rt, $ot);
    };
    (impl $imp:ident, $method:ident for mut $lt:ty, mut $rt:ty, $ot:ty) => {
        impl<'a, 'b> $imp<&'b mut $rt> for &'a mut $lt {
            type Output = $ot;

            #[inline]
            fn $method(self, rhs: &'b mut $rt) -> $ot {
                $imp::$method(&*self, &*rhs)
            }
        }
    }
}

// addition
// A + B
// out of place
impl<'a, 'b> Add<&'b RMatrix> for &'a RMatrix {
    type Output = RMatrix;

    fn add(self, mat: &'b RMatrix) -> RMatrix {
        if mat.x == 1 && mat.y == 1 {
            return self + mat.data[0]
        }
        if self.x == 1 && self.y == 1 {
            return self.data[0] + mat
        }
        assert_eq!(self.x, mat.x, "RMatrix.add(RMatrix): matrix size mismatch");
        assert_eq!(self.y, mat.y, "RMatrix.add(RMatrix): matrix size mismatch");
        let mut ret_data: Vec<f64> = vec![0.0; self.x * self.y];
        for i in 0..(self.x * self.y) {
            ret_data[i] = self.data[i] + mat.data[i];
        }
        RMatrix {
            x: self.x,
            y: self.y,
            data: ret_data
        }
    }
}

impl_op_2_mm!{impl Add, add for RMatrix, RMatrix, RMatrix}

// A + b
// out of place
impl<'a, 'b> Add<&'b f64> for &'a RMatrix {
    type Output = RMatrix;

    fn add(self, num: &'b f64) -> RMatrix {
        let mut ret_data: Vec<f64> = vec![0.0; self.x * self.y];
        for i in 0..(self.x * self.y) {
            ret_data[i] = self.data[i] + num;
        }
        RMatrix {
            x: self.x,
            y: self.y,
            data: ret_data
        }
    }
}

impl_op_2_mi!{impl Add, add for RMatrix, f64, RMatrix}

// a + B
// out of place
impl<'a, 'b> Add<&'b RMatrix> for &'a f64 {
    type Output = RMatrix;

    fn add(self, mat: &'b RMatrix) -> RMatrix {
        let mut ret_data: Vec<f64> = vec![0.0; mat.x * mat.y];
        for i in 0..(mat.x * mat.y) {
            ret_data[i] = self + mat.data[i];
        }
        RMatrix {
            x: mat.x,
            y: mat.y,
            data: ret_data
        }
    }
}

impl_op_2_im!{impl Add, add for f64, RMatrix, RMatrix}

// A += &B
// in place
impl<'b> AddAssign<&'b RMatrix> for RMatrix {
    fn add_assign(&mut self, mat: &RMatrix) {
        if mat.x == 1 && mat.y == 1 {
            *self += mat.data[0];
            return;
        }
        assert_eq!(self.x, mat.x, "RMatrix.add_assign(&RMatrix): matrix size mismatch");
        assert_eq!(self.y, mat.y, "RMatrix.add_assign(&RMatrix): matrix size mismatch");
        for i in 0..(self.x * self.y) {
            self.data[i] += mat.data[i];
        }
    }
}

// A += B
// in place
impl AddAssign<RMatrix> for RMatrix {
    fn add_assign(&mut self, mat: RMatrix) {
        if mat.x == 1 && mat.y == 1 {
            *self += mat.data[0];
            return;
        }
        assert_eq!(self.x, mat.x, "RMatrix.add_assign(RMatrix): matrix size mismatch");
        assert_eq!(self.y, mat.y, "RMatrix.add_assign(RMatrix): matrix size mismatch");
        for i in 0..(self.x * self.y) {
            self.data[i] += mat.data[i];
        }
    }
}

// A += b
// in place
impl AddAssign<f64> for RMatrix {
    fn add_assign(&mut self, num: f64) {
        for i in 0..(self.x * self.y) {
            self.data[i] += num;
        }
    }
}

// subtraction
// A - B
// out of place
impl<'a, 'b> Sub<&'b RMatrix> for &'a RMatrix {
    type Output = RMatrix;

    fn sub(self, mat: &RMatrix) -> RMatrix {
        if mat.x == 1 && mat.y == 1 {
            return self - mat.data[0]
        }
        if self.x == 1 && self.y == 1 {
            return self.data[0] - mat
        }
        assert_eq!(self.x, mat.x, "RMatrix.sub(RMatrix): matrix size mismatch");
        assert_eq!(self.y, mat.y, "RMatrix.sub(RMatrix): matrix size mismatch");
        let mut ret_data: Vec<f64> = vec![0.0; self.x * self.y];
        for i in 0..(self.x * self.y) {
            ret_data[i] = self.data[i] - mat.data[i];
        }
        RMatrix {
            x: self.x,
            y: self.y,
            data: ret_data
        }
    }
}

impl_op_2_mm!{impl Sub, sub for RMatrix, RMatrix, RMatrix}

// A - b
// out of place
impl<'a, 'b> Sub<&'b f64> for &'a RMatrix {
    type Output = RMatrix;

    fn sub(self, num: &'b f64) -> RMatrix {
        let mut ret_data: Vec<f64> = vec![0.0; self.x * self.y];
        for i in 0..(self.x * self.y) {
            ret_data[i] = self.data[i] - num;
        }
        RMatrix {
            x: self.x,
            y: self.y,
            data: ret_data
        }
    }
}

impl_op_2_mi!{impl Sub, sub for RMatrix, f64, RMatrix}

// a - B
// out of place
impl<'a, 'b> Sub<&'b RMatrix> for &'a f64 {
    type Output = RMatrix;

    fn sub(self, mat: &'b RMatrix) -> RMatrix {
        let mut ret_data: Vec<f64> = vec![0.0; mat.x * mat.y];
        for i in 0..(mat.x * mat.y) {
            ret_data[i] = self - mat.data[i];
        }
        RMatrix {
            x: mat.x,
            y: mat.y,
            data: ret_data
        }
    }
}

impl_op_2_im!{impl Sub, sub for f64, RMatrix, RMatrix}

// A -= &B
// in place
impl<'b> SubAssign<&'b RMatrix> for RMatrix {
    fn sub_assign(&mut self, mat: &RMatrix) {
        if mat.x == 1 && mat.y == 1 {
            *self -= mat.data[0];
            return;
        }
        assert_eq!(self.x, mat.x, "RMatrix.sub_assign(&RMatrix): matrix size mismatch");
        assert_eq!(self.y, mat.y, "RMatrix.sub_assign(&RMatrix): matrix size mismatch");
        for i in 0..(self.x * self.y) {
            self.data[i] -= mat.data[i];
        }
    }
}

// A -= B
// in place
impl SubAssign<RMatrix> for RMatrix {
    fn sub_assign(&mut self, mat: RMatrix) {
        if mat.x == 1 && mat.y == 1 {
            *self -= mat.data[0];
            return;
        }
        assert_eq!(self.x, mat.x, "RMatrix.sub_assign(RMatrix): matrix size mismatch");
        assert_eq!(self.y, mat.y, "RMatrix.sub_assign(RMatrix): matrix size mismatch");
        for i in 0..(self.x * self.y) {
            self.data[i] -= mat.data[i];
        }
    }
}

// A -= b
// in place
impl SubAssign<f64> for RMatrix {
    fn sub_assign(&mut self, num: f64) {
        for i in 0..(self.x * self.y) {
            self.data[i] -= num;
        }
    }
}

// multiplication
// A * B
// out of place
impl<'a, 'b> Mul<&'b RMatrix> for &'a RMatrix {
    type Output = RMatrix;

    fn mul(self, mat: &RMatrix) -> RMatrix {
        if mat.x == 1 && mat.y == 1 {
            return self * mat.data[0]
        }
        if self.x == 1 && self.y == 1 {
            return self.data[0] * mat
        }
        assert_eq!(self.x, mat.x, "RMatrix.mul(RMatrix): matrix size mismatch");
        assert_eq!(self.y, mat.y, "RMatrix.mul(RMatrix): matrix size mismatch");
        let mut ret_data: Vec<f64> = vec![0.0; self.x * self.y];
        for i in 0..(self.x * self.y) {
            ret_data[i] = self.data[i] * mat.data[i];
        }
        RMatrix {
            x: self.x,
            y: self.y,
            data: ret_data
        }
    }
}

impl_op_2_mm!{impl Mul, mul for RMatrix, RMatrix, RMatrix}

// A * b
// out of place
impl<'a, 'b> Mul<&'b f64> for &'a RMatrix {
    type Output = RMatrix;

    fn mul(self, num: &'b f64) -> RMatrix {
        let mut ret_data: Vec<f64> = vec![0.0; self.x * self.y];
        for i in 0..(self.x * self.y) {
            ret_data[i] = self.data[i] * num;
        }
        RMatrix {
            x: self.x,
            y: self.y,
            data: ret_data
        }
    }
}

impl_op_2_mi!{impl Mul, mul for RMatrix, f64, RMatrix}

// a * B
// out of place
impl<'a, 'b> Mul<&'b RMatrix> for &'a f64 {
    type Output = RMatrix;

    fn mul(self, mat: &'b RMatrix) -> RMatrix {
        let mut ret_data: Vec<f64> = vec![0.0; mat.x * mat.y];
        for i in 0..(mat.x * mat.y) {
            ret_data[i] = self * mat.data[i];
        }
        RMatrix {
            x: mat.x,
            y: mat.y,
            data: ret_data
        }
    }
}

impl_op_2_im!{impl Mul, mul for f64, RMatrix, RMatrix}

// A *= &B
// in place
impl<'b> MulAssign<&'b RMatrix> for RMatrix {
    fn mul_assign(&mut self, mat: &RMatrix) {
        if mat.x == 1 && mat.y == 1 {
            *self *= mat.data[0];
            return;
        }
        assert_eq!(self.x, mat.x, "RMatrix.mul_assign(&RMatrix): matrix size mismatch");
        assert_eq!(self.y, mat.y, "RMatrix.mul_assign(&RMatrix): matrix size mismatch");
        for i in 0..(self.x * self.y) {
            self.data[i] *= mat.data[i];
        }
    }
}

// A *= B
// in place
impl MulAssign<RMatrix> for RMatrix {
    fn mul_assign(&mut self, mat: RMatrix) {
        if mat.x == 1 && mat.y == 1 {
            *self *= mat.data[0];
            return;
        }
        assert_eq!(self.x, mat.x, "RMatrix.mul_assign(RMatrix): matrix size mismatch");
        assert_eq!(self.y, mat.y, "RMatrix.mul_assign(RMatrix): matrix size mismatch");
        for i in 0..(self.x * self.y) {
            self.data[i] *= mat.data[i];
        }
    }
}

// A *= b
// in place
impl MulAssign<f64> for RMatrix {
    fn mul_assign(&mut self, num: f64) {
        for i in 0..(self.x * self.y) {
            self.data[i] *= num;
        }
    }
}

// division
// A / B
// out of place
impl<'a, 'b> Div<&'b RMatrix> for &'a RMatrix {
    type Output = RMatrix;

    fn div(self, mat: &RMatrix) -> RMatrix {
        if mat.x == 1 && mat.y == 1 {
            return self / mat.data[0]
        }
        if self.x == 1 && self.y == 1 {
            return self.data[0] / mat
        }
        assert_eq!(self.x, mat.x, "RMatrix.div(RMatrix): matrix size mismatch");
        assert_eq!(self.y, mat.y, "RMatrix.div(RMatrix): matrix size mismatch");
        let mut ret_data: Vec<f64> = vec![0.0; self.x * self.y];
        for i in 0..(self.x * self.y) {
            ret_data[i] = self.data[i] / mat.data[i];
        }
        RMatrix {
            x: self.x,
            y: self.y,
            data: ret_data
        }
    }
}

impl_op_2_mm!{impl Div, div for RMatrix, RMatrix, RMatrix}

// A / b
// out of place
impl<'a, 'b> Div<&'b f64> for &'a RMatrix {
    type Output = RMatrix;

    fn div(self, num: &'b f64) -> RMatrix {
        let mut ret_data: Vec<f64> = vec![0.0; self.x * self.y];
        for i in 0..(self.x * self.y) {
            ret_data[i] = self.data[i] / num;
        }
        RMatrix {
            x: self.x,
            y: self.y,
            data: ret_data
        }
    }
}

impl_op_2_mi!{impl Div, div for RMatrix, f64, RMatrix}

// a / B
// out of place
impl<'a, 'b> Div<&'b RMatrix> for &'a f64 {
    type Output = RMatrix;

    fn div(self, mat: &'b RMatrix) -> RMatrix {
        let mut ret_data: Vec<f64> = vec![0.0; mat.x * mat.y];
        for i in 0..(mat.x * mat.y) {
            ret_data[i] = self / mat.data[i];
        }
        RMatrix {
            x: mat.x,
            y: mat.y,
            data: ret_data
        }
    }
}

impl_op_2_im!{impl Div, div for f64, RMatrix, RMatrix}

// A /= &B
// in place
impl<'b> DivAssign<&'b RMatrix> for RMatrix {
    fn div_assign(&mut self, mat: &RMatrix) {
        if mat.x == 1 && mat.y == 1 {
            *self /= mat.data[0];
            return;
        }
        assert_eq!(self.x, mat.x, "RMatrix.div_assign(&RMatrix): matrix size mismatch");
        assert_eq!(self.y, mat.y, "RMatrix.div_assign(&RMatrix): matrix size mismatch");
        for i in 0..(self.x * self.y) {
            self.data[i] /= mat.data[i];
        }
    }
}

// A /= B
// in place
impl DivAssign<RMatrix> for RMatrix {
    fn div_assign(&mut self, mat: RMatrix) {
        if mat.x == 1 && mat.y == 1 {
            *self /= mat.data[0];
            return;
        }
        assert_eq!(self.x, mat.x, "RMatrix.div_assign(RMatrix): matrix size mismatch");
        assert_eq!(self.y, mat.y, "RMatrix.div_assign(RMatrix): matrix size mismatch");
        for i in 0..(self.x * self.y) {
            self.data[i] /= mat.data[i];
        }
    }
}

// A /= b
// in place
impl DivAssign<f64> for RMatrix {
    fn div_assign(&mut self, num: f64) {
        for i in 0..(self.x * self.y) {
            self.data[i] /= num;
        }
    }
}

// matrix multiplication
// A ^ B
// out of place matrix multiplication
impl<'a, 'b> BitXor<&'b RMatrix> for &'a RMatrix {
    type Output = RMatrix;

    fn bitxor(self, mat: &RMatrix) -> RMatrix {
        if mat.x == 1 && mat.y == 1 {
            return self * mat.data[0]
        }
        if self.x == 1 && self.y == 1 {
            return self.data[0] * mat
        }
        assert_eq!(self.y, mat.x, "RMatrix.bitxor(RMatrix): matrix size mismatch");
        let mut ret_data: Vec<f64> = vec![0.0; self.x * mat.y];
        for i in 0..self.x {
            for k in 0..self.y {
                for j in 0..mat.y {
                    ret_data[i * mat.y + j] += self.data[i * self.y + k] * mat.data[k * mat.y + j];
                }
            }
        }
        RMatrix {
            x: self.x,
            y: mat.y,
            data: ret_data
        }
    }
}

impl_op_2_mm!{impl BitXor, bitxor for RMatrix, RMatrix, RMatrix}

// in place left matrix multiplication
// A <<= &B
// number or square matrix only
// fast, 0.6x time of BitXor
impl<'b> ShlAssign<&'b RMatrix> for RMatrix {
    fn shl_assign(&mut self, mat: &RMatrix) {
        if mat.x == 1 && mat.y == 1 {
            *self *= mat.data[0];
            return;
        }
        assert_eq!(self.x, mat.x, "RMatrix.shl_assign(&RMatrix): matrix size mismatch");
        assert_eq!(self.x, mat.y, "RMatrix.shl_assign(&RMatrix): matrix size mismatch");
        let mut tmp_s: Vec<f64>;
        for j in 0..self.y {
            tmp_s = self.get_col(j);
            for i in 0..mat.x {
                let tmp_m = mat.row(i);
                self.data[i * self.y + j] = 0.0;
                for k in 0..self.x {
                    self.data[i * self.y + j] += tmp_s[k] * tmp_m[k];
                }
            }
        }
    }
}

// A <<= B
// number or square matrix only
// fast, 0.6x time of BitXor
impl ShlAssign<RMatrix> for RMatrix {
    fn shl_assign(&mut self, mat: RMatrix) {
        if mat.x == 1 && mat.y == 1 {
            *self *= mat.data[0];
            return;
        }
        assert_eq!(self.x, mat.x, "RMatrix.shl_assign(RMatrix): matrix size mismatch");
        assert_eq!(self.x, mat.y, "RMatrix.shl_assign(RMatrix): matrix size mismatch");
        let mut tmp_s: Vec<f64>;
        for j in 0..self.y {
            tmp_s = self.get_col(j);
            for i in 0..mat.x {
                let tmp_m = mat.row(i);
                self.data[i * self.y + j] = 0.0;
                for k in 0..self.x {
                    self.data[i * self.y + j] += tmp_s[k] * tmp_m[k];
                }
            }
        }
    }
}

// in place right matrix multiplication
// A >>= &B
// number or square matrix only
// slow, 2.5x time of Shl or 1.5x time of BitXor
impl<'b> ShrAssign<&'b RMatrix> for RMatrix {
    fn shr_assign(&mut self, mat: &RMatrix) {
        if mat.x == 1 && mat.y == 1 {
            *self *= mat.data[0];
            return;
        }
        assert_eq!(self.y, mat.x, "RMatrix.shr_assign(&RMatrix): matrix size mismatch");
        assert_eq!(self.y, mat.y, "RMatrix.shr_assign(&RMatrix): matrix size mismatch");
        let mut tmp_s: Vec<f64>;
        let mut tmp_m: Vec<f64>;
        for i in 0..self.x {
            tmp_s = self.get_row(i);
            for j in 0..mat.y {
                // lose speed here
                tmp_m = mat.get_col(j);
                self.data[i * mat.y + j] = 0.0;
                for k in 0..self.y {
                    self.data[i * mat.y + j] += tmp_s[k] * tmp_m[k];
                }
            }
        }
    }
}

// A >>= B
// number or square matrix only
// slow, 2.5x time of Shl or 1.5x time of BitXor
impl ShrAssign<RMatrix> for RMatrix {
    fn shr_assign(&mut self, mat: RMatrix) {
        if mat.x == 1 && mat.y == 1 {
            *self *= mat.data[0];
            return;
        }
        assert_eq!(self.y, mat.x, "RMatrix.shr_assign(RMatrix): matrix size mismatch");
        assert_eq!(self.y, mat.y, "RMatrix.shr_assign(RMatrix): matrix size mismatch");
        let mut tmp_s: Vec<f64>;
        let mut tmp_m: Vec<f64>;
        for i in 0..self.x {
            tmp_s = self.get_row(i);
            for j in 0..mat.y {
                // lose speed here
                tmp_m = mat.get_col(j);
                self.data[i * mat.y + j] = 0.0;
                for k in 0..self.y {
                    self.data[i * mat.y + j] += tmp_s[k] * tmp_m[k];
                }
            }
        }
    }
}

// transpose
// !&A
// out of place
impl<'a> Not for &'a RMatrix {
    type Output = RMatrix;

    fn not(self) -> RMatrix {
        let mut ret_data: Vec<f64> = vec![0.0; self.x * self.y];
        if self.x == 1 || self.y == 1 {
            ret_data.copy_from_slice(self.data.as_slice());
        } else {
            for i in 0..self.x {
                for j in 0..self.y {
                    ret_data[j * self.x + i] = self.data[i * self.y + j];
                }
            }
        }
        RMatrix {
            x: self.y,
            y: self.x,
            data: ret_data
        }
    }
}

impl_op_1_i!{impl Not, not for RMatrix, RMatrix}

// !&mut A
// (partly) in place
// vector or square matrix only
impl<'a> Not for &'a mut RMatrix {
    type Output = &'a RMatrix;

    fn not(self) -> &'a RMatrix {
        let x = self.x;
        let y = self.y;
        self.x = y;
        self.y = x;
        if x == 1 || y == 1 {
            return self;
        }
        if x == y {
            let mut tmp: f64;
            for i in 0..x {
                for j in 0..i {
                    tmp = self.data[j * x + i];
                    self.data[j * x + i] = self.data[i * y + j];
                    self.data[i * y + j] = tmp;
                }
            }
        } else {
            // out of place
            let mut ret_data: Vec<f64> = vec![0.0; x * y];
            for i in 0..x {
                for j in 0..y {
                    ret_data[j * x + i] = self.data[i * y + j];
                }
            }
            self.data = ret_data;
        }
        self
    }
}

// get parallel
// cos<A | B, B> = 1
// A | B
// out of place
// vector only
impl<'a, 'b> BitOr<&'b RMatrix> for &'a RMatrix {
    type Output = RMatrix;

    fn bitor(self, mat: &RMatrix) -> RMatrix {
        assert_eq!(mat.y, 1, "RMatrix.bitor(RMatrix): vector only");
        assert_eq!(self.x, mat.x, "RMatrix.bitor(RMatrix): vector size mismatch");
        let mut norm: f64 = 0.0;
        for i in 0..self.x {
            norm += mat.data[i] * mat.data[i];
        }
        let mut length = !mat ^ self;
        length /= norm;
        mat ^ length
    }
}

impl_op_2_mm!{impl BitOr, bitor for RMatrix, RMatrix, RMatrix}

// A |= &B
// in place
// vector only
impl<'b> BitOrAssign<&'b RMatrix> for RMatrix {
    fn bitor_assign(&mut self, mat: &RMatrix) {
        assert_eq!(mat.y, 1, "RMatrix.bitor_assign(&RMatrix): vector only");
        assert_eq!(self.x, mat.x, "RMatrix.bitor_assign(&RMatrix): vector size mismatch");
        let mut norm: f64 = 0.0;
        for i in 0..self.x {
            norm += mat.data[i] * mat.data[i];
        }
        let mut length = !mat ^ &*self;
        length /= norm;
        for j in 0..self.y {
            for i in 0..self.x {
                self.data[i * self.y + j] = mat.data[i] * length.data[j];
            }
        }
    }
}

// A |= B
// in place
// vector only
impl BitOrAssign<RMatrix> for RMatrix {
    fn bitor_assign(&mut self, mat: RMatrix) {
        assert_eq!(mat.y, 1, "RMatrix.bitor_assign(RMatrix): vector only");
        assert_eq!(self.x, mat.x, "RMatrix.bitor_assign(RMatrix): vector size mismatch");
        let mut norm: f64 = 0.0;
        for i in 0..self.x {
            norm += mat.data[i] * mat.data[i];
        }
        let mut length = !&mat ^ &*self;
        length /= norm;
        for j in 0..self.y {
            for i in 0..self.x {
                self.data[i * self.y + j] = mat.data[i] * length.data[j];
            }
        }
    }
}

// get perpendicular
// cos<A % B, B> = 0
// A % B
// out of place
// vector only
impl<'a, 'b> Rem<&'b RMatrix> for &'a RMatrix {
    type Output = RMatrix;

    fn rem(self, mat: &RMatrix) -> RMatrix {
        assert_eq!(mat.y, 1, "RMatrix.rem(RMatrix): vector only");
        assert_eq!(self.x, mat.x, "RMatrix.rem(RMatrix): vector size mismatch");
        let mut norm: f64 = 0.0;
        for i in 0..self.x {
            norm += mat.data[i] * mat.data[i];
        }
        let mut length = !mat ^ self;
        length /= norm;
        let mut ret_data: Vec<f64> = vec![0.0; self.x * self.y];
        for j in 0..self.y {
            for i in 0..self.x {
                ret_data[i * self.y + j] = self.data[i * self.y + j] - (mat.data[i] * length.data[j]);
            }
        }
        RMatrix {
            x: self.x,
            y: self.y,
            data: ret_data
        }
    }
}

impl_op_2_mm!{impl Rem, rem for RMatrix, RMatrix, RMatrix}

// A %= &B
// in place
// vector only
impl<'b> RemAssign<&'b RMatrix> for RMatrix {
    fn rem_assign(&mut self, mat: &RMatrix) {
        assert_eq!(mat.y, 1, "RMatrix.rem_assign(&RMatrix): vector only");
        assert_eq!(self.x, mat.x, "RMatrix.rem_assign(&RMatrix): vector size mismatch");
        let mut norm: f64 = 0.0;
        for i in 0..self.x {
            norm += mat.data[i] * mat.data[i];
        }
        let mut length = !mat ^ &*self;
        length /= norm;
        for j in 0..self.y {
            for i in 0..self.x {
                self.data[i * self.y + j] -= mat.data[i] * length.data[j];
            }
        }
    }
}

// A %= B
// in place
// vector only
impl RemAssign<RMatrix> for RMatrix {
    fn rem_assign(&mut self, mat: RMatrix) {
        assert_eq!(mat.y, 1, "RMatrix.rem_assign(RMatrix): vector only");
        assert_eq!(self.x, mat.x, "RMatrix.rem_assign(RMatrix): vector size mismatch");
        let mut norm: f64 = 0.0;
        for i in 0..self.x {
            norm += mat.data[i] * mat.data[i];
        }
        let mut length = !&mat ^ &*self;
        length /= norm;
        for j in 0..self.y {
            for i in 0..self.x {
                self.data[i * self.y + j] -= mat.data[i] * length.data[j];
            }
        }
    }
}

