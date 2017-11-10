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

    // 1-norm of a VECTOR
    pub fn norm_1(&self) -> f64 {
        assert_eq!(self.y, 1, "RMatrix.norm_1(): vector only");
        let mut ret: f64 = 0.0;
        for i in 0..self.x {
            ret += self.data[i].abs();
        }
        ret
    }

    // 2-norm of a VECTOR
    pub fn norm_2(&self) -> f64 {
        assert_eq!(self.y, 1, "RMatrix.norm_2(): vector only");
        let mut ret: f64 = 0.0;
        for i in 0..self.x {
            ret += self.data[i] * self.data[i];
        }
        ret.sqrt()
    }

    // f-norm of a matrix
    pub fn norm_f(&self) -> f64 {
        let mut ret: f64 = 0.0;
        for i in 0..(self.x * self.y) {
            ret += self.data[i] * self.data[i];
        }
        ret.sqrt()
    }

    // infinity-norm of a VECTOR
    pub fn norm_i(&self) -> f64 {
        assert_eq!(self.y, 1, "RMatrix.norm_i(): vector only");
        let mut ret: f64 = self.data[0];
        for i in 1..self.x {
            ret = ret.max(self.data[i]);
        }
        ret
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

    // Cholesky decomposition
    // A = L ^ !L
    pub fn cholesky(&self) -> RMatrix {
        assert_eq!(self.x, self.y, "RMatrix.cholesky(): square matrix only");
        let mut ret_data: Vec<f64> = vec![0.0; self.x * self.y];
        for j in 0..self.y {
            ret_data[j * self.y + j] = self.data[j * self.y + j];
            for k in 0..j {
                ret_data[j * self.y + j] -= ret_data[j * self.y + k] * ret_data[j * self.y + k];
            }
            ret_data[j * self.y + j] = ret_data[j * self.y + j].sqrt();
            for i in (j + 1)..self.x {
                ret_data[i * self.y + j] = self.data[i * self.y + j];
                for k in 0..j {
                    ret_data[i * self.y + j] -= ret_data[i * self.y + k] * ret_data[j * self.y + k];
                }
                ret_data[i * self.y + j] /= ret_data[j * self.y + j];
            }
        }
        RMatrix {
            x: self.x,
            y: self.y,
            data: ret_data
        }
    }

    // Cholesky decomposition
    // A = L ^ !L
    pub fn pp_cholesky(&self) -> RMatrix {
        assert_eq!(self.x, self.y, "RMatrix.pp_cholesky(): square matrix only");
        let mut ret_data: Vec<f64> = vec![0.0; self.x * self.y];
        for i in 0..self.x {
            for j in 0..(i + 1) {
                ret_data[i * self.y + j] = self.data[i * self.y + j];
            }
        }
        for i in 0..self.x {
            for j in 0..i {
                for k in 0..j {
                    ret_data[i * self.y + j] -= ret_data[i * self.y + k] * ret_data[j * self.y + k];
                }
                ret_data[i * self.y + j] /= ret_data[j * self.y + j];
                ret_data[i * self.y + i] -= ret_data[i * self.y + j] * ret_data[i * self.y + j];
            }
            ret_data[i * self.y + i] = ret_data[i * self.y + i].sqrt();
        }
        RMatrix {
            x: self.x,
            y: self.y,
            data: ret_data
        }
    }

    // LU decomposition
    // A = L ^ U
    pub fn lu(&self) -> (RMatrix, RMatrix) {
        assert_eq!(self.x, self.y, "RMatrix.lu(): square matrix only");
        let mut ret_l_data: Vec<f64> = vec![0.0; self.x * self.y];
        let mut ret_u_data: Vec<f64> = self.data.clone();
        for j in 0..self.y {
            for i in (j + 1)..self.x {
                ret_l_data[i * self.y + j] = ret_u_data[i * self.y + j] / ret_u_data[j * self.y + j];
                for k in j..self.y {
                    ret_u_data[i * self.y + k] -= ret_u_data[j * self.y + k] * ret_l_data[i * self.y + j];
                }
            }
        }
        for i in 0..self.x {
            ret_l_data[i * self.y + i] = 1.0;
        }
        (RMatrix {
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
    // A = Q ^ R
    pub fn qr_hh(&self) -> (RMatrix, RMatrix) {
        let mut ret_q_data: Vec<f64> = vec![0.0; self.x * self.x];
        let mut ret_r_data: Vec<f64> = self.data.clone();
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
                tmp[i] = ret_r_data[i * self.y + j];
            }
            n2 = 0.0;
            for i in j..self.x {
                n2 += tmp[i] * tmp[i];
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
                    n2 += ret_r_data[i * self.y + k] * tmp[i];
                }
                for i in j..self.x {
                    ret_r_data[i * self.y + k] -= 2.0 * n2 * tmp[i];
                }
            }
            for i in (j + 1)..self.x {
                ret_r_data[i * self.y + j] = 0.0;
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
        (RMatrix {
            x: self.x,
            y: self.x,
            data: ret_q_data
        },
        RMatrix {
            x: self.x,
            y: self.y,
            data: ret_r_data
        })
    }

    // QR decomposition, Householder
    // in place
    // without Q
    pub fn iqr_hh(&mut self) {
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
    // A = Q ^ R
    pub fn qr_givens(&self) -> (RMatrix, RMatrix) {
        let mut ret_q_data: Vec<f64> = vec![0.0; self.x * self.x];
        let mut ret_r_data: Vec<f64> = self.data.clone();
        let mut x: f64;
        let mut y: f64;
        let mut c: f64;
        let mut s: f64;
        for i in 0..self.x {
            ret_q_data[i * self.x + i] = 1.0;
        }
        for i in 0..self.x {
            for j in 0..i.min(self.y) {
                // Givens
                x = ret_r_data[j * self.y + j];
                y = ret_r_data[i * self.y + j];
                c = x / (x * x + y * y).sqrt();
                s = y / (x * x + y * y).sqrt();
                for k in j..self.y {
                    x = ret_r_data[j * self.y + k];
                    y = ret_r_data[i * self.y + k];
                    ret_r_data[j * self.y + k] =  x * c + y * s;
                    ret_r_data[i * self.y + k] = -x * s + y * c;
                }
                ret_r_data[i * self.y + j] = 0.0;
                // get Q
                for k in 0..(i + 1) {
                    x = ret_q_data[k * self.x + j];
                    y = ret_q_data[k * self.x + i];
                    ret_q_data[k * self.x + j] =  x * c + y * s;
                    ret_q_data[k * self.x + i] = -x * s + y * c;
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
            y: self.y,
            data: ret_r_data
        })
    }

    // QR decomposition, Givens
    // in place
    // without Q
    pub fn iqr_givens(&mut self) {
        let mut x: f64;
        let mut y: f64;
        let mut c: f64;
        let mut s: f64;
        for i in 0..self.x {
            for j in 0..i.min(self.y) {
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
        assert_eq!(self.x, self.y, "RMatrix.iqhq_hh(): square matrix only");
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
        assert_eq!(self.x, self.y, "RMatrix.iqhq_givens(): square matrix only");
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
            }
        }
        RMatrix {
            x: self.x,
            y: self.x,
            data: ret_q_data
        }
    }

    // tridiag matrix decomposition, Householder
    // A must be (skew-)symmetric
    // in place
    // A = Q ^ T ^ !Q
    pub fn iqtq_hh(&mut self) -> RMatrix {
        assert_eq!(self.x, self.y, "RMatrix.iqtq_hh(): square matrix only");
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

    // tridiag matrix decomposition, Givens
    // A must be (skew-)symmetric
    // in place
    // A = Q ^ T ^ !Q
    pub fn iqtq_givens(&mut self) -> RMatrix {
        assert_eq!(self.x, self.y, "RMatrix.iqtq_givens(): square matrix only");
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

    // up/dn bidiag matrix decomposition, Householder
    // in place
    // A = P ^ B ^ Q
    pub fn ipqb_hh(&mut self) -> (RMatrix, RMatrix) {
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

    // up/dn bidiag matrix decomposition, Givens
    // in place
    // A = P ^ B ^ Q
    pub fn ipqb_givens(&mut self) -> (RMatrix, RMatrix) {
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

    // QR decomposition for up bidiag, Givens
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

    // QR decomposition for down bidiag, Givens
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

    // two QR decompositions for bidiag matrix, Givens
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
        assert_eq!(self.x, self.y, "RMatrix.ih_hh(): square matrix only");
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
        }
    }

    // Hessenberg matrix decomposition, Givens
    // in place
    // without Q
    pub fn ih_givens(&mut self) {
        assert_eq!(self.x, self.y, "RMatrix.ih_givens(): square matrix only");
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
            }
        }
    }

    // tridiag matrix decomposition, Householder
    // A must be (skew-)symmetric
    // in place
    // without Q
    pub fn it_hh(&mut self) {
        assert_eq!(self.x, self.y, "RMatrix.it_hh(): square matrix only");
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

    // tridiag matrix decomposition, Givens
    // A must be (skew-)symmetric
    // in place
    // without Q
    pub fn it_givens(&mut self) {
        assert_eq!(self.x, self.y, "RMatrix.it_givens(): square matrix only");
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

    // up/dn bidiag matrix decomposition, Householder
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

                n += 1;
            }
        }
    }

    // up/dn bidiag matrix decomposition, Givens
    // in place
    // without P or Q
    pub fn ib_givens(&mut self) {
        let mut x: f64;
        let mut y: f64;
        let mut c: f64;
        let mut s: f64;
        if self.x < self.y {
            for n in 0..(self.x - 1) {
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
            for n in 0..(self.y - 1) {
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

    // QR decomposition for up bidiag, Givens
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

    // QR decomposition for down bidiag, Givens
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

    // two QR decompositions for bidiag matrix, Givens
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

    // SVD
    // in place
    // A = U ^ S ^ V
    pub fn isvd_qr(&mut self) -> (RMatrix, RMatrix) {
        let (mut u, mut v) = self.ipqb_hh();
        let n: usize = self.x.min(self.y);
        let mut n1: f64;
        let mut n2: f64;
        let mut delta: f64 = std::f64::MAX;
        while delta > 0.001 {
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
        (u, v)
    }

    // SVD
    // in place
    // without U or V
    pub fn isv_qr(&mut self) {
        self.ib_hh();
        let n: usize = self.x.min(self.y);
        let mut n1: f64;
        let mut n2: f64;
        let mut delta: f64 = std::f64::MAX;
        while delta > 0.0000000000000000001 {
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

// operators
// addition
// out of place
impl<'a, 'b> Add<&'b RMatrix> for &'a RMatrix {
    type Output = RMatrix;

    fn add(self, mat: &RMatrix) -> RMatrix {
        if mat.x == 1 && mat.y == 1 {
            return self + mat.data[0]
        }
        if self.x == 1 && self.y == 1 {
            return self.data[0] + mat
        }
        assert_eq!(self.x, mat.x, "RMatrix.add(&RMatrix): matrix size mismatch");
        assert_eq!(self.y, mat.y, "RMatrix.add(&RMatrix): matrix size mismatch");
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

// out of place
impl<'a> Add<f64> for &'a RMatrix {
    type Output = RMatrix;

    fn add(self, num: f64) -> RMatrix {
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

// out of place
impl<'a> Add<&'a RMatrix> for f64 {
    type Output = RMatrix;

    fn add(self, mat: &'a RMatrix) -> RMatrix {
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

// in place
impl AddAssign<f64> for RMatrix {
    fn add_assign(&mut self, num: f64) {
        for i in 0..(self.x * self.y) {
            self.data[i] += num;
        }
    }
}

// subtraction
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
        assert_eq!(self.x, mat.x, "RMatrix.sub(&RMatrix): matrix size mismatch");
        assert_eq!(self.y, mat.y, "RMatrix.sub(&RMatrix): matrix size mismatch");
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

// out of place
impl<'a> Sub<f64> for &'a RMatrix {
    type Output = RMatrix;

    fn sub(self, num: f64) -> RMatrix {
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

// out of place
impl<'a> Sub<&'a RMatrix> for f64 {
    type Output = RMatrix;

    fn sub(self, mat: &'a RMatrix) -> RMatrix {
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

// in place
impl SubAssign<f64> for RMatrix {
    fn sub_assign(&mut self, num: f64) {
        for i in 0..(self.x * self.y) {
            self.data[i] -= num;
        }
    }
}

// multiplication
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
        assert_eq!(self.x, mat.x, "RMatrix.mul(&RMatrix): matrix size mismatch");
        assert_eq!(self.y, mat.y, "RMatrix.mul(&RMatrix): matrix size mismatch");
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

// out of place
impl<'a> Mul<f64> for &'a RMatrix {
    type Output = RMatrix;

    fn mul(self, num: f64) -> RMatrix {
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

// out of place
impl<'a> Mul<&'a RMatrix> for f64 {
    type Output = RMatrix;

    fn mul(self, mat: &'a RMatrix) -> RMatrix {
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

// in place
impl MulAssign<f64> for RMatrix {
    fn mul_assign(&mut self, num: f64) {
        for i in 0..(self.x * self.y) {
            self.data[i] *= num;
        }
    }
}

// division
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
        assert_eq!(self.x, mat.x, "RMatrix.div(&RMatrix): matrix size mismatch");
        assert_eq!(self.y, mat.y, "RMatrix.div(&RMatrix): matrix size mismatch");
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

// out of place
impl<'a> Div<f64> for &'a RMatrix {
    type Output = RMatrix;

    fn div(self, num: f64) -> RMatrix {
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

// out of place
impl<'a> Div<&'a RMatrix> for f64 {
    type Output = RMatrix;

    fn div(self, mat: &'a RMatrix) -> RMatrix {
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

// in place
impl DivAssign<f64> for RMatrix {
    fn div_assign(&mut self, num: f64) {
        for i in 0..(self.x * self.y) {
            self.data[i] /= num;
        }
    }
}

// matrix multiplication
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
        assert_eq!(self.y, mat.x, "RMatrix.bitxor(&RMatrix): matrix size mismatch");
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

// in place left matrix multiplication
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

// in place right matrix multiplication
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

// transpose
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
// out of place
// vector only
impl<'a, 'b> BitOr<&'b RMatrix> for &'a RMatrix {
    type Output = RMatrix;

    fn bitor(self, mat: &RMatrix) -> RMatrix {
        assert_eq!(mat.y, 1, "RMatrix.bitor(&RMatrix): vector only");
        assert_eq!(self.x, mat.x, "RMatrix.bitor(&RMatrix): vector size mismatch");
        let mut norm: f64 = 0.0;
        for i in 0..self.x {
            norm += mat.data[i] * mat.data[i];
        }
        let mut length = &!mat ^ self;
        length /= norm;
        mat ^ &length
    }
}

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
        let mut length = &!mat ^ self;
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
// out of place
// vector only
impl<'a, 'b> Rem<&'b RMatrix> for &'a RMatrix {
    type Output = RMatrix;

    fn rem(self, mat: &RMatrix) -> RMatrix {
        assert_eq!(mat.y, 1, "RMatrix.rem(&RMatrix): vector only");
        assert_eq!(self.x, mat.x, "RMatrix.rem(&RMatrix): vector size mismatch");
        let mut norm: f64 = 0.0;
        for i in 0..self.x {
            norm += mat.data[i] * mat.data[i];
        }
        let mut length = &!mat ^ self;
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
        let mut length = &!mat ^ self;
        length /= norm;
        for j in 0..self.y {
            for i in 0..self.x {
                self.data[i * self.y + j] -= mat.data[i] * length.data[j];
            }
        }
    }
}

