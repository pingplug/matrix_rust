mod r_matrix;

#[cfg(test)]
mod tests {
    fn feq(n1: f64, n2: f64) -> bool {
        let result: f64 = (n1 - n2).abs();
        if result > 0.0000000001 {
            false
        } else {
            true
        }
    }

    #[test]
    fn out_of_place() {
        use r_matrix::RMatrix;
        let diag = RMatrix::gen_diag(4, 3, vec![1.0, 2.0, 3.0]);
        let eye = RMatrix::gen_eye(4, 3);
        let ones = RMatrix::gen_ones(3, 1);
        let zeros = RMatrix::gen_zeros(4, 3);
        let rand = RMatrix::gen_rand(1, 3);

        // out of place
        (&ones + &!&rand);
        (&ones + 3.5);
        (3.5 + &!&rand);
        (&ones + &!&(&rand ^ &ones));
        (&(&rand ^ &ones) + &!&rand);
        (&ones - &!&rand);
        (&ones - 3.5);
        (3.5 - &!&rand);
        (&ones - &!&(&rand ^ &ones));
        (&(&rand ^ &ones) - &!&rand);
        (&ones * &!&rand);
        (&ones * 3.5);
        (3.5 * &!&rand);
        (&ones * &!&(&rand ^ &ones));
        (&(&rand ^ &ones) * &!&rand);
        (&ones / &!&rand);
        (&ones / 3.5);
        (3.5 / &!&rand);
        (&ones / &!&(&rand ^ &ones));
        (&(&rand ^ &ones) / &!&rand);
        (&rand ^ &ones);
        (&ones ^ &rand);
        (&ones ^ &(&rand ^ &ones));
        (&(&rand ^ &ones) ^ &rand);
        (&!&diag | &ones);
        (&!&diag % &ones);
    
        // in place
        let mut a = ones.clone();
        a += &!&rand;
        a = ones.clone();
        a += 3.5;
        a = ones.clone();
        a += &!&(&rand ^ &ones);
        a = ones.clone();
        a -= &!&rand;
        a = ones.clone();
        a -= 3.5;
        a = ones.clone();
        a -= &!&(&rand ^ &ones);
        a = ones.clone();
        a *= &!&rand;
        a = ones.clone();
        a *= 3.5;
        a = ones.clone();
        a *= &!&(&rand ^ &ones);
        a = ones.clone();
        a /= &!&rand;
        a = ones.clone();
        a /= 3.5;
        a = ones.clone();
        a /= &!&(&rand ^ &ones);
        a = diag.clone();
        a <<= &(&diag ^ &!&eye);
        a = diag.clone();
        a >>= &(&!&diag ^ &zeros);
        a = diag.clone();
        a >>= &(&rand ^ &ones);
        a = diag.clone();
        a <<= &(&rand ^ &ones);
        a = !&diag.clone();
        a |= &ones;
        a = !&diag.clone();
        a %= &ones;
    }

    #[test]
    fn cholesky() {
        use r_matrix::RMatrix;
        let n: usize = 100;
        let mut rand = RMatrix::gen_eye(n, n);
        rand *= 100.0;
        rand += &RMatrix::gen_rand_sym(n);
        let l = rand.cholesky();
        assert!(feq((&l ^ &!&l).norm_f() / rand.norm_f(), 1.0));
        let l = rand.pp_cholesky();
        assert!(feq((&l ^ &!&l).norm_f() / rand.norm_f(), 1.0));
    }

    #[test]
    fn lu() {
        use r_matrix::RMatrix;
        let n: usize = 100;
        let rand = RMatrix::gen_rand(n, n);
        let (l, u) = rand.lu();
        assert!(feq((&l ^ &u).norm_f() / rand.norm_f(), 1.0));
        let (p, l, u) = rand.plu();
        assert!(feq((&l ^ &u).norm_f() / (&p ^ &rand).norm_f(), 1.0));
        let (q, l, u) = rand.qlu();
        assert!(feq((&l ^ &u).norm_f() / (&rand ^ &q).norm_f(), 1.0));
        let (p, l, u) = rand.pplu();
        assert!(feq((&l ^ &u).norm_f() / (&(&p ^ &rand) ^ &!&p).norm_f(), 1.0));
        let (p, q, l, u) = rand.pqlu();
        assert!(feq((&l ^ &u).norm_f() / (&(&p ^ &rand) ^ &q).norm_f(), 1.0));
    }

    #[test]
    fn qr_long() {
        use r_matrix::RMatrix;
        let x: usize = 50;
        let y: usize = 200;
        let fx: f64 = 50.0;
        let fy: f64 = 200.0;
        let rand = RMatrix::gen_rand(x, y);

        let (q, r) = rand.qr_hh();
        assert!(feq((&(&!&q ^ &q) - &RMatrix::gen_eye(x, x)).norm_f() / (fx * fx), 0.0));
        assert!(feq((&q ^ &r).norm_f() / rand.norm_f(), 1.0));
    
        let mut rand1 = rand.clone();
        rand1.iqr_hh();
    
        let (q, r) = rand.qr_givens();
        assert!(feq((&(&!&q ^ &q) - &RMatrix::gen_eye(x, x)).norm_f() / (fx * fx), 0.0));
        assert!(feq((&q ^ &r).norm_f() / rand.norm_f(), 1.0));
    
        let mut rand1 = rand.clone();
        rand1.iqr_givens();

        let mut rand1 = rand.clone();
        let (u, v) = rand1.isvd_qr();
        assert!(feq((&(&!&u ^ &u) - &RMatrix::gen_eye(x, x)).norm_f() / (fx * fx), 0.0));
        assert!(feq((&(&!&v ^ &v) - &RMatrix::gen_eye(y, y)).norm_f() / (fy * fy), 0.0));
        assert!(feq((&(&u ^ &rand1) ^ &v).norm_f() / rand.norm_f(), 1.0));
    
        let mut rand1 = rand.clone();
        rand1.isv_qr();
    }

    #[test]
    fn qr_square() {
        use r_matrix::RMatrix;
        let x: usize = 100;
        let y: usize = 100;
        let fx: f64 = 100.0;
        let fy: f64 = 100.0;
        let rand = RMatrix::gen_rand(x, y);

        let (q, r) = rand.cqr_cgs();
        assert!(feq((&(&!&q ^ &q) - &RMatrix::gen_eye(y, y)).norm_f() / (fy * fy), 0.0));
        assert!(feq((&q ^ &r).norm_f() / rand.norm_f(), 1.0));
    
        let (q, r) = rand.cqr_mgs();
        assert!(feq((&(&!&q ^ &q) - &RMatrix::gen_eye(y, y)).norm_f() / (fy * fy), 0.0));
        assert!(feq((&q ^ &r).norm_f() / rand.norm_f(), 1.0));
    
        let (q, r) = rand.qr_hh();
        assert!(feq((&(&!&q ^ &q) - &RMatrix::gen_eye(x, x)).norm_f() / (fx * fx), 0.0));
        assert!(feq((&q ^ &r).norm_f() / rand.norm_f(), 1.0));
    
        let mut rand1 = rand.clone();
        rand1.iqr_hh();
    
        let (q, r) = rand.qr_givens();
        assert!(feq((&(&!&q ^ &q) - &RMatrix::gen_eye(x, x)).norm_f() / (fx * fx), 0.0));
        assert!(feq((&q ^ &r).norm_f() / rand.norm_f(), 1.0));
    
        let mut rand1 = rand.clone();
        rand1.iqr_givens();

        let mut rand1 = rand.clone();
        let (u, v) = rand1.isvd_qr();
        assert!(feq((&(&!&u ^ &u) - &RMatrix::gen_eye(x, x)).norm_f() / (fx * fx), 0.0));
        assert!(feq((&(&!&v ^ &v) - &RMatrix::gen_eye(y, y)).norm_f() / (fy * fy), 0.0));
        assert!(feq((&(&u ^ &rand1) ^ &v).norm_f() / rand.norm_f(), 1.0));
    
        let mut rand1 = rand.clone();
        rand1.isv_qr();
    }

    #[test]
    fn qr_tall() {
        use r_matrix::RMatrix;
        let x: usize = 200;
        let y: usize = 50;
        let fx: f64 = 200.0;
        let fy: f64 = 50.0;
        let rand = RMatrix::gen_rand(x, y);

        let (q, r) = rand.cqr_cgs();
        assert!(feq((&(&!&q ^ &q) - &RMatrix::gen_eye(y, y)).norm_f() / (fy * fy), 0.0));
        assert!(feq((&q ^ &r).norm_f() / rand.norm_f(), 1.0));
    
        let (q, r) = rand.cqr_mgs();
        assert!(feq((&(&!&q ^ &q) - &RMatrix::gen_eye(y, y)).norm_f() / (fy * fy), 0.0));
        assert!(feq((&q ^ &r).norm_f() / rand.norm_f(), 1.0));
    
        let (q, r) = rand.qr_hh();
        assert!(feq((&(&!&q ^ &q) - &RMatrix::gen_eye(x, x)).norm_f() / (fx * fx), 0.0));
        assert!(feq((&q ^ &r).norm_f() / rand.norm_f(), 1.0));
    
        let mut rand1 = rand.clone();
        rand1.iqr_hh();
    
        let (q, r) = rand.qr_givens();
        assert!(feq((&(&!&q ^ &q) - &RMatrix::gen_eye(x, x)).norm_f() / (fx * fx), 0.0));
        assert!(feq((&q ^ &r).norm_f() / rand.norm_f(), 1.0));
    
        let mut rand1 = rand.clone();
        rand1.iqr_givens();

        let mut rand1 = rand.clone();
        let (u, v) = rand1.isvd_qr();
        assert!(feq((&(&!&u ^ &u) - &RMatrix::gen_eye(x, x)).norm_f() / (fx * fx), 0.0));
        assert!(feq((&(&!&v ^ &v) - &RMatrix::gen_eye(y, y)).norm_f() / (fy * fy), 0.0));
        assert!(feq((&(&u ^ &rand1) ^ &v).norm_f() / rand.norm_f(), 1.0));
    
        let mut rand1 = rand.clone();
        rand1.isv_qr();
    }
}
