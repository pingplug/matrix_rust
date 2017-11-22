mod r_matrix;

#[cfg(test)]
mod tests {
    fn feq(n1: f64, n2: f64) -> bool {
        let result: f64 = (n1 - n2).abs();
        if result > 0.00000000000001 {
            false
        } else {
            true
        }
    }

    #[test]
    fn basic() {
        use r_matrix::RMatrix;
        let diag = &RMatrix::gen_diag(4, 3, vec![1.0, 2.0, 3.0]);
        let eye = &RMatrix::gen_eye(4, 3);
        let ones = &RMatrix::gen_ones(3, 1);
        let zeros = &RMatrix::gen_zeros(4, 3);
        let rand = &RMatrix::gen_rand(1, 3);

        // out of place
        (ones + !rand);
        (ones + 3.5);
        (3.5 + !rand);
        (ones + !(rand ^ ones));
        ((rand ^ ones) + !rand);
        (ones - !rand);
        (ones - 3.5);
        (3.5 - !rand);
        (ones - !(rand ^ ones));
        ((rand ^ ones) - !rand);
        (ones * !rand);
        (ones * 3.5);
        (3.5 * !rand);
        (ones * !(rand ^ ones));
        ((rand ^ ones) * !rand);
        (ones / !rand);
        (ones / 3.5);
        (3.5 / !rand);
        (ones / !(rand ^ ones));
        ((rand ^ ones) / !rand);
        (rand ^ ones);
        (ones ^ rand);
        (ones ^ (rand ^ ones));
        ((rand ^ ones) ^ rand);
        (!diag | ones);
        (!diag % ones);

        // in place
        let mut a = ones.clone();
        a += !rand;
        a = ones.clone();
        a += 3.5;
        a = ones.clone();
        a += !(rand ^ ones);
        a = ones.clone();
        a -= !rand;
        a = ones.clone();
        a -= 3.5;
        a = ones.clone();
        a -= !(rand ^ ones);
        a = ones.clone();
        a *= !rand;
        a = ones.clone();
        a *= 3.5;
        a = ones.clone();
        a *= !(rand ^ ones);
        a = ones.clone();
        a /= !rand;
        a = ones.clone();
        a /= 3.5;
        a = ones.clone();
        a /= !(rand ^ ones);
        a = diag.clone();
        a <<= diag ^ !eye;
        a = diag.clone();
        a >>= !diag ^ zeros;
        a = diag.clone();
        a >>= rand ^ ones;
        a = diag.clone();
        a <<= rand ^ ones;
        a = !diag.clone();
        a |= ones;
        a = !diag.clone();
        a %= ones;

        let rand = &RMatrix::gen_rand(50, 200);
        let square1 = rand ^ !rand;
        let square2 = rand.square();
        assert!(feq((&square1 - square2).norm_2() / square1.norm_2(), 0.0));

        let mut rand = RMatrix::gen_rand(100, 100);
        let square1 = &rand ^ !&rand;
        rand.isquare();
        assert!(feq((&square1 - rand).norm_2() / square1.norm_2(), 0.0));
    }

    #[test]
    fn cholesky() {
        use r_matrix::RMatrix;
        let n: usize = 100;
        let mut rand = RMatrix::gen_rand_sym(n);
        rand.isquare();
        let l = rand.cholesky();
        assert!(feq((l.square() - &rand).norm_2() / rand.norm_2(), 0.0));

        let mut rand = RMatrix::gen_rand_ubi(n);
        rand.isquare();
        let l = rand.cholesky_tri();
        assert!(feq((l.square() - &rand).norm_f() / rand.norm_f(), 0.0));
    }

    #[test]
    fn lu() {
        use r_matrix::RMatrix;
        let n: usize = 100;
        let rand = &RMatrix::gen_rand(n, n);
        let (l, u) = rand.lu();
        assert!(feq(((l ^ u) - rand).norm_2() / rand.norm_2(), 0.0));
        let (p, l, u) = rand.plu();
        assert!(feq(((l ^ u) - (&p ^ rand)).norm_2() / (&p ^ rand).norm_2(), 0.0));
        let (q, l, u) = rand.qlu();
        assert!(feq(((l ^ u) - (rand ^ &q)).norm_2() / (rand ^ &q).norm_2(), 0.0));
        let (p, l, u) = rand.pplu();
        assert!(feq(((l ^ u) - (&p ^ rand ^ !&p)).norm_2() / (&p ^ rand ^ !&p).norm_2(), 0.0));
        let (p, q, l, u) = rand.pqlu();
        assert!(feq(((l ^ u) - (&p ^ rand ^ &q)).norm_2() / (&p ^ rand ^ &q).norm_2(), 0.0));

        let rand = &RMatrix::gen_rand_tri(n, n);
        let (l, u) = rand.lu_tri();
        assert!(feq(((l ^ u) - rand).norm_f() / rand.norm_f(), 0.0));
    }

    fn qr_test1(x: usize, y: usize, fx: f64, fy: f64) {
        use r_matrix::RMatrix;
        let rand = &RMatrix::gen_rand(x, y);

        let (q, r) = rand.cqr_cgs();
        // should fail if no fx
        assert!(feq(((!&q).square() - RMatrix::gen_eye(y, y)).norm_f() / (fx * fy), 0.0));
        assert!(feq(((q ^ r) - rand).norm_2() / rand.norm_2(), 0.0));

        let (q, r) = rand.cqr_mgs();
        assert!(feq(((!&q).square() - RMatrix::gen_eye(y, y)).norm_f() / fy, 0.0));
        assert!(feq(((q ^ r) - rand).norm_2() / rand.norm_2(), 0.0));
    }

    fn qr_test2(x: usize, y: usize, fx: f64, fy: f64) {
        use r_matrix::RMatrix;
        let rand = &RMatrix::gen_rand(x, y);

        let (q, r) = rand.qr_hh();
        assert!(feq((q.square() - RMatrix::gen_eye(x, x)).norm_f() / fx, 0.0));
        assert!(feq(((q ^ &r) - rand).norm_2() / rand.norm_2(), 0.0));

        let mut rand1 = rand.clone();
        rand1.iqr_hh();
        assert!(feq((rand1 - &r).norm_2() / r.norm_2(), 0.0));

        let (q, r) = rand.qr_givens();
        assert!(feq((q.square() - RMatrix::gen_eye(x, x)).norm_f() / fx, 0.0));
        assert!(feq(((q ^ &r) - rand).norm_2() / rand.norm_2(), 0.0));

        let mut rand1 = rand.clone();
        rand1.iqr_givens();
        assert!(feq((rand1 - &r).norm_2() / r.norm_2(), 0.0));

        let mut rand1 = rand.clone();
        let (u, v) = rand1.ipqb_hh();
        assert!(feq((u.square() - RMatrix::gen_eye(x, x)).norm_f() / fx, 0.0));
        assert!(feq((v.square() - RMatrix::gen_eye(y, y)).norm_f() / fy, 0.0));
        assert!(feq(((u ^ &rand1 ^ v) - rand).norm_2() / rand.norm_2(), 0.0));

        let r = &rand1.clone();
        let mut rand1 = rand.clone();
        rand1.ib_hh();
        assert!(feq((rand1 - r).norm_2() / r.norm_2(), 0.0));

        let mut rand1 = rand.clone();
        let (u, v) = rand1.ipqb_givens();
        assert!(feq((u.square() - RMatrix::gen_eye(x, x)).norm_f() / fx, 0.0));
        assert!(feq((v.square() - RMatrix::gen_eye(y, y)).norm_f() / fy, 0.0));
        assert!(feq(((u ^ &rand1 ^ v) - rand).norm_2() / rand.norm_2(), 0.0));

        let r = &rand1.clone();
        let mut rand1 = rand.clone();
        rand1.ib_givens();
        assert!(feq((rand1 - r).norm_2() / r.norm_2(), 0.0));
    }

    fn qr_test3(x: usize, _y: usize, fx: f64, _fy: f64) {
        use r_matrix::RMatrix;
        let rand = &RMatrix::gen_rand(x, x);

        let mut rand1 = rand.clone();
        let q = rand1.iqhq_hh();
        assert!(feq((q.square() - RMatrix::gen_eye(x, x)).norm_f() / fx, 0.0));
        assert!(feq(((&q ^ &rand1 ^ !&q) - rand).norm_2() / rand.norm_2(), 0.0));

        let r = &rand1.clone();
        let mut rand1 = rand.clone();
        rand1.ih_hh();
        assert!(feq((rand1 - r).norm_2() / r.norm_2(), 0.0));

        let mut rand1 = rand.clone();
        let q = rand1.iqhq_givens();
        assert!(feq((q.square() - RMatrix::gen_eye(x, x)).norm_f() / fx, 0.0));
        assert!(feq(((&q ^ &rand1 ^ !&q) - rand).norm_2() / rand.norm_2(), 0.0));

        let r = &rand1.clone();
        let mut rand1 = rand.clone();
        rand1.ih_givens();
        assert!(feq((rand1 - r).norm_2() / r.norm_2(), 0.0));

        let rand = &RMatrix::gen_rand_sym(x);

        let mut rand1 = rand.clone();
        let q = rand1.iqtq_hh();
        assert!(feq((q.square() - RMatrix::gen_eye(x, x)).norm_f() / fx, 0.0));
        assert!(feq(((&q ^ &rand1 ^ !&q) - rand).norm_2() / rand.norm_2(), 0.0));

        let r = &rand1.clone();
        let mut rand1 = rand.clone();
        rand1.it_hh();
        assert!(feq((rand1 - r).norm_2() / r.norm_2(), 0.0));

        let mut rand1 = rand.clone();
        let q = rand1.iqtq_givens();
        assert!(feq((q.square() - RMatrix::gen_eye(x, x)).norm_f() / fx, 0.0));
        assert!(feq(((&q ^ &rand1 ^ !&q) - rand).norm_2() / rand.norm_2(), 0.0));

        let r = &rand1.clone();
        let mut rand1 = rand.clone();
        rand1.it_givens();
        assert!(feq((rand1 - r).norm_2() / r.norm_2(), 0.0));
    }

    #[test]
    fn qr_long() {
        qr_test2(50, 200, 50.0, 200.0);
    }

    #[test]
    fn qr_square() {
        qr_test1(100, 100, 100.0, 100.0);
        qr_test2(100, 100, 100.0, 100.0);
        qr_test3(100, 100, 100.0, 100.0);
    }

    #[test]
    fn qr_tall() {
        qr_test1(200, 50, 200.0, 50.0);
        qr_test2(200, 50, 200.0, 50.0);
    }

    fn svd_test(x: usize, y: usize, fx: f64, fy: f64) {
        use r_matrix::RMatrix;
        let rand = &RMatrix::gen_rand(x, y);

        let mut rand1 = rand.clone();
        let (u, v) = rand1.isvd_qr();
        assert!(feq((u.square() - RMatrix::gen_eye(x, x)).norm_f() / fx, 0.0));
        assert!(feq((v.square() - RMatrix::gen_eye(y, y)).norm_f() / fy, 0.0));
        // should fail if no fx
        assert!(feq(((u ^ &rand1 ^ v) - rand).norm_2() / (rand.norm_2() * fx), 0.0));

        let r = &rand1.clone();
        let mut rand1 = rand.clone();
        rand1.isv_qr();
        assert!(feq((rand1 - r).norm_2() / r.norm_2(), 0.0));
    }

    #[test]
    fn svd_long() {
        svd_test(50, 200, 50.0, 200.0);
    }

    #[test]
    fn svd_square() {
        svd_test(100, 100, 100.0, 100.0);
    }

    #[test]
    fn svd_tall() {
        svd_test(200, 50, 200.0, 50.0);
    }

}
