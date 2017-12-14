mod r_matrix;

#[cfg(test)]
mod tests {
    use r_matrix::RMatrix;

    fn feq(n1: f64, n2: f64) -> bool {
        let result: f64 = (n1 - n2).abs();
        if result < 0.00000000000001 {
            true
        } else {
            false
        }
    }

    #[test]
    fn basic() {
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
        let n: usize = 100;
        let rand = &RMatrix::gen_rand_sym(n).square();
        let l = rand.chol();
        assert!(feq((l.square() - rand).norm_2() / rand.norm_2(), 0.0));

        let rand = &RMatrix::gen_rand_ubi(n).square();
        let l = rand.chol_tri();
        assert!(feq((l.square() - rand).norm_f() / rand.norm_f(), 0.0));
    }

    #[test]
    fn lu() {
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
        let rand = &RMatrix::gen_rand(x, y);

        let (q, r) = rand.qr_cgs();
        // should fail if no fx
        assert!(feq((q.square() - RMatrix::gen_eye(x, x)).norm_f() / (fx * fx), 0.0));
        assert!(feq(((q ^ &r) - rand).norm_2() / rand.norm_2(), 0.0));

        let (q, r) = rand.qr_mgs();
        assert!(feq((q.square() - RMatrix::gen_eye(x, x)).norm_f() / fx, 0.0));
        assert!(feq(((q ^ &r) - rand).norm_2() / rand.norm_2(), 0.0));

        let (q, r) = rand.qr_hh();
        assert!(feq((q.square() - RMatrix::gen_eye(x, x)).norm_f() / fx, 0.0));
        assert!(feq(((q ^ &r) - rand).norm_2() / rand.norm_2(), 0.0));

        let r1 = rand.r_hh();
        assert!(feq((r1 - &r).norm_2() / r.norm_2(), 0.0));

        let (q, r) = rand.qr_givens();
        assert!(feq((q.square() - RMatrix::gen_eye(x, x)).norm_f() / fx, 0.0));
        assert!(feq(((q ^ &r) - rand).norm_2() / rand.norm_2(), 0.0));

        let r1 = rand.r_givens();
        assert!(feq((r1 - &r).norm_2() / r.norm_2(), 0.0));

        let (u, b, v) = rand.pbq_hh();
        assert!(feq((u.square() - RMatrix::gen_eye(x, x)).norm_f() / fx, 0.0));
        assert!(feq((v.square() - RMatrix::gen_eye(y, y)).norm_f() / fy, 0.0));
        assert!(feq(((u ^ &b ^ v) - rand).norm_2() / rand.norm_2(), 0.0));

        let b1 = rand.b_hh();
        assert!(feq((b1 - &b).norm_2() / b.norm_2(), 0.0));

        let (u, b, v) = rand.pbq_givens();
        assert!(feq((u.square() - RMatrix::gen_eye(x, x)).norm_f() / fx, 0.0));
        assert!(feq((v.square() - RMatrix::gen_eye(y, y)).norm_f() / fy, 0.0));
        assert!(feq(((u ^ &b ^ v) - rand).norm_2() / rand.norm_2(), 0.0));

        let b1 = rand.b_givens();
        assert!(feq((b1 - &b).norm_2() / b.norm_2(), 0.0));
    }

    fn qr_test2(x: usize, y: usize, fx: f64, fy: f64) {
        qr_test1(x, y, fx, fy);
        let rand = &RMatrix::gen_rand(x, y);

        let (q, r) = rand.cqr_cgs();
        // should fail if no fx
        assert!(feq(((!&q).square() - RMatrix::gen_eye(y, y)).norm_f() / (fx * fy), 0.0));
        assert!(feq(((q ^ r) - rand).norm_2() / rand.norm_2(), 0.0));

        let (q, r) = rand.cqr_mgs();
        assert!(feq(((!&q).square() - RMatrix::gen_eye(y, y)).norm_f() / fy, 0.0));
        assert!(feq(((q ^ r) - rand).norm_2() / rand.norm_2(), 0.0));
    }

    fn qr_test3(x: usize, y: usize, fx: f64, fy: f64) {
        qr_test2(x, y, fx, fy);
        let rand = &RMatrix::gen_rand(x, x);

        let (q, h) = rand.qhq_hh();
        assert!(feq((q.square() - RMatrix::gen_eye(x, x)).norm_f() / fx, 0.0));
        assert!(feq(((&q ^ &h ^ !&q) - rand).norm_2() / rand.norm_2(), 0.0));

        let h1 = rand.h_hh();
        assert!(feq((h1 - &h).norm_2() / h.norm_2(), 0.0));

        let (q, h) = rand.qhq_givens();
        assert!(feq((q.square() - RMatrix::gen_eye(x, x)).norm_f() / fx, 0.0));
        assert!(feq(((&q ^ &h ^ !&q) - rand).norm_2() / rand.norm_2(), 0.0));

        let h1 = rand.h_givens();
        assert!(feq((h1 - &h).norm_2() / h.norm_2(), 0.0));

        let (_q, _h) = rand.qhq_arnoldi();
        //assert!(feq((q.square() - RMatrix::gen_eye(x, x)).norm_f() / fx, 0.0));
        //assert!(feq(((&q ^ h ^ !&q) - rand).norm_2() / rand.norm_2(), 0.0));

        let rand = &RMatrix::gen_rand_sym(x);

        let (q, t) = rand.qtq_hh();
        assert!(feq((q.square() - RMatrix::gen_eye(x, x)).norm_f() / fx, 0.0));
        assert!(feq(((&q ^ &t ^ !&q) - rand).norm_2() / rand.norm_2(), 0.0));

        let t1 = rand.t_hh();
        assert!(feq((t1 - &t).norm_2() / t.norm_2(), 0.0));

        let (q, t) = rand.qtq_givens();
        assert!(feq((q.square() - RMatrix::gen_eye(x, x)).norm_f() / fx, 0.0));
        assert!(feq(((&q ^ &t ^ !&q) - rand).norm_2() / rand.norm_2(), 0.0));

        let t1 = rand.t_givens();
        assert!(feq((t1 - &t).norm_2() / t.norm_2(), 0.0));

        let (_q, _t) = rand.qtq_lanczos();
        //assert!(feq((q.square() - RMatrix::gen_eye(x, x)).norm_f() / fx, 0.0));
        //assert!(feq(((&q ^ t ^ !&q) - rand).norm_2() / rand.norm_2(), 0.0));
    }

    fn qr_test4(x: usize, y: usize, fx: f64, fy: f64) {
        qr_test3(x, y, fx, fy);
        // TODO
        // SVD for skew-symmetric matrix is slow now
        // use norm_f instead of norm_2
        let rand = &RMatrix::gen_rand_ssym(x);

        let (q, t) = rand.qtq_hh();
        assert!(feq((q.square() - RMatrix::gen_eye(x, x)).norm_f() / fx, 0.0));
        assert!(feq(((&q ^ &t ^ !&q) - rand).norm_f() / rand.norm_f(), 0.0));

        let t1 = rand.t_hh();
        assert!(feq((t1 - &t).norm_f() / t.norm_f(), 0.0));

        let (q, t) = rand.qtq_givens();
        assert!(feq((q.square() - RMatrix::gen_eye(x, x)).norm_f() / fx, 0.0));
        assert!(feq(((&q ^ &t ^ !&q) - rand).norm_f() / rand.norm_f(), 0.0));

        let t1 = rand.t_givens();
        assert!(feq((t1 - &t).norm_f() / t.norm_f(), 0.0));

        let (_q, _t) = rand.qtq_lanczos();
        //assert!(feq((q.square() - RMatrix::gen_eye(x, x)).norm_f() / fx, 0.0));
        //assert!(feq(((&q ^ t ^ !&q) - rand).norm_f() / rand.norm_f(), 0.0));
    }

    #[test]
    fn qr_long() {
        qr_test1(1, 100, 1.0, 100.0);
        qr_test1(50, 200, 50.0, 200.0);
    }

    #[test]
    fn qr_square() {
        qr_test3(1, 1, 1.0, 1.0);
        qr_test4(100, 100, 100.0, 100.0);
    }

    #[test]
    fn qr_tall() {
        qr_test2(100, 1, 100.0, 1.0);
        qr_test2(200, 50, 200.0, 50.0);
    }

    fn svd_test(x: usize, y: usize, fx: f64, fy: f64) {
        let rand = &RMatrix::gen_rand(x, y);

        let (u, s, v) = rand.svd_qr();
        assert!(feq((u.square() - RMatrix::gen_eye(x, x)).norm_f() / fx, 0.0));
        assert!(feq((v.square() - RMatrix::gen_eye(y, y)).norm_f() / fy, 0.0));
        // should fail if no fx
        assert!(feq(((u ^ &s ^ v) - rand).norm_2() / (rand.norm_2() * fx), 0.0));

        let s1 = rand.sv_qr();
        assert!(feq((s1 - &s).norm_2() / s.norm_2(), 0.0));
    }

    #[test]
    fn svd_long() {
        svd_test(1, 100, 1.0, 100.0);
        svd_test(50, 200, 50.0, 200.0);
    }

    #[test]
    fn svd_square() {
        svd_test(1, 1, 1.0, 1.0);
        svd_test(100, 100, 100.0, 100.0);
    }

    #[test]
    fn svd_tall() {
        svd_test(100, 1, 100.0, 1.0);
        svd_test(200, 50, 200.0, 50.0);
    }

    fn solve_test1(rand: &RMatrix, b: &RMatrix) {
        let s = rand.cond() * b.norm_2();
        let _x = rand.solve_gdnr(b);
        //assert!(feq(((rand ^ x) - b).norm_2() / b.norm_2() / s, 0.0));

        let x = rand.solve_cgnr(b);
        assert!(feq(((rand ^ x) - b).norm_2() / b.norm_2() / s, 0.0));

        let x = rand.solve_lu(b);
        assert!(feq(((rand ^ x) - b).norm_2() / b.norm_2() / s, 0.0));
    }

    fn solve_test2(rand: &RMatrix, b: &RMatrix) {
        solve_test1(rand, b);
        let _s = rand.cond() * b.norm_2();

        // FIXME
        // this function can easily stuck in loop
        //let x = rand.solve_bicg(b);
        //assert!(feq(((rand ^ x) - b).norm_2() / b.norm_2() / s, 0.0));
    }

    fn solve_test3(rand: &RMatrix, b: &RMatrix) {
        solve_test2(rand, b);
        // FIXME?
        // this will break when A's eigenvalue close to zero or A is indefinite
        let _s = rand.cond() * b.norm_2();
        let _x = rand.solve_lanczos(b);
        //assert!(feq(((rand ^ x) - b).norm_2() / b.norm_2() / s, 0.0));

        let _x = rand.solve_minres(b);
        //assert!(feq(((rand ^ x) - b).norm_2() / b.norm_2() / s, 0.0));
    }

    fn solve_test4(rand: &RMatrix, b: &RMatrix) {
        solve_test3(rand, b);
        let s = rand.cond() * b.norm_2();
        let x = rand.solve_chol(b);
        assert!(feq(((rand ^ x) - b).norm_2() / b.norm_2() / s, 0.0));

        let _x = rand.solve_gd(b);
        //assert!(feq(((rand ^ x) - b).norm_2() / b.norm_2() / s, 0.0));

        let x = rand.solve_cg(b);
        assert!(feq(((rand ^ x) - b).norm_2() / b.norm_2() / s, 0.0));

        let x = rand.solve_pcg1(b);
        assert!(feq(((rand ^ x) - b).norm_2() / b.norm_2() / s, 0.0));

        let x = rand.solve_pcg3(b);
        assert!(feq(((rand ^ x) - b).norm_2() / b.norm_2() / s, 0.0));
    }

    fn solve_tri_test1(rand: &RMatrix, b: &RMatrix) {
        let s = rand.cond() * b.norm_2();
        let x = rand.solve_tri_lu(b);
        assert!(feq(((rand ^ x) - b).norm_2() / b.norm_2() / s, 0.0));
    }

    fn solve_tri_test2(rand: &RMatrix, b: &RMatrix) {
        solve_tri_test1(rand, b);
        let s = rand.cond() * b.norm_2();
        let x = rand.solve_tri_chol(b);
        assert!(feq(((rand ^ x) - b).norm_2() / b.norm_2() / s, 0.0));
    }

    #[test]
    fn solve_pos_sym() {
        let n: usize = 20;
        let b = &RMatrix::gen_rand(n, 1);
        let rand = &RMatrix::gen_rand_ubi(n).square();
        solve_tri_test2(rand, b);
        solve_test4(rand, b);

        let n: usize = 100;
        let b = &RMatrix::gen_rand(n, 1);
        let rand = &RMatrix::gen_rand_eig(n, (RMatrix::gen_rand(n, 1) * 100.0 + 10.0).get_data());
        solve_test4(rand, b);
    }

    #[test]
    fn solve_sym() {
        let n: usize = 100;
        let b = &RMatrix::gen_rand(n, 1);

        let rand = &RMatrix::gen_rand_eig(n, (RMatrix::gen_rand(n, 1) * 100.0 - 50.0).get_data());
        solve_test3(rand, b);
    }

    #[test]
    fn solve_pos() {
        let n: usize = 100;
        let b = &RMatrix::gen_rand(n, 1);

        let rand = &(RMatrix::gen_rand_eig(n, (RMatrix::gen_rand(n, 1) * 100.0 + 10.0).get_data()) + RMatrix::gen_rand_ssym(n));
        solve_test2(rand, b);
    }

    #[test]
    fn solve() {
        let n: usize = 100;
        let b = &RMatrix::gen_rand(n, 1);

        let rand = &RMatrix::gen_rand_tri(n, n);
        solve_tri_test1(rand, b);
        solve_test1(rand, b);

        let rand = &RMatrix::gen_rand(n, n);
        solve_test1(rand, b);
    }

    fn lsq_test(x: usize, y: usize) {
        let rand = &RMatrix::gen_rand(x, y);
        let b = &RMatrix::gen_rand(x, 1);
        let s = rand.cond() * b.norm_2();

        let _x = rand.solve_gdnr(b);
        //assert!(feq((!(!(rand ^ x) ^ rand) - !(!b ^ rand)).norm_2() / (!(!b ^ rand)).norm_2() / s, 0.0));

        let x = rand.solve_cgnr(b);
        assert!(feq((!(!(rand ^ x) ^ rand) - !(!b ^ rand)).norm_2() / (!(!b ^ rand)).norm_2() / s, 0.0));

        let x = rand.lsq_qr(b);
        assert!(feq((!(!(rand ^ x) ^ rand) - !(!b ^ rand)).norm_2() / (!(!b ^ rand)).norm_2() / s, 0.0));
    }

    #[test]
    fn lsq() {
        lsq_test(100, 1);
        lsq_test(200, 50);
    }
}

