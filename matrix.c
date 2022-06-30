#include <immintrin.h>
#include <omp.h>
#define BLOCK_SIZE (64)
#define arr(x, y) ((x)*n + (y))
#define idx(blk, off) ((blk)*BLOCK_SIZE + (off))
void matrix(double *a, double *b, double *c, long n) { /* assume $n \in 64\mathbb{Z}$ */
  long N;
  __m256d tmp0, tmp1, tmp2, tmp3, res0, res1, res2, res3;

  N = n / BLOCK_SIZE;
  omp_set_num_threads(8); /* physical cores */
#pragma omp parallel for schedule(static)
  for (long it = 0; it < N; it++)
  {
    for (long k = 0; k < n; k += 4)
      for (long i = 0; i < BLOCK_SIZE; i++) {
        tmp0 = _mm256_broadcast_sd(&a[arr(idx(it, i), k)]);
        tmp1 = _mm256_broadcast_sd(&a[arr(idx(it, i), k + 1)]);
        tmp2 = _mm256_broadcast_sd(&a[arr(idx(it, i), k + 2)]);
        tmp3 = _mm256_broadcast_sd(&a[arr(idx(it, i), k + 3)]);
        for (long j = 0; j < n; j += 4) {
          res0 = _mm256_loadu_pd(&b[arr(k, j)]);
          res1 = _mm256_loadu_pd(&b[arr(k + 1, j)]);
          res2 = _mm256_loadu_pd(&b[arr(k + 2, j)]);
          res3 = _mm256_loadu_pd(&b[arr(k + 3, j)]);
          res1 = _mm256_mul_pd(res1, tmp1);
          res3 = _mm256_mul_pd(res3, tmp3);
          res0 = _mm256_fmadd_pd(res0, tmp0, res1);
          res2 = _mm256_fmadd_pd(res2, tmp2, res3);
          res0 = _mm256_add_pd(res0, res2);
          _mm256_storeu_pd(
              &c[arr(idx(it, i), j)],
              _mm256_add_pd(res0, _mm256_loadu_pd(&c[arr(idx(it, i), j)])));
        }
      }
  }
}
#undef arr
#undef idx
