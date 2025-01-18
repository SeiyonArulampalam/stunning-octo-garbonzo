// assume we applied fill-in to the original rowptr, colptr now from suitesparse
// can we now solve with cusparse correctly?

// #include "cusparse_solve_utils.h"
// #include "fea_utils.h"

#include <cholmod.h>
#include <cstring>
#include <stdio.h>

int main() {
  using T = double;

  // ignore the block size aspect of this
  size_t n = 6;     // Matrix dimension
  int nvalues = 28; // Number of nonzeros
  int nzmax = nvalues;

  // Initialize Cholmod
  cholmod_common common;
  cholmod_start(&common);

  // switch to rowPtr, colPtr format
  // apply fill-in to lower and upper triangular places
  int Ap[] = {0, 4, 10, 14, 20, 24, 28}; // rowPtr
  int Ai[] = {0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3,
              0, 1, 2, 3, 4, 5, 1, 3, 4, 5, 1, 3, 4, 5};
  double Ax[nvalues];
  memset(Ax, 0.0, nvalues * sizeof(double));
  // for (int ival = 0; ival < nvalues; ival++) {
  //     values[ival] = rand() % 1;
  // }
  // double b[] = {1, 2, 3, 4, 5, 6};

  // Allocate the sparse matrix
  cholmod_sparse *A =
      cholmod_allocate_sparse(n, n, nzmax, 1, 1, 1, CHOLMOD_REAL, &common);
  A->p = Ap;
  A->i = Ai;
  A->x = Ax;
  A->stype = -1; // lower triangular

  // define the RHS as well
  // size_t n = 4;
  cholmod_dense *b = cholmod_allocate_dense(n, 1, n, CHOLMOD_REAL, &common);
  double *b_data = (double *)b->x;
  for (int i = 0; i < n; i++) {
    b_data[i] = i + 1;
  }

  // Perform symbolic factorization
  cholmod_factor *L = cholmod_analyze(A, &common);
  // or I can figure out how to get the fill-in pattern from this

  // Check if factorization was successful
  if (L == NULL) {
    printf("Error: Symbolic factorization failed.\n");
    cholmod_free_sparse(&A, &common);
    cholmod_finish(&common);
    return 0;
  }

  // to actually get numerical values in L
  // we can just call this on the GPU to get the fill-in pattern I guess
  cholmod_factorize(A, L, &common);

  // Cast the void* pointers to the appropriate types
  int *Lp = (int *)L->p; // Column pointers in the factorization
  int *Li = (int *)L->i; // Row indices in the factorization

  //   printf("L->n = %d\n", (int)L->n);
  //   printf("L->nzmax = %d\n", (int)L->nzmax);
  //   printf("L->nsuper = %d\n", (int)L->nsuper);
  //   // printf("Lp[0] = %d\n", (int)L->p[0]);

  // Print the symbolic fill-in pattern
  printf("Symbolic fill-in pattern (CSC format):\n");
  for (int col = 0; col < L->n;
       ++col) { // Use A->ncol for the number of columns
    printf("Column %d: ", col);
    for (int i = Lp[col]; i < Lp[col + 1]; ++i) {
      printf("%d ", Li[i]); // Row indices in the factor L
    }
    printf("\n");
  }

  // // TODO : is it a problem that the matrix A is in CSC format here and not
  // rowPtr format?
  // // is it fine? we'll see..

  // // define BSR matrix on the host
  // int mb = n; // number of block rows
  // // int nnzb already defined
  // int block_dim = 1;
  // const double alpha = 1.0; // scalar for solve?
  // // rowPtr, colPtr, values all already defined
  // T temp[n];
  // T soln[n];
  // memset(temp, 0.0, n * sizeof(T));
  // memset(soln, 0.0, n * sizeof(T));

  // // Initialize the cuda cusparse handle
  // cusparseHandle_t handle;
  // cusparseCreate(&handle);
  // cusparseStatus_t status;

  // // create device vars and copy memory over
  // double *d_values = nullptr; // NOTE : in the case of GPU assembly d_values
  // will be already on device int *d_rowPtr = nullptr; int *d_colPtr = nullptr;
  // double *d_rhs = nullptr;
  // double *d_temp = nullptr;
  // double *d_soln = nullptr;

  // cudaMalloc((void **)&d_values, nvalues * sizeof(double));
  // cudaMemcpy(d_values, values, nvalues * sizeof(double),
  // cudaMemcpyHostToDevice);

  // cudaMalloc((void **)&d_rowPtr, (n+1) * sizeof(int));
  // cudaMemcpy(d_rowPtr, rowPtr, (n+1) * sizeof(int), cudaMemcpyHostToDevice);

  // cudaMalloc((void **)&d_colPtr, nnzb * sizeof(int));
  // cudaMemcpy(d_colPtr, colPtr, nnzb * sizeof(int), cudaMemcpyHostToDevice);

  // cudaMalloc((void **)&d_rhs, n * sizeof(double));
  // cudaMemcpy(d_rhs, b, n * sizeof(double), cudaMemcpyHostToDevice);

  // cudaMalloc((void **)&d_temp, n * sizeof(double));
  // cudaMemcpy(d_temp, temp, n * sizeof(double), cudaMemcpyHostToDevice);

  // cudaMalloc((void **)&d_soln, n * sizeof(double));
  // cudaMemcpy(d_soln, soln, n * sizeof(double), cudaMemcpyHostToDevice);

  // // solve linear system with cusparse
  // cusparse_solve(
  //     handle, status, mb, nnzb, block_dim,
  //     d_values, d_rowPtr, d_colPtr,
  //     d_rhs, d_soln, d_temp
  // );

  // // copy solution back to the host
  // cudaMemcpy(soln, d_soln, n * sizeof(double), cudaMemcpyDeviceToHost);

  // // print out the solution
  // printf("soln: ");
  // printVec<double>(n, soln);

  return 0;
}