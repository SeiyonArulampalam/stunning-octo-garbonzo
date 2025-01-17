#include <cholmod.h>
#include <stdio.h>

void cholmod_solve() {
  // Initialize Cholmod
  cholmod_common common;
  cholmod_start(&common);

  // Define the input matrix A in CSC format
  int n = 4;                                         // Matrix dimension
  int nzmax = 7;                                     // Number of nonzeros
  int Ap[] = {0, 3, 5, 6, 7};                        // Column pointers
  int Ai[] = {0, 1, 3, 1, 2, 2, 3};                  // Row indices
  double Ax[] = {4.0, 1.0, 1.0, 3.0, 1.0, 2.0, 5.0}; // Values (can be dummy)

  // Allocate the sparse matrix
  cholmod_sparse *A =
      cholmod_allocate_sparse(n, n, nzmax, 1, 1, 1, CHOLMOD_REAL, &common);
  A->p = Ap;
  A->i = Ai;
  A->x = Ax;
  A->stype = -1; // lower triangular

  // define the RHS as well
  size_t n = 4;
  cholmod_dense *b = cholmod_allocate_dense(n, 1, n, CHOLMOD_REAL, &common);
  double *b_data = (double *)b->x;
  for (int i = 0; i < 4; i++) {
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
    return;
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

  // now we solve the system
  cholmod_dense *x = cholmod_solve(CHOLMOD_A, L, b, &common);

  for (int i = 0; i < 4; i++) {
    printf("x[%d] = %.8e\n", i, x[i]);
  }

  // Clean up
  cholmod_free_factor(&L, &common);
  // cholmod_free_sparse(&L_pattern, &common);
  cholmod_free_sparse(&A, &common);
  cholmod_finish(&common);
}

int main() {
  cholmod_solve();
  return 0;
};
