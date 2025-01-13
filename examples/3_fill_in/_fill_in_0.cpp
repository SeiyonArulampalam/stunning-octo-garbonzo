#include <cholmod.h>
/*
need SuiteSparse which can be installed from:
https://github.com/DrTimothyAldenDavis/SuiteSparse

OR can install with:
sudo apt update
sudo apt install cmake libblas-dev liblapack-dev libsuitesparse-dev libmetis-dev
*/

void compute_fill_in_pattern() {
    // Initialize Cholmod
    cholmod_common common;
    cholmod_start(&common);

    // Example: Create a symmetric sparse matrix A (4x4) in CSC format
    cholmod_sparse *A;
    int n = 4;  // Matrix dimension
    int nzmax = 7;  // Number of nonzeros
    int Ap[] = {0, 3, 5, 6, 7};  // Column pointers
    int Ai[] = {0, 1, 3, 1, 2, 2, 3};  // Row indices
    double Ax[] = {4.0, 1.0, 1.0, 3.0, 1.0, 2.0, 5.0};  // Values (can be dummy)

    A = cholmod_allocate_sparse(n, n, nzmax, 1, 1, 1, CHOLMOD_REAL, &common);
    A->stype = -1; // set lower triangular storage
    A->p = Ap;
    A->i = Ai;
    A->x = Ax;

    // Symbolic Analysis
    cholmod_factor *L = cholmod_analyze(A, &common);  // Analyze phase only

    printf("L->nzmax = %d\n", (int)L->nzmax);

    // Numerical Factorization (Required to populate L with numerical values)
    cholmod_factorize(A, L, &common);

    // Extract fill-in pattern
    cholmod_sparse *L_pattern = cholmod_factor_to_sparse(L, &common);

    // Print fill-in pattern (nonzero structure)
    int *Lp = (int *)L_pattern->p;  // Column pointers
    int *Li = (int *)L_pattern->i;  // Row indices

    printf("Fill-in pattern (CSC format):\n");


    double *Lx = (double *)L_pattern->x;
    printf("Numerical values in L:\n");
    for (int col = 0; col < L_pattern->ncol; ++col) {
        printf("Column %d: ", col);
        for (int j = Lp[col]; j < Lp[col + 1]; ++j) {
            printf("(%d, %f) ", Li[j], Lx[j]);
        }
        printf("\n");
    }
    
    
    //for (int col = 0; col < L_pattern->ncol; ++col) {
    //    printf("Column %d: ", col);
    //    for (int j = Lp[col]; j < Lp[col + 1]; ++j) {
    //        printf("%d ", Li[j]);
    //    }
    //    printf("\n");
    //}

    // Clean up
    cholmod_free_sparse(&L_pattern, &common);
    cholmod_free_factor(&L, &common);
    cholmod_free_sparse(&A, &common);
    cholmod_finish(&common);
}

int main() {
    compute_fill_in_pattern();
    return 0;
}
