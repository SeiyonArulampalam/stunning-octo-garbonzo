#include <cholmod.h>
#include <stdio.h>

void compute_fill_in_pattern() {
    // Initialize Cholmod
    cholmod_common common;
    cholmod_start(&common);

    // Define the input matrix A in CSC format
    int n = 4;  // Matrix dimension
    int nzmax = 7;  // Number of nonzeros
    int Ap[] = {0, 3, 5, 6, 7};  // Column pointers
    int Ai[] = {0, 1, 3, 1, 2, 2, 3};  // Row indices
    double Ax[] = {4.0, 1.0, 1.0, 3.0, 1.0, 2.0, 5.0};  // Values (can be dummy)

    // Allocate the sparse matrix
    cholmod_sparse *A = cholmod_allocate_sparse(n, n, nzmax, 1, 1, 1, CHOLMOD_REAL, &common);
    A->p = Ap;
    A->i = Ai;
    A->x = Ax;
    A->stype = -1; // lower triangular

    // Perform symbolic factorization
    cholmod_factor *L = cholmod_analyze(A, &common);
    
    // how expensive is the factorization step for L below?
    // I don't really want to do that.. would prefer to just
    // to cholmod_analyze, but just that doesn't seem to work
    // only need rowPtr, colPtr stuff in L not values here

    // to actually get numerical values in L
    cholmod_factorize(A, L, &common);

    // Check if factorization was successful
    if (L == NULL) {
        printf("Error: Symbolic factorization failed.\n");
        cholmod_free_sparse(&A, &common);
        cholmod_finish(&common);
        return;
    }


    // Cast the void* pointers to the appropriate types
    int *Lp = (int *)L->p;  // Column pointers in the factorization
    int *Li = (int *)L->i;  // Row indices in the factorization

    printf("L->n = %d\n", (int)L->n);
    printf("L->nzmax = %d\n", (int)L->nzmax);
    printf("L->nsuper = %d\n", (int)L->nsuper);
    //printf("Lp[0] = %d\n", (int)L->p[0]);

    // Print the symbolic fill-in pattern
    printf("Symbolic fill-in pattern (CSC format):\n");
    for (int col = 0; col < L->n; ++col) {  // Use A->ncol for the number of columns
        printf("Column %d: ", col);
        for (int i = Lp[col]; i < Lp[col + 1]; ++i) {
            printf("%d ", Li[i]);  // Row indices in the factor L
        }
        printf("\n");
    }    

    // Clean up
    cholmod_free_factor(&L, &common);
    // cholmod_free_sparse(&L_pattern, &common);
    cholmod_free_sparse(&A, &common);
    cholmod_finish(&common);
}

int main() {
    compute_fill_in_pattern();
    return 0;
};
