// assume we applied fill-in to the original rowptr, colptr now from suitesparse
// can we now solve with cusparse correctly?

#include "cusparse_solve_utils.h"
#include "fea_utils.h"

int main() {
    using T = double;

    size_t n = 4;                                         // Matrix dimension
    int nvalues = 12;                                     // Number of nonzeros
    int nnzb = 12;
    int Ap[] = {0, 3, 7, 10, 12};                        // Column pointers
    int Ai[] = {0, 1, 3, 0, 1, 2, 3, 1, 2, 3, 1, 3};                  // Row indices
    double Ax[] = {3, 1, 1, 1, 2, -1, 0, -1, 4, 0, 1, 3}; // Values (can be dummy)

    // I probably need to add the fill-in to the upper and lower parts? or just lower triangular part
    // I think just lower triangular part is fine.. think about this real quick (let's just try it and see what hapens)

    double b[] = {1, 2, 3, 4};

    // TODO : is it a problem that the matrix A is in CSC format here and not rowPtr format?
    // is it fine? we'll see..

    // define BSR matrix on the host
    int mb = 4; // number of block rows
    // int nnzb already defined
    int block_dim = 1;
    const double alpha = 1.0; // scalar for solve?
    // rowPtr, colPtr, values all already defined
    T temp[4];
    T soln[4];
    memset(temp, 0.0, 4 * sizeof(T));
    memset(soln, 0.0, 4 * sizeof(T));

    // create device vars and copy memory over
    double *d_values = nullptr; // NOTE : in the case of GPU assembly d_values will be already on device
    int *d_rowPtr = nullptr;
    int *d_colPtr = nullptr;
    double *d_rhs = nullptr;
    double *d_temp = nullptr;
    double *d_soln = nullptr;

    cudaMalloc((void **)&d_values, nvalues * sizeof(double));
    cudaMemcpy(d_values, Ax, nvalues * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_rowPtr, 5 * sizeof(int));
    cudaMemcpy(d_rowPtr, Ap, 5 * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_colPtr, nnzb * sizeof(int));
    cudaMemcpy(d_colPtr, Ai, nnzb * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_rhs, 4 * sizeof(double));
    cudaMemcpy(d_rhs, b, 4 * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_temp, 4 * sizeof(double));
    cudaMemcpy(d_temp, temp, 4 * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_soln, 4 * sizeof(double));
    cudaMemcpy(d_soln, soln, 4 * sizeof(double), cudaMemcpyHostToDevice);

    // solve linear system with cusparse
    cusparse_solve(
        handle, status, mb, nnzb, block_dim,
        d_values, d_rowPtr, d_colPtr,
        d_rhs, d_soln, d_temp
    );

    // copy solution back to the host
    cudaMemcpy(soln, d_soln, 4 * sizeof(double), cudaMemcpyDeviceToHost);

    // print out the solution
    printf("soln: ");
    printVec<double>(4, soln);

}