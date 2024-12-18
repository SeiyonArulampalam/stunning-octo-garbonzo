#include "cusparse_solve_utils.h"
#include "fea_utils.h"

int main() {
    using T = double;

    // define FEA problem
    constexpr int nelems = 3;
    constexpr int nnodes = 8; // global num elements
    constexpr int vars_per_node = 6; // e.g. for shell
    constexpr int nodes_per_elem = 4;
    constexpr int dof_per_elem = vars_per_node * nodes_per_elem;
    constexpr int blocks_per_elem = nodes_per_elem * nodes_per_elem; // 4x4 block matrix for each element
    constexpr int nnz_per_block = vars_per_node * vars_per_node; // 6x6 nonzero dense entries per block
    constexpr int nglobal = nnodes * vars_per_node;
    constexpr int num_dof = nglobal * nglobal;
    int conn[nelems][nodes_per_elem] = { {0,1,2,3}, {2,3,4,5}, {4,5,6,7} };

    // generate dense block matrix to check matches nz python examples/1_python_sparse_assembly
    T kblock[nnodes*nnodes];
    assemble_dense_nodal_block_matrix<T,nodes_per_elem>(
        nelems, nnodes, &conn[0][0], kblock);
    printf("dense block nodal matrix:\n");
    for (int i = 0 ; i < nnodes; i++) {
        for (int j = 0; j < nnodes; j++) {
            printf("%.2f ", kblock[nnodes * i + j]);
        }
        printf("\n");
    }


    // generate sparse k matrix in BSR format
    printf("\n\n");
    int nnzb, *rowPtr, *colPtr, *elemIndMap, nvalues;
    T *values;
    assemble_sparse_bsr_matrix<T,vars_per_node,nodes_per_elem, dof_per_elem>(
        nelems, nnodes, vars_per_node, &conn[0][0],
        nnzb, rowPtr, colPtr, elemIndMap, nvalues, values
    );

    printf("RowPtr: ");
    printVec<int>(nnodes+1,rowPtr);

    printf("ColPtr: ");
    printVec<int>(nnzb,colPtr);

    printf("elemIndMap[ielem=0]: ");
    printVec<int>(16, &elemIndMap[0]);
    printf("elemIndMap[ielem=1]: ");
    printVec<int>(16, &elemIndMap[16]);
    printf("elemIndMap[ielem=2]: ");
    printVec<int>(16, &elemIndMap[32]);
    // verified that these rowPtr, colPtr, elemIndMap give the right answer (check)

    // test print values
    printf("Values at overlap block (2,3) or block 10: ");
    printVec<T>(nnz_per_block,&values[36*10]);
    // some vals over 1 here so overlap works! indicates values may be right..
    // printVec<T>(nnz_per_block * nnzb, values);

    // gen random global rhs
    T rhs[nglobal];
    gen_fake_vec<T>(nglobal,rhs);

    // Initialize the cuda cusparse handle
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    cusparseStatus_t status;

    // define BSR matrix on the host
    int mb = nnodes; // number of block rows
    // int nnzb already defined
    int block_dim = vars_per_node;
    const double alpha = 1.0; // scalar for solve?
    // rowPtr, colPtr, values all already defined
    T temp[nglobal];
    T soln[nglobal];
    memset(temp, 0.0, nglobal * sizeof(T));
    memset(soln, 0.0, nglobal * sizeof(T));

    // create device vars and copy memory over
    double *d_values = nullptr; // NOTE : in the case of GPU assembly d_values will be already on device
    int *d_rowPtr = nullptr;
    int *d_colPtr = nullptr;
    double *d_rhs = nullptr;
    double *d_temp = nullptr;
    double *d_soln = nullptr;

    cudaMalloc((void **)&d_values, nvalues * sizeof(double));
    cudaMemcpy(d_values, values, nvalues * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_rowPtr, (nnodes+1) * sizeof(int));
    cudaMemcpy(d_rowPtr, rowPtr, (nnodes+1) * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_colPtr, nnzb * sizeof(int));
    cudaMemcpy(d_colPtr, colPtr, nnzb * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_rhs, nglobal * sizeof(double));
    cudaMemcpy(d_rhs, rhs, nglobal * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_temp, nglobal * sizeof(double));
    cudaMemcpy(d_temp, temp, nglobal * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_soln, nglobal * sizeof(double));
    cudaMemcpy(d_soln, soln, nglobal * sizeof(double), cudaMemcpyHostToDevice);

    // solve linear system with cusparse
    cusparse_solve(
        handle, status, mb, nnzb, block_dim,
        d_values, d_rowPtr, d_colPtr,
        d_rhs, d_soln, d_temp
    );

    // copy solution back to the host
    cudaMemcpy(soln, d_soln, nglobal * sizeof(double), cudaMemcpyDeviceToHost);

    // print out the solution
    printf("soln: ");
    printVec<double>(nglobal, soln);

    return 0;
};