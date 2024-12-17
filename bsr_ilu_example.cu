#include <assert.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include <iostream>
/*
nvcc bsr_ilu_example.cu -lcublas -lcusparse -lcudart -lcusolver && ./a.out
*/
#define CHECK_CUSPARSE(call)                                                  \
  {                                                                           \
    cusparseStatus_t err;                                                     \
    if ((err = (call)) != CUSPARSE_STATUS_SUCCESS) {                          \
      fprintf(stderr, "Got error %d at %s:%d\n", err, __FILE__, __LINE__);    \
      fprintf(stderr, " cuSPARSE error = %s\n", cusparseGetErrorString(err)); \
      cudaError_t cuda_err = cudaGetLastError();                              \
      if (cuda_err != cudaSuccess) {                                          \
        fprintf(stderr, "  CUDA error \"%s\" also detected\n",                \
                cudaGetErrorString(cuda_err));                                \
      }                                                                       \
      exit(1);                                                                \
    }                                                                         \
  }

int main() {
  // Initialize the cuda cusparse handle
  cusparseHandle_t handle;
  cusparseCreate(&handle);
  cusparseStatus_t status;

  // Define BSR matrix on the host
  int mb = 1;        // Number of block rows
  int nnzb = 1;      // Number of non-zero blocks
  int blockDim = 2;  // Dimension of a block
  const double alpha = 1.0;
  double h_bsrVal[] = {2.8, 10, 1.8, 2};
  int h_bsrRowPtr[] = {0, 1};
  int h_bsrColInd[] = {0};
  double h_x[] = {1, 2};  // RHS vector
  double h_y[] = {0, 0};  // Solution vector
  double h_z[] = {0, 0};  // Intermediate result

  // Device variables
  double *d_bsrVal = nullptr;
  int *d_bsrRowPtr = nullptr;
  int *d_bsrColInd = nullptr;
  double *d_x = nullptr;
  double *d_y = nullptr;
  double *d_z = nullptr;

  // Allocate space on the GPU for the device variables
  cudaMalloc((void **)&d_bsrVal, sizeof(h_bsrVal));
  cudaMalloc((void **)&d_bsrRowPtr, sizeof(h_bsrRowPtr));
  cudaMalloc((void **)&d_bsrColInd, sizeof(h_bsrColInd));
  cudaMalloc((void **)&d_x, sizeof(double) * 2);
  cudaMalloc((void **)&d_y, sizeof(double) * 2);
  cudaMalloc((void **)&d_z, sizeof(double) * 2);

  cudaMemcpy(d_bsrVal, h_bsrVal, sizeof(h_bsrVal), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bsrRowPtr, h_bsrRowPtr, sizeof(h_bsrRowPtr),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_bsrColInd, h_bsrColInd, sizeof(h_bsrColInd),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, h_x, 2 * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y, 2 * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_z, h_z, 2 * sizeof(double), cudaMemcpyHostToDevice);

  // Suppose that A is m x m sparse matrix represented by BSR format,
  // The number of block rows/columns is mb, and
  // the number of nonzero blocks is nnzb.
  // Assumption:
  // - handle is already created by cusparseCreate(),
  // - (d_bsrRowPtr, d_bsrColInd, d_bsrVal) is BSR of A on device memory,
  // - d_x is right hand side vector on device memory.
  // - d_y is solution vector on device memory.
  // - d_z is intermediate result on device memory.
  // - d_x, d_y and d_z are of size m.

  cusparseMatDescr_t descr_M = 0;
  cusparseMatDescr_t descr_L = 0;
  cusparseMatDescr_t descr_U = 0;
  bsrilu02Info_t info_M = 0;
  bsrsv2Info_t info_L = 0;
  bsrsv2Info_t info_U = 0;
  int pBufferSize_M;
  int pBufferSize_L;
  int pBufferSize_U;
  int pBufferSize;
  void *pBuffer = 0;
  int structural_zero;
  int numerical_zero;
  const cusparseSolvePolicy_t policy_M = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
  const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
  const cusparseSolvePolicy_t policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
  const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE;
  const cusparseOperation_t trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;
  const cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;

  // step 1: create a descriptor which contains
  // - matrix M is base-0
  // - matrix L is base-0
  // - matrix L is lower triangular
  // - matrix L has unit diagonal
  // - matrix U is base-0
  // - matrix U is upper triangular
  // - matrix U has non-unit diagonal
  cusparseCreateMatDescr(&descr_M);
  cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseCreateMatDescr(&descr_L);
  cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
  cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT);
  cusparseCreateMatDescr(&descr_U);
  cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER);
  cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT);

  // step 2: create a empty info structure
  // we need one info for bsrilu02 and two info's for bsrsv2
  cusparseCreateBsrilu02Info(&info_M);
  cusparseCreateBsrsv2Info(&info_L);
  cusparseCreateBsrsv2Info(&info_U);

  // step 3: query how much memory used in bsrilu02 and bsrsv2, and allocate the
  // buffer
  cusparseDbsrilu02_bufferSize(handle, dir, mb, nnzb, descr_M, d_bsrVal,
                               d_bsrRowPtr, d_bsrColInd, blockDim, info_M,
                               &pBufferSize_M);
  cusparseDbsrsv2_bufferSize(handle, dir, trans_L, mb, nnzb, descr_L, d_bsrVal,
                             d_bsrRowPtr, d_bsrColInd, blockDim, info_L,
                             &pBufferSize_L);
  cusparseDbsrsv2_bufferSize(handle, dir, trans_U, mb, nnzb, descr_U, d_bsrVal,
                             d_bsrRowPtr, d_bsrColInd, blockDim, info_U,
                             &pBufferSize_U);
  pBufferSize = max(pBufferSize_M, max(pBufferSize_L, pBufferSize_U));
  // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
  cudaMalloc((void **)&pBuffer, pBufferSize);

  // step 4: perform analysis of incomplete LU factorization on M
  //     perform analysis of triangular solve on L
  //     perform analysis of triangular solve on U
  //     The lower(upper) triangular part of M has the same sparsity pattern as
  //     L(U), we can do analysis of bsrilu0 and bsrsv2 simultaneously.
  //
  // Notes:
  // bsrilu02_analysis() ->
  //   Executes the 0 fill-in ILU with no pivoting
  //
  // cusparseXbsrilu02_zeroPivot() ->
  //   is a blocking call. It calls
  //   cudaDeviceSynchronize() to make sure all previous kernels are done.
  //
  // cusparseDbsrsv2_analysis() ->
  //   output is the info structure filled with information collected
  //   during he analysis phase (that should be passed to the solve phase
  //   unchanged).
  //
  // The variable "info" contains the structural zero or numerical zero

  cusparseDbsrilu02_analysis(handle, dir, mb, nnzb, descr_M, d_bsrVal,
                             d_bsrRowPtr, d_bsrColInd, blockDim, info_M,
                             policy_M, pBuffer);
  status = cusparseXbsrilu02_zeroPivot(handle, info_M, &structural_zero);
  if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
    printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
  }
  cusparseDbsrsv2_analysis(handle, dir, trans_L, mb, nnzb, descr_L, d_bsrVal,
                           d_bsrRowPtr, d_bsrColInd, blockDim, info_L, policy_L,
                           pBuffer);
  cusparseDbsrsv2_analysis(handle, dir, trans_U, mb, nnzb, descr_U, d_bsrVal,
                           d_bsrRowPtr, d_bsrColInd, blockDim, info_U, policy_U,
                           pBuffer);

  // step 5: M = L * U
  cusparseDbsrilu02(handle, dir, mb, nnzb, descr_M, d_bsrVal, d_bsrRowPtr,
                    d_bsrColInd, blockDim, info_M, policy_M, pBuffer);
  status = cusparseXbsrilu02_zeroPivot(handle, info_M, &numerical_zero);
  if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
    printf("block U(%d,%d) is not invertible\n", numerical_zero,
           numerical_zero);
  }

  // step 6: solve L*z = x
  cusparseDbsrsv2_solve(handle, dir, trans_L, mb, nnzb, &alpha, descr_L,
                        d_bsrVal, d_bsrRowPtr, d_bsrColInd, blockDim, info_L,
                        d_x, d_z, policy_L, pBuffer);

  // step 7: solve U*y = z
  cusparseDbsrsv2_solve(handle, dir, trans_U, mb, nnzb, &alpha, descr_U,
                        d_bsrVal, d_bsrRowPtr, d_bsrColInd, blockDim, info_U,
                        d_z, d_y, policy_U, pBuffer);

  cudaMemcpy(h_y, d_y, 2 * sizeof(double), cudaMemcpyDeviceToHost);
  std::cout << "Solution x = [";
  for (int i = 0; i < 2; i++) {
    std::cout << h_y[i] << " ";
  }
  std::cout << "]" << std::endl;

  // step 6: free resources
  cudaFree(pBuffer);
  cusparseDestroyMatDescr(descr_M);
  cusparseDestroyMatDescr(descr_L);
  cusparseDestroyMatDescr(descr_U);
  cusparseDestroyBsrilu02Info(info_M);
  cusparseDestroyBsrsv2Info(info_L);
  cusparseDestroyBsrsv2Info(info_U);
  cusparseDestroy(handle);

  return 0;
}