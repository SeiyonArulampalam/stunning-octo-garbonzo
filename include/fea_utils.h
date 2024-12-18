#pragma once
#include "stdlib.h"
#include <algorithm>
#include <vector>

// this class assembles k matrix for FEA into a BSR format

template <typename T, int N>
void get_fake_kelem(const T& minVal, T *kelem) {
    // kelem is an array of size N x N dense matrix
    //  but stored in N**2 array data structure
    // gen random values for it here

    for (int i = 0; i < N*N; i++) {
        kelem[i] = minVal + static_cast<double>(rand()) / RAND_MAX;
    }
}

template <typename T>
void gen_fake_vec(int N, T *rhs) {
    for (int i = 0; i < N; i++) {
        rhs[i] = static_cast<double>(rand()) / RAND_MAX;
    }
}

template <typename T, int n> // n here is nodes_per_elem
void assemble_dense_nodal_block_matrix(
    const int& nelems, const int& nnodes,
    const int *conn, T *dense) {
    // here we are just constructing the block matrix
    // nz pattern (not the true dense matrix)
    for (int ielem = 0; ielem < nelems; ielem++) {
        T kelem[n*n];
        get_fake_kelem<T,n>(1.0,kelem);

        // const int* local_node_conn = conn[ielem];
        // add into global matrix
        for (int i = 0; i < n; i++) {
            int inode = conn[ielem * n + i];
            for (int j = 0; j < n; j++) {
                int jnode = conn[ielem * n + j];
                dense[nnodes * inode + jnode] += kelem[n * i + j];
            }
        }
    }
}

template <int nodes_per_elem>
bool node_in_elem_conn(const int inode, const int *elem_conn) {
    for (int i = 0; i < nodes_per_elem; i++) {
        if (elem_conn[i] == inode) {
            return true;
        }
    }
    return false;
}

template <typename T>
void printVec(const int N, const T* vec);

template <>
void printVec<int>(const int N, const int *vec) {
    for (int i = 0; i < N; i++) {
        printf("%d,", vec[i]);
    }
    printf("\n");
}

template<>
void printVec<double>(const int N, const double *vec) {
    for (int i = 0; i < N; i++) {
        printf("%.3f,", vec[i]);
    }
    printf("\n");
}

template <typename T, int block_dim, int nodes_per_elem>
void get_row_col_ptrs(
    const int& nelems, const int& nnodes,
    const int *conn, 
    int& nnzb, // num nonzero blocks 
    int*& rowPtr, // array of len nnodes+1 for how many cols in each row
    int*& colPtr // array of len nnzb of the column indices for each block
) {

    // could launch a kernel to do this somewhat in parallel?

    nnzb = 0;
    std::vector<int> _rowPtr(nnodes+1,0);
    std::vector<int> _colPtr;

    // loop over each block row checking nz values
    for (int inode = 0; inode < nnodes; inode++) {
        std::vector<int> temp;
        for (int ielem = 0; ielem < nelems; ielem++) {
            const int *elem_conn = &conn[ielem * nodes_per_elem];
            if (node_in_elem_conn<nodes_per_elem>(inode, elem_conn)) {
                for (int in = 0; in < nodes_per_elem; in++) {
                    temp.push_back(elem_conn[in]);
                }
            }
        }

        // first sort the vector
        std::sort(temp.begin(), temp.end());

        // remove duplicates
        auto last = std::unique(temp.begin(), temp.end());
        temp.erase(last, temp.end());

        // add this to _colPtr
        _colPtr.insert(_colPtr.end(), temp.begin(), temp.end());

        // add num non zeros to nnzb for this row, also update
        nnzb += temp.size();
        _rowPtr[inode+1] = nnzb;
    }

    // copy data to output pointers (deep copy)
    rowPtr = new int[nnodes+1];
    std::copy(_rowPtr.begin(), _rowPtr.end(), rowPtr);
    colPtr = new int[nnzb];
    std::copy(_colPtr.begin(), _colPtr.end(), colPtr);
}

template <typename T, int block_dim, int nodes_per_elem>
void get_elem_ind_map(
    const int& nelems, const int& nnodes,
    const int *conn, 
    const int& nnzb, // num nonzero blocks 
    int*& rowPtr, // array of len nnodes+1 for how many cols in each row
    int*& colPtr, // array of len nnzb of the column indices for each block
    int*& elemIndMap
) {

    // determine where each global_node node of this elem
    // should be added into the ind of colPtr as map
    int nodes_per_elem2 = nodes_per_elem * nodes_per_elem;
    elemIndMap = new int[nelems * nodes_per_elem2];
    // elemIndMap is nelems x 4 x 4 array for nodes_per_elem = 4
    // shows how to add each block matrix into global
    for (int ielem = 0; ielem < nelems; ielem++) {

        // loop over each block row
        for (int n = 0; n < nodes_per_elem; n++) {
            int global_node_row = conn[nodes_per_elem * ielem + n];
            // get the block col range of colPtr for this block row
            int col_istart = rowPtr[global_node_row];
            int col_iend = rowPtr[global_node_row+1];

            // loop over each block col
            for (int m = 0; m < nodes_per_elem; m++) {
                int global_node_col = conn[nodes_per_elem * ielem + m];

                // find the matching indices in colPtr for the global_node_col of this elem_conn
                for (int i = col_istart; i < col_iend; i++) {
                    if (colPtr[i] == global_node_col) {
                        // add this component of block kelem matrix into elem_ind_map
                        int nm = nodes_per_elem * n + m;
                        elemIndMap[nodes_per_elem2 * ielem + nm] = i;
                    }
                }
            }
        }
    }
}

template <typename T, int block_dim, int nodes_per_elem, int dof_per_elem>
void assemble_sparse_bsr_matrix(
    const int& nelems, const int& nnodes,
    const int& vars_per_node,
    const int *conn, 
    int& nnzb, // num nonzero blocks 
    int*& rowPtr, // array of len nnodes+1 for how many cols in each row
    int*& colPtr, // array of len nnzb of the column indices for each block
    int*& elemIndMap, // map of rowPtr, colPtr assembly locations for kelem
    int& nvalues, // length of values array = block_dim^2 * nnzb
    T*& values
) {
    // get nnzb, rowPtr, colPtr
    get_row_col_ptrs<T,block_dim,nodes_per_elem>(
        nelems, nnodes, conn, nnzb, rowPtr, colPtr);

    // get elemIndMap to know how to add into the values array
    get_elem_ind_map<T,block_dim,nodes_per_elem>(
        nelems, nnodes, conn, nnzb, rowPtr, colPtr, elemIndMap
    );

    // create values array inside here
    // of size block_dim^2 * nnzb
    // nnzb = 1; // debug
    nvalues = block_dim*block_dim*nnzb;
    // int dof_per_elem = vars_per_node * nodes_per_elem;
    int nnz_per_block = block_dim * block_dim;
    int blocks_per_elem = nodes_per_elem * nodes_per_elem;
    values = new T[nnz_per_block*nnzb];
    

    // now add each kelem into values array as part of assembly process
    for (int ielem = 0; ielem < nelems; ielem++) {
        T kelem[dof_per_elem*dof_per_elem];
        get_fake_kelem<T,dof_per_elem>(0.0,kelem);

        // now use elemIndxMap to add into values
        for (int elem_block = 0; elem_block < blocks_per_elem; elem_block++) {
            int istart = nnz_per_block * elemIndMap[blocks_per_elem * ielem + elem_block];
            T *val = &values[istart];
            int elem_block_row = elem_block / nodes_per_elem;
            int elem_block_col = elem_block % nodes_per_elem;
            for (int inz = 0; inz < nnz_per_block; inz++) {
                int inner_row = inz / block_dim;
                int row = vars_per_node * elem_block_row + inner_row;
                int inner_col = inz % block_dim;
                int col = vars_per_node * elem_block_col + inner_col;

                val[inz] += kelem[dof_per_elem * row + col];
            }
        }
    }
}