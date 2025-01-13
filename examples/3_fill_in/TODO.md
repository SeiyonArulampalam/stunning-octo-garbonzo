# TODO : get used to fill-in

- [ ] do some demo fill-in examples
- [ ] apply fill-in to a BSR / CSR format matrix in python (w/ row and col ptrs)
    - [ ] also add zero values when we add new entries
- [ ] use filled in matrix to do complete Cholesky and solve
    - [ ] choose small matrix BSR with maybe block size of 1x1 first for easy debug problem
    - [ ] check if solves matrix equation exactly now
- [ ] how to do fill-in for BSR if block size > 1 (can we just fill-in whole block) => do it for each block
- [ ] try routines like SuiteSparse which compute fill-in and postordering on CPU
    - [ ] should we use this? can we use BSR format and it basically fills-in whole block?
    - [ ] or do we need to convert to CSR first or something?
- [ ] try also cusparse solve (on GPU) after suitesparse fill-in (on CPU)