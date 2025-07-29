// wrapper/fplll_wrapper.hpp
#ifndef FPLLL_WRAPPER_H
#define FPLLL_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

void run_bkz_on_lattice(int *flat_matrix, int dim, int block_size);

#ifdef __cplusplus
}
#endif

#endif
