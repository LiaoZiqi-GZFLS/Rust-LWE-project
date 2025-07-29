// bkz_wrapper.cpp
#include <fplll.h>
#include <iostream>

extern "C" {
    void run_bkz_on_lattice(int *flat_matrix, int dim, int block_size)
    {
        using namespace fplll;

        /* 1. 把 int 数组读入 IntMatrix */
        IntMatrix A(dim, dim);
        for (int i = 0; i < dim; ++i)
            for (int j = 0; j < dim; ++j)
                A[i][j] = flat_matrix[i * dim + j];

        /* 2. LLL 预规约（高层接口） */
        lll_reduction(A, LLL_DEF_DELTA, LLL_DEF_ETA, LM_WRAPPER);

        /* 3. 构造 BKZ 参数（空策略） */
        BKZParam param;
        param.block_size = block_size;
        param.strategies = BKZParam::empty_strategy(block_size);  // fplll ≥5.4
        param.delta      = LLL_DEF_DELTA;
        param.eta        = LLL_DEF_ETA;
        param.flags      = BKZ_DEFAULT;

        /* 4. 执行 BKZ */
        int ret = bkz_reduction(A, param, FT_DEFAULT, 0);
        if (ret != RED_SUCCESS)
            std::cerr << "BKZ failed, code = " << ret << std::endl;

        /* 5. 把结果写回 int 数组（溢出风险自担） */
        for (int i = 0; i < dim; ++i)
            for (int j = 0; j < dim; ++j)
                flat_matrix[i * dim + j] = A[i][j].get_si();
    }
}