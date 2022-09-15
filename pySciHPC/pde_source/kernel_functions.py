import cupy as cp

neg_multi_sum_init = cp.ElementwiseKernel('float64 a, float64 b', 'float64 c',
                                          'c = - a * b', 'multi_sum_init', no_return=True)
neg_multi_sum = cp.ElementwiseKernel('float64 a, float64 b', 'float64 c',
                                     'c = c - a * b', 'multi_sum', no_return=True)
