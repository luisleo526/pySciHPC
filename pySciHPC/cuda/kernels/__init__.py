import cupy as cp

assign_zero = cp.ElementwiseKernel('', 'float64 x', 'x=0.0', 'assign_zero', no_return=True)
neg_multi_sum_init = cp.ElementwiseKernel('float64 a, float64 b', 'float64 c',
                                          'c = - a * b', 'multi_sum_init', no_return=True)
neg_multi_sum = cp.ElementwiseKernel('float64 a, float64 b', 'float64 c',
                                     'c = c - a * b', 'multi_sum', no_return=True)

rk3_1 = cp.ElementwiseKernel('float64 s1, float64 dt', 'float64 f',
                             'f = f + s1 * dt', 'rk3_1', no_return=True)
rk3_2 = cp.ElementwiseKernel('float64 s1, float64 s2, float64 dt', 'float64 f',
                             'f = f + (-3.0 * s1 + s2) / 4.0 * dt', 'rk3_2', no_return=True)
rk3_3 = cp.ElementwiseKernel('float64 s1, float64 s2, float64 s3, float64 dt', 'float64 f',
                             'f = f + (-s1 - s2 + 8.0 * s3) / 12.0 * dt', 'rk3_3', no_return=True)