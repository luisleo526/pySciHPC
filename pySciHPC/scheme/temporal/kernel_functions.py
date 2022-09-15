import cupy as cp

rk3_1 = cp.ElementwiseKernel('float64 s1, float64 dt', 'float64 f',
                             'f = f + s1 * dt', 'rk3_1', no_return=True)
rk3_2 = cp.ElementwiseKernel('float64 s1, float64 s2, float64 dt', 'float64 f',
                             'f = f + (-3.0 * s1 + s2) / 4.0 * dt', 'rk3_2', no_return=True)
rk3_3 = cp.ElementwiseKernel('float64 s1, float64 s2, float64 s3, float64 dt', 'float64 f',
                             'f = f + (-s1 - s2 + 8.0 * s3) / 12.0 * dt', 'rk3_3', no_return=True)