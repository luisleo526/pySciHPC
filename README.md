# pySciHPC - python Scientific High Performance Computing library
## Templates for user-defined source
### Numba(nopython jit)
```python
def source(f: numpy.ndarray, grids: numpy.ndarray, ghc: int, ndim: int, vel: numpy.ndarray, *args)
```
### CUDA
```python
def cuda_source(f: Scalar, geo: Scalar, vel: Vector, solver: CudaDerivativesSolver, s: cupy.ndarray, *args)
```
