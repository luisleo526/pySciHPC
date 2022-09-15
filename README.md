# pySciHPC - python Scientific High Performance Computing library
A high performance library for computational fluid dynamics spefices in two-phase flows with level set method
## Templates for user-defined source
### Numba(nopython jit)
```python
def source(f: numpy.ndarray, grids: numpy.ndarray, ghc: int, ndim: int, vel: numpy.ndarray, *args)
```
### CUDA
```python
def cuda_source(f: Scalar, geo: Scalar, vel: Vector, solver: CudaDerivativesSolver, s: cupy.ndarray, *args)
```

# Under devlopment
