import pickle

from ..core.data import Scalar


def command_extractor(conditions: list, depth=0, role: str = 'f[i,j]', value: str = 1.0) -> str:
    pos = "\t" * depth
    commands = ""
    for i, msg in enumerate(conditions, 1):
        if i == 1:
            assert type(msg) == str
            commands += f"{pos}if {msg}:\n"
            if len(conditions) > i + 1 and type(conditions[i + 1]) != list or len(conditions) == i:
                commands += f"\t{pos}{role} += {value}\n"
        else:
            if type(msg) == list:
                commands += command_extractor(msg, depth + 1, role, value)
            else:
                commands += f"{pos}elif {msg}:\n"
                if len(conditions) > i + 1 and type(conditions[i + 1]) != list or len(conditions) == i:
                    commands += f"\t{pos}{role} += {value}\n"

    return commands


def as_density_2d(f: Scalar, geo: Scalar, conditions: list, resolution: int = 30):
    commands = command_extractor(conditions, role="f[i,j]", value="1.0/resolution**2", depth=5)
    print(f"Your initializer:\n {commands}")

    func = """
from numba import float64, njit, prange, int32
import numpy as np
import pickle
@njit(float64[:,:](float64[:,:], float64[:], float64[:], float64, float64, int32))
def init(f:np.ndarray, x_grids:np.ndarray, y_grids:np.ndarray, dx:float, dy:float, resolution:int):
    assert f.shape[0] == x_grids.size - 1
    assert f.shape[1] == y_grids.size - 1
    for i in prange(f.shape[0]):
        for j in prange(f.shape[1]):
        
            xc = x_grids[i]
            yc = y_grids[j]

            for ii in range(resolution):
                for jj in range(resolution):
                    x = xc + ii * dx / resolution
                    y = yc + jj * dy / resolution
%s
            f[i, j] = 2.0 * f[i, j] - 1.0
    return f
with open('./init.pkl','wb') as file:
    pickle.dump(init(f_for_init, x_for_init, y_for_init, dx_for_init, dy_for_init, res_for_init), file)
    """ % commands.expandtabs(4)

    codeObject = compile(func, 'init_as_density2D', 'exec')
    exec(codeObject, {'f_for_init': f.core, 'x_for_init': geo.x.cpu, 'y_for_init': geo.y.cpu,
                      'dx_for_init': geo.dx, 'dy_for_init': geo.dy, 'res_for_init': resolution})

    with open('init.pkl', 'rb') as file:
        f.core = pickle.load(file)
