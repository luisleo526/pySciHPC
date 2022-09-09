from uvw import RectilinearGrid, DataArray
from pySciHPC.objects import Scalar
from pathlib import Path
import numpy as np
import json


class VTKPlotter:

    def __init__(self, geometry: Scalar, project_name: str):
        assert geometry.ndim > 1
        Path(f"./{project_name}_VTK").mkdir(parents=True, exist_ok=True)
        self.project_name = project_name
        self.directory = f"./{project_name}_VTK"
        self.axis = [geometry.x.cpu, geometry.y.cpu]
        if geometry.ndim > 2:
            self.axis.append(geometry.z)
        self.ndim = geometry.ndim
        self.index = -1

    def create(self):
        self.index += 1
        self.vtk = RectilinearGrid(f"{self.directory}/{self.index}.vtr", self.axis, compression=True)

    def add_scalar(self, data: np.ndarray, name: str):
        self.vtk.addPointData(DataArray(data, range(self.ndim), name))

    def add_vector(self, data: np.ndarray, name: str):
        self.vtk.addPointData(DataArray(data, range(1, self.ndim + 1), name))

    def close(self):
        self.vtk.write()
        del self.vtk

    def joins(self, dt):
        data = {}
        data["file-series-version"] = "1.0"
        data["files"] = []
        for i in range(self.index + 1):
            dat = {}
            dat["name"] = f"{self.directory}/{i}.vtr"
            dat["time"] = i * dt
            data["files"].append(dat)
        with open(f"{self.project_name}.vtk.series", "w") as f:
            json.dump(data, f, indent=4)
