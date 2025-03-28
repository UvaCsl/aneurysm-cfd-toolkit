from vtk import (
    vtkVRMLImporter,
    vtkRenderer, 
    vtkActor, 
    vtkMapper, 
    vtkPolyData
)
from vmtk.vmtkcenterlines import vmtkCenterlines
from vmtk.vmtksurfacewriter import vmtkSurfaceWriter
from vmtk.vmtkendpointextractor import vmtkEndpointExtractor
from vmtk.vmtkbranchclipper import vmtkBranchClipper
from vmtk.vmtksurfaceconnectivity import vmtkSurfaceConnectivity
from vmtk.vmtkflowextensions import vmtkFlowExtensions
from vmtk.vmtkflowextensions import vmtkFlowExtensions
from vmtk.vmtksurfaceremeshing import vmtkSurfaceRemeshing
from vmtk.vmtktetgen import vmtkTetGen
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import os
import logging
from copy import deepcopy

class Scan2CFD():
    def __init__(self, session_name: str, input_file: str, output_folder: str, data_folder: str = 'data', media_folder: str = 'media', reuse_state: bool = True) -> None:
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f'{__name__}_{session_name}')
        self.logger.info(f'Initializing Scan2CFD session for {session_name}')
        self.session_name = session_name
        self.input_file = input_file
        self.output_file = os.path.join(output_folder, f'{session_name}.vtu')
        self.data_folder = os.path.join(data_folder, session_name)
        self.media_folder = os.path.join(media_folder, session_name)
        self.reuse_state = reuse_state
        [os.makedirs(f, exist_ok=True) for f in [self.data_folder, self.media_folder]]
        
    def run(self) -> None:
        meshes = []
        names = []
        original_mesh = self.load_mesh_from_file()
        meshes.append(deepcopy(original_mesh))
        names.append('input')
        
        name = 'smoothed.vtp'
        if self.reuse_state and os.path.exists(os.path.join(self.data_folder, name)):
            self.logger.info(f'Loading smoothed mesh from {self.data_folder}')
            self.mesh = pv.read(os.path.join(self.data_folder, name))
        else:
            self.mesh = deepcopy(original_mesh)
            self.logger.info(f'Smoothing mesh')
            self.mesh.smooth(n_iter=500, relaxation_factor=0.05, inplace=True, progress_bar=False)
            self.mesh.save(os.path.join(self.data_folder, name))
        meshes.append(deepcopy(self.mesh))
        names.append('smoothed')

        name = 'centerlines.vtp'
        if self.reuse_state and os.path.exists(os.path.join(self.data_folder, name)):
            self.logger.info(f'Loading centerlines from {self.data_folder}')
            self.centerlines = pv.read(os.path.join(self.data_folder, name))
        else:
            self.logger.info(f'Extracting centerlines')
            cl = vmtkCenterlines()
            cl.Surface = self.mesh
            cl.Execute()
            self.centerlines = cl.Centerlines
            pv.wrap(self.centerlines).save(os.path.join(self.data_folder, name))

        self.logger.info(f'Extracting endpoints')
        endpointExtractor = vmtkEndpointExtractor()
        endpointExtractor.Centerlines = self.centerlines
        endpointExtractor.Execute()
        self.centerlines = endpointExtractor.Centerlines

        name = 'clipped.vtp'
        if self.reuse_state and os.path.exists(os.path.join(self.data_folder, name)):
            self.logger.info(f'Loading clipped mesh from {self.data_folder}')
            self.mesh = pv.read(os.path.join(self.data_folder, name))
        else:
            self.logger.info(f'Clipping branches')
            branchClipper = vmtkBranchClipper()
            branchClipper.Centerlines = self.centerlines
            branchClipper.Surface = self.mesh
            branchClipper.Execute()
            self.mesh = branchClipper.Surface
            pv.wrap(self.mesh).save(os.path.join(self.data_folder, name))
        meshes.append(deepcopy(pv.wrap(self.mesh)))
        names.append('clipped')

        name = 'connected_after_clipped.vtp'
        if self.reuse_state and os.path.exists(os.path.join(self.data_folder, name)):
            self.logger.info(f'Loading connected mesh from {self.data_folder}')
            self.mesh = pv.read(os.path.join(self.data_folder, name))
        else:
            self.logger.info(f'Connecting surface')
            surfaceConnectivity = vmtkSurfaceConnectivity()
            surfaceConnectivity.Surface = self.mesh
            surfaceConnectivity.CleanOutput = 1
            surfaceConnectivity.Execute()
            self.mesh = surfaceConnectivity.Surface
            pv.wrap(self.mesh).save(os.path.join(self.data_folder, name))

        name = 'flow_extensions.vtp'
        if self.reuse_state and os.path.exists(os.path.join(self.data_folder, name)):
            self.logger.info(f'Loading flow extensions from {self.data_folder}')
            self.mesh = pv.read(os.path.join(self.data_folder, name))
        else:
            self.logger.info(f'Adding flow extensions')
            flow_extender = vmtkFlowExtensions()
            flow_extender.Surface = self.mesh
            flow_extender.Centerlines = self.centerlines
            flow_extender.Interactive = 0
            flow_extender.ExtensionLength = 100
            flow_extender.Execute()
            self.mesh = flow_extender.Surface
            pv.wrap(self.mesh).save(os.path.join(self.data_folder, name))

        # name = 'remeshed.vtp'
        # if self.reuse_state and os.path.exists(os.path.join(self.data_folder, name)):
        #     self.logger.info(f'Loading remeshed mesh from {self.data_folder}')
        #     self.mesh = pv.read(os.path.join(self.data_folder, name))
        # else:
        #     self.logger.info(f'Remeshing surface')
        #     remesh = vmtkSurfaceRemeshing()
        #     remesh.Surface = self.mesh
        #     remesh.NumberOfIterations = 10
        #     remesh.TargetAreaFactor = 0.5
        #     remesh.Execute()
        #     self.mesh = remesh.Surface
        #     pv.wrap(self.mesh).save(os.path.join(self.data_folder, name))
        # meshes.append(deepcopy(pv.wrap(self.mesh)))
        # names.append('remeshed')

        name = 'tet_volume_mesh.vtu'
        if self.reuse_state and os.path.exists(os.path.join(self.data_folder, name)):
            self.logger.info(f'Loading volume mesh from {self.data_folder}')
            self.volume_mesh = pv.read(os.path.join(self.data_folder, name))
        else:
            self.logger.info(f'Creating volume mesh')
            tetgen = vmtkTetGen()
            tetgen.Mesh = pv.wrap(self.mesh).cast_to_unstructured_grid()
            tetgen.Execute()
            self.volume_mesh = pv.wrap(tetgen.Mesh)
            self.volume_mesh.save(os.path.join(self.data_folder, name))
        meshes.append(deepcopy(self.volume_mesh))
        names.append('tet_volume_mesh')

        self.histogram_comparison_of_meshes(meshes, names)

        self.show_mesh(self.volume_mesh)
        self.logger.info(f'Finished Scan2CFD session')

    def load_mesh_from_file(self) -> pv.PolyData:
        if self.input_file.lower().endswith('.wrl'):
            return self.load_mesh_from_wrl()
        else:
            return pv.read(self.input_file)

    def load_mesh_from_wrl(self) -> pv.PolyData:
        reader = vtkVRMLImporter()
        reader.SetFileName(self.input_file)
        reader.Read()

        renderer: vtkRenderer = reader.GetRenderer()
        actor: vtkActor = renderer.GetActors().GetLastActor()
        mapper: vtkMapper = actor.GetMapper()
        polydata: vtkPolyData = mapper.GetInput()
        mesh = pv.wrap(polydata)

        self.logger.info(f'Extracted polydata from {self.input_file} and converted into mesh')

        return mesh
    
    def histogram_comparison_of_meshes(self, meshes: list[pv.PolyData], mesh_names: list[str]) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True, sharex=True)
        for i, (mesh, name) in enumerate(zip(meshes, mesh_names)):
            mesh = mesh.compute_cell_sizes(length=True, area=True, volume=True)
            areas = mesh.get_array('Area')
            lengths = mesh.get_array('Length')
            volumes = mesh.get_array('Volume')
            if len(np.unique(lengths)) > 1:
                axes[0].hist(lengths, bins=100, alpha=0.5, label=name)
            if len(np.unique(areas)) > 1:
                axes[1].hist(areas, bins=100, alpha=0.5, label=name)
            if len(np.unique(volumes)) > 1:
                axes[2].hist(volumes, bins=100, alpha=0.5, label=name)
        for ax in axes:
            ax.set_xlabel('Value')
            ax.legend(loc='upper right')
            ax.grid(True)
            ax.set_yscale('symlog')
            ax.set_xscale('symlog')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Mesh lengths')
        axes[1].set_title(f'Mesh areas')
        axes[2].set_title(f'Mesh volumes')
        plt.tight_layout()
        plt.savefig(os.path.join(self.media_folder, f'mesh_comparison_histogram.png'))

    @staticmethod
    def show_mesh(mesh: pv.PolyData) -> None:
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color='white', label='Mesh')
        plotter.show()
        plotter.close()
