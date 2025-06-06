from matplotlib import pyplot as plt
import numpy as np
import pyvista as pv
from scipy.spatial import KDTree
import os
from math import ceil, floor
import logging
from typing import Optional, Callable
from copy import deepcopy

from vmtk.vmtkcenterlines import vmtkCenterlines
from vmtk.vmtkendpointextractor import vmtkEndpointExtractor
from vmtk.vmtkflowextensions import vmtkFlowExtensions
# from vmtk.vmtksurfaceremeshing import vmtkSurfaceRemeshing
# from vmtk.vmtkbranchclipper import vmtkBranchClipper
# from vmtk.vmtksurfaceconnectivity import vmtkSurfaceConnectivity
# from vmtk.vmtksurfacedecimation import vmtkSurfaceDecimation
# from vmtk.vmtksurfacecapper import vmtkSurfaceCapper
# from vmtk.vmtkmeshgenerator import vmtkMeshGenerator
# from vmtk.vmtktetgen import vmtkTetGen

class ProcessingSession():
    def __init__(self, session_name: str, input_file: str, output_folder: str, data_folder: str = 'data_meshfix', media_folder: str = 'media', use_cache: bool = False, do_clip: bool = False, vol_mesh_backend: str = 'tetgen') -> None:
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f'{__name__}_{session_name}')
        self.logger.info(f'Initializing processing session for {session_name}')
        self.session_name = session_name
        self.input_file = input_file
        self.use_cache = use_cache
        self.do_clip = do_clip
        self.vol_mesh_backend = vol_mesh_backend
        if self.vol_mesh_backend not in ['tetgen', 'pygmsh']:
            raise ValueError('volume mesh must either use `tetgen` or `pygmsh` backend')
        if self.vol_mesh_backend == 'pygmsh':
            raise NotImplementedError('pygmsh backend is not currently implemented but should be easy to integrate.')
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        self.data_folder = os.path.join(data_folder, session_name)
        self.media_folder = os.path.join(media_folder, session_name)
        [os.makedirs(f, exist_ok=True) for f in [self.data_folder, self.media_folder, output_folder]]

        self.meshes = []
        self.names = []
        self.mesh = self.load_mesh_from_file()
        self.vol_mesh = None
        self.centerlines = None
        

    def run(self) -> None:
        steps = [
            ProcessingStep(
                func=self.mesh.connectivity,
                extraction_mode='largest',
                inplace=True, 
                progress_bar=False,
                step_name='connectivity_filtered',
                show_difference=True,
                show_output_mesh=True,
            ),
            ProcessingStep(
                func=self.mesh.clean,
                inplace=True,
                progress_bar=False,
                step_name='cleaned',
                show_output_mesh=True,
            ),
            ProcessingStep(
                func=self.mesh.decimate,
                target_reduction=0.95,
                volume_preservation=True,
                inplace=True,
                step_name='remeshed surface',
                show_difference=True,
                show_output_mesh=True,
            ),
            ProcessingStep(
                func=self.mesh.smooth_taubin,
                n_iter=20, 
                pass_band=0.1,
                feature_smoothing=False,
                boundary_smoothing=False,
                non_manifold_smoothing=True,
                inplace=True,
                progress_bar=False,
                step_name='taubin smoothed',
                show_difference=True,
            ),
            ProcessingStep(
                func=vmtkCenterlines(),
                step_name='centerlines extracted',
            ),
            # Step 7: Extract endpoints from centerlines
            ProcessingStep(
                func=vmtkEndpointExtractor(),
                step_name='endpoints extracted'
            ),
            # Step 8: Add flow extensions
            ProcessingStep(
                func=vmtkFlowExtensions(),
                Interactive=0, 
                ExtensionLength=100,
                step_name='flow extended',
                show_difference=True,
            ),
            ProcessingStep(
                func=self.fix_mesh,
                step_name='mesh fixed'
            ),
            ProcessingStep(
                func=self.tetgen_vol_mesh,
                step_name='tetgen volume mesh',
            )
        ]

        for step in steps:
            step.execute(self)

        steps_v2 = [
            ProcessingStep(
                func=self.tetgen_vol_mesh,
                step_name='tetgen volume mesh',
            )
        ]

        for step in steps_v2:
            step.execute(self)

    def fix_mesh(self):
        from pymeshfix import MeshFix
        self.logger.info(f'Running meshfix on {self.input_file}')
        mfix = MeshFix(self.mesh)
        mfix.repair(verbose=True)
        self.mesh = mfix.mesh
    

    def tetgen_vol_mesh(self):
        name = 'tetgen_volume_mesh.vtu'
        self.logger.info(f'Creating volume mesh using tetgen')
        import tetgen
        tet = tetgen.TetGen(self.mesh)
        tet.make_manifold(verbose=True)
        nodes, elements = tet.tetrahedralize()
        self.vol_mesh = tet.grid

        # TODO: store .node and .ele files in tetgen format for alternative representation akin to:
        # np.savetxt(os.path.join(self.data_folder, 'tetgen_volume_mesh.node'), nodes, fmt='%.6f', header=str(nodes.shape[0]) + ' 3 0 0', comments='')
        # np.savetxt(os.path.join(self.data_folder, 'tetgen_volume_mesh.ele'), elements + 1, fmt='%d', header=str(elements.shape[0]) + ' 4 0', comments='')

        self.vol_mesh.save(os.path.join(self.output_folder, f'{self.session_name}_tetgen_output.vtu'))
        self.vol_mesh.save(os.path.join(self.output_folder, f'{self.session_name}_tetgen_output.vtk'), binary=False) # for FEBio support
        self.meshes.append(deepcopy(self.vol_mesh))
        self.names.append('output_tetgen_volume_mesh')


    def load_mesh_from_wrl(self) -> pv.PolyData:
        from vtk import (
            vtkVRMLImporter,
            vtkRenderer,
            vtkActor,
            vtkMapper,
            vtkPolyData
        )
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

    def load_mesh_from_file(self) -> pv.PolyData:
        if self.input_file.lower().endswith('.wrl'):
            data = self.load_mesh_from_wrl()
        else:
            data = pv.read(self.input_file)
        data.save(os.path.join(self.data_folder, f'input.vtp'))
        return data
    
    
    @staticmethod
    def show_mesh(mesh: pv.PolyData) -> None:
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color='white', label='Mesh')
        plotter.show(auto_close=True)
        plotter.close()


    @staticmethod
    def plot_mesh_diff_chain(meshes: list[pv.PolyData], titles: list[str], first_mesh_reference: bool, suptitle: str = 'Mesh Differences', use_closest_point_distance_fallback: bool = True) -> None:
        """
        By default, generates a plots of passed meshes and applies colormaps to visualize changes between subsequent meshes in order. If first_mesh_reference is set, comparisons are assessed relative to the first mesh of the list.
        Args:
            meshes (list[pv.PolyData]): List of meshes to compare to each other.
            titles (list[str]): Titles for each mesh.
            first_mesh_reference (bool): Whether the first mesh in the meshes list shall be used as a reference such that all other meshes are compared to it instead of their predecessor
            use_closest_point_distance (bool): If True, compute the distance from each point to the closest point on the reference mesh as fallback.
        """
        n = len(meshes)
        if n != len(titles):
            raise ValueError("Number of meshes and titles must match.")
        if n < 1:
            raise ValueError("At least one mesh is required for comparison with the reference.")

        differences = []
        differences.append(np.zeros(len(meshes[0].points)))
        for i in range(n-1):
            mesh = meshes[i+1]
            reference_mesh = meshes[0] if first_mesh_reference else meshes[i]
            if mesh.points.shape[0] == reference_mesh.points.shape[0]: # assumes points are in the same order
                diff = np.linalg.norm(reference_mesh.points - mesh.points, axis=1)
                differences.append(diff)
            elif use_closest_point_distance_fallback:
                tree = KDTree(reference_mesh.points)
                distances, indices = tree.query(mesh.points)
                differences.append(distances)
            else:
                raise Exception('Meshes cannot be compared properly with the provided settings. Consider allowing closest point distance as fallback.')

        # Create a plotter
        x_max = 4
        y = ceil(n/x_max)
        plotter = pv.Plotter(shape=(y, min(x_max, n)), title=suptitle)

        for i, (mesh, title) in enumerate(zip(meshes, titles)):
            plotter.subplot(floor(i/x_max), i%x_max)
            plotter.add_mesh(mesh, scalars=differences[i], show_scalar_bar=True, cmap='coolwarm', label=f'mesh ({i})')
            plotter.add_title(title)
            plotter.show_bounds()

        # Show the plot
        plotter.link_views()
        plotter.show(auto_close=True)
        # plotter.save_graphic()
        plotter.close()


    def _update_statistics(self):
        data = self.mesh.compute_cell_sizes(length=True, area=True)
        areas = data.get_array('Area')
        
        self.median_area = np.median(areas)
        self.mean_area = np.mean(areas)
        self.min_area = np.min(areas)
        self.max_area = np.max(areas)
                

class ProcessingStep:
    n_step: int = 0

    def __init__(self, func, step_name: str, file_name: Optional[str] = None, show_difference: bool = False, show_output_mesh: bool = False, save: bool = True, **kwargs):
        ProcessingStep.n_step += 1  # Increment the shared class variable
        self.func = func
        self.kwargs = kwargs  # Store the keyword arguments directly
        self.file_name = file_name # without suffix
        self.step_name = f'{ProcessingStep.n_step}_{step_name}'
        if not file_name:
            self.file_name = f'{self.step_name.lower().replace(" ", "_")}'
        self.show_difference = show_difference
        self.show_output_mesh = show_output_mesh
        self.save = save
        
    def execute(self, s2cfd: ProcessingSession):
        before_mesh = deepcopy(s2cfd.mesh)
        s2cfd.logger.info(f'Performing processing step: {self.step_name}')
        if callable(self.func): # for PyVista or other directly callable
            self.func(**self.kwargs)
        else: # for vmtk
            if hasattr(self.func, 'Surface'):
                self.func.Surface = s2cfd.mesh
            if hasattr(self.func, 'Mesh'):
                self.func.Mesh = s2cfd.mesh.cast_to_unstructured_grid()
            if hasattr(self.func, 'Centerlines') and s2cfd.centerlines:
                self.func.Centerlines = s2cfd.centerlines

            for key, value in self.kwargs.items():
                setattr(self.func, key, value)
            self.func.Execute()

            if hasattr(self.func, 'Surface'):
                s2cfd.mesh = pv.wrap(self.func.Surface)
            if hasattr(self.func, 'Mesh'):
                s2cfd.vol_mesh = pv.wrap(self.func.Mesh)
            if hasattr(self.func, 'Centerlines'):
                s2cfd.centerlines = pv.wrap(self.func.Centerlines)
        s2cfd.logger.info(f'Finished processing step: {self.step_name}')
        if self.save:
            s2cfd.mesh.save(os.path.join(s2cfd.data_folder, f'{self.file_name}.vtp'))
            s2cfd.mesh.save(os.path.join(s2cfd.data_folder, f'{self.file_name}.stl'))
            if s2cfd.vol_mesh:
                s2cfd.vol_mesh.save(os.path.join(s2cfd.data_folder, f'{self.file_name}.vtu'))

        s2cfd.meshes.append(deepcopy(s2cfd.mesh))
        s2cfd.names.append(self.step_name)
        if self.show_difference:
            s2cfd.plot_mesh_diff_chain(meshes=[before_mesh, s2cfd.mesh], titles=['before', f'after'], first_mesh_reference=True, suptitle=self.step_name)
        if self.show_output_mesh:
            s2cfd.show_mesh(s2cfd.mesh)
        s2cfd._update_statistics()