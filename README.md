# Aneurysm CFD Toolkit

This project is the result of a student research project and provides a simple but easily extendible toolchain to automate the mesh processing and CFD simulation setup of vascular scans of aneurysms. Its inputs are initial segmentations of 3D vascular scans in the form of `vtk` or `wrl` files and outputs are tetrahedral volumetric meshes in `vtk` and `vtu` format compatible with FEBio for further CFD simulation. It primarily relies on functionality from [`pyvista`](https://pyvista.org/), the vascular modelling toolkit [`vmtk`](http://www.vmtk.org/).

## Setup

You will require a python environment (preferrably `conda`) with packages installed as specified in the `requirements.txt` file. This can be done like so:

```sh
conda create -n cfd-toolkit --file requirements.txt
conda activate cfd-toolkit
```

Note for Apple Silicon (Mx) Mac users: You are required to use a rosetta terminal with an intel miniconda installation due to `vmtk` lacking ARM support. Whilst `vmtk` is not actively developed, it is a heavy dependency of this project since it contains a larger range of functionality especially designed for vascular modelling compared to other more generic meshing libraries. Instructions for such a setup can be found [here](https://archive.ph/ALAd8/again?url=https://towardsdatascience.com/how-to-install-miniconda-x86-64-apple-m1-side-by-side-on-mac-book-m1-a476936bfaf0).

## Usage

### Modelling

Simply run

```sh
python model_to_sim.py -i <input> -o <output>
```

where 
- `input` is a path to a single file or folder containing any number of WRL or VTK files and 
- `<output>` is a path to a folder where the mesh and simulation files will be saved

In case a folder is passed as input, the script will process files in parallel.

With the default toolchain, one window per file will pop up that request to define inlet and outlet centerpoints. In this step, place the cursor on the center of the inlet, press `space` and then `q` to confirm. Then select the outlets with `space` individually and press `q` to confirm. The window may not auto-close but processing will continue from there onwards. Other interactive steps for vessel clipping may be introduced but are not part of the default chain.


### CFD

After generating the volumetric mesh model, to perform CFD simulations, you must:

1. create a new FeBioStudio project
2. import the output `vtk` file from the above process as a geometry
3. optionally make any further meshing changes
4. import the `febio/blood_material.pvm` or otherwise define the fluid and apply the material to the geometry model
5. define all surfaces that require dedicated boundary conditions (inlets, outlets, vascular wall), easiest done via "Select faces" in the interface
6. define prescribed inlet and outlet velocities: if assuming normal to surface you may add a "Physics" > "Add Surface Load" of type "Fluid normal velocity", ensure you specify the vector for in and outlets correctly (mind the sign)
7. "Step" > "Add Analysis Step" and specify your duration, time stepping and solver options
8. "FeBio" > "Run FeBio" or export using "Export FE model" to simulate via cli

Some useful walkthroughs with illustrative examples can be found on the [FeBio YouTube channel](https://www.youtube.com/@FEBioVideos)

## Under the hood

### `model_to_sim.py`

The `model_to_sim.py` script is simply the interface that takes command line arguments and processed each file asynchronosly in its own thread for a maximum number of threads according to the number of processing cores the machine has available (-1 for responsivness). The files are processed through the `mesh_processing.py` `run` function.

### `mesh_processing.py`

The mesh processing logic is implemented in the `mesh_processing.py` file, notable functionality here includes:
- an abstracted `ProcessingStep` class that enables chaining processing steps from different libraries or custom functions in a compact, convenient and consistent way (e.g. storing meshes to enable comparitive visualization, writing files to cache, automatically loading properties from `vmtk` function executions, etc.)
- a `run` method that performs a predefined chain of processing steps deemed an acceptable start to convert a vascular segmentation to a volumetric mesh for CFD
- a `plot_mesh_diff_chain` method that visualizes the differences between a series of meshes. It has two modes of operation, either comparing all meshes to the first mesh (reference mesh) or comparing each mesh to its predecessor in the chain. Comparison is either done via point-wise distance assuming identical point ordering or alternatively using a `KDTree` for nearest neighbor search (akin to Hausdorff distance)
- general utility like reading wrl/vtk files and calculating mesh statistics

#### Default toolchain steps

| Step | library   | function               | purpose                                     | notes                                                                                                          |
|------|-----------|------------------------|---------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| 1    | pyvista   | Polydata.connectivity  | select (largest) connected mesh             | this removes initial undesireable fragments of unconnected mesh                                                |
| 2    | pyvista   | Polydata.clean         | removes undesirable mesh components         | merges duplicate points, removes unused points, removes degenerate cells                                       |
| 3    | pyvista   | Polydata.smooth_taubin | smoothenes surface of mesh                  | does not alter volume unlike Laplacian smoothing which may be considered a more aggressive smoothing technique |
| 4    | vmtk      | vmtkCenterlines        | calculate centerlines of vascular structure | very few alternatives to this function, it is very good at inferring the centerlines also for bifurcations     |
| 5    | vmtk      | vmtkEndpointExtractor  | extract endpoints                           | endpoints as defined by centerpoints on in and outlets                                                         |
| 6    | vmtk      | vmtkFlowExtensions     | add tubular extensions                      | helps for smooth in and outflow handling without causing undesired flow disturbance from boundary conditions   |
| 7    | pymeshfix | MeshFix.repair         | repair mesh through various steps           | very good at helping to keep mesh manifold                                                                     |
| 8    | vmtk      | vmtkSurfaceRemeshing   | remesh surface                              | scale mesh resolution and make it as uniform and stable for simulation as possible                             |



## Ownership

This project has been developed by Jonas Schäfer and based on a repository of Frank Otto [(link)](https://github.com/fwhotto/segmentation-post-processing) who experimented with a partially automated pipeline via `vmtk`.