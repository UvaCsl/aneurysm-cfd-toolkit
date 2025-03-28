# Aneurysm CFD Toolkit

This project aims to automate the mesh processing and CFD simulation setup of vascular scans of aneurysms making use of the vascular modelling toolkit [`vmtk`](http://www.vmtk.org/), [`vtk`](https://vtk.org/) and [`FEBio`](https://febio.org/).

In its current state, you can use it by simply running

```sh
conda create -n cfd-toolkit --file requirements.txt
conda activate cfd-toolkit
python model_to_sim.py -i input -o output
```

where 
- `input` is the path to a file or folder containing WRL or VTK files and 
- `output` is the path to the folder where the mesh and simulation files will be saved.

In the case of a folder, the script will process files in parallel.

Note for M1 Mac users: you may be required to use a rosetta terminal with intel miniconda due to `vmtk` lacking ARM support.