import argparse
from mesh_processing import ProcessingSession
import os
from multiprocessing import Pool, cpu_count

def process_file(session_name, input_file, output_folder, show_all: bool, hide_all: bool):
    session = ProcessingSession(session_name=session_name, input_file=input_file, output_folder=output_folder, show_all=show_all, hide_all=hide_all)
    session.run()

def main():
    parser = argparse.ArgumentParser(description='Convert WRL or VTK mesh to CFD simulation')
    parser.add_argument('-i', '--input', type=str, required=True, help='Input WRL file or directory')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output directory')
    parser.add_argument('--show-all', action='store_true', help='Use this flag to show the mesh and changes at each processing step.')
    parser.add_argument('--hide-all', action='store_true', help='Use this flag to hide the mesh and changes at each processing step.')
    args = parser.parse_args()

    input_path = args.input
    output_dir = args.output

    os.makedirs(output_dir, exist_ok=True)

    if os.path.isdir(input_path):
        files = [f for f in os.listdir(input_path) if f.lower().endswith(('.wrl', '.vtk'))]
    elif os.path.isfile(input_path):
        files = [input_path]
    else:
        raise ValueError(f'Could not find file or folder at {input_path}.')
    
    # Keep one CPU core free to not freeze system for normal use
    n_procs = min(cpu_count()-1, len(files))

    if len(files) == 1:
        file_name = os.path.splitext(os.path.basename(files[0]))[0]
        session_name = file_name
        process_file(session_name=session_name, input_file=files[0], output_folder=output_dir, show_all=args.show_all, hide_all=args.hide_all)
    else:
        with Pool(processes=n_procs) as pool:
            for file in files:
                print(f'Launching process for file {file}')
                input_file = os.path.join(input_path, file) if os.path.isdir(input_path) else file
                file_name = os.path.splitext(os.path.basename(file))[0]
                session_name = file_name
                pool.apply_async(process_file, args=(session_name, input_file, output_dir, args.show_all, args.hide_all))
    
            pool.close()
            pool.join()

if __name__ == '__main__':
    main()