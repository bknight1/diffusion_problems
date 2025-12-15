#!/bin/bash

module load python/3.11.6 petsc/3.21.1-nocomplex py-cython/3.0.8 py-mpi4py/4.0.1-py3.11.6 py-numpy/1.25.2 py-h5py/3.12.1
source /software/projects/pawsey1147/pyvenv/uw3-test/bin/activate

SCRIPT="/home/bknight/models/Diffusion_models/diffusion_only/Ex-Diffusion-benchmark_clean.py"
RESULTS_BASE="${MYSCRATCH}/Diffusion_models"
NPROCS=1  # or set as needed

# Parameter sweeps
cell_sizes=(0.001) #(0.1 0.05 0.025 0.01 0.005) # 0.0025 0.001 0.0005 0.00025 0.0001
U_degrees=(1 2 3)
CFL_facs=(0.1 0.5 1.)
temps=(850)
outputPath='./'

for temp in "${temps[@]}"; do
  for degree in "${U_degrees[@]}"; do
    for cfl in "${CFL_facs[@]}"; do
      for csize in "${cell_sizes[@]}"; do  # csize loop iterates at the innermost level
        RESULTS_DIR="${RESULTS_BASE}/diffusion_convergence"
        mkdir -p "$RESULTS_DIR"
        JOB_NAME="Diff_convergence-csize_${csize}_degree_${degree}_cfl_${cfl}_temp_${temp}"
        SLURM_SCRIPT="$RESULTS_DIR/${JOB_NAME}.sh"
        cat > "$SLURM_SCRIPT" << EOF
#!/bin/bash
module load python/3.11.6 petsc/3.21.1-nocomplex py-cython/3.0.8 py-mpi4py/4.0.1-py3.11.6 py-numpy/1.25.2 py-h5py/3.12.1
source /software/projects/pawsey1147/pyvenv/uw3-test/bin/activate
export OMP_NUM_THREADS=1
cd $RESULTS_DIR
srun python3 $SCRIPT -uw_csize $csize -uw_U_degree $degree -uw_CFL_fac $cfl -uw_Temp_start $temp -uw_Temp_end $temp
EOF
        sbatch --account=pawsey1147 --job-name=$JOB_NAME --ntasks=$NPROCS --cpus-per-task=1 --time=24:00:00 --partition=work --mem-per-cpu=20G --output=$RESULTS_DIR/${JOB_NAME}.out --error=$RESULTS_DIR/${JOB_NAME}.err "$SLURM_SCRIPT"
        sleep 60
      done
    done
  done
done

