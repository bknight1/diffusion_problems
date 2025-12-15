#!/bin/bash

module load python/3.11.6 petsc/3.21.1-nocomplex py-cython/3.0.8 py-mpi4py/4.0.1-py3.11.6 py-numpy/1.25.2 py-h5py/3.12.1
source /software/projects/pawsey1147/pyvenv/uw3-test/bin/activate

# SCRIPT="/home/bknight/models/Diffusion_models/diffusion_only/Ex-Diffusion-U-Pb_test-zircon_Mesh.py"
SCRIPT="/home/bknight/models/Diffusion_models/diffusion-decay-ingrowth/zircon-TB-U-Pb_zircon_model.py"

RESULTS_BASE="${MYSCRATCH}/Diffusion_models"
NPROCS=1  # or set as needed

# Parameter sweeps
csize=0.01
U_degree=2
cfl=0.5

profiles=(2 3 4) #(1 2 3 4)
zircon_sizes=(4) #(1 2 3 4)
outputPath='./'

for profile in "${profiles[@]}"; do
  for zsize in "${zircon_sizes[@]}"; do
    RESULTS_DIR="${RESULTS_BASE}/TB-zircon_U-Pb-test"
    mkdir -p "$RESULTS_DIR"
    JOB_NAME="TB_U-Pb_test-profile=${profile}-zsize=${zsize}"
    SLURM_SCRIPT="$RESULTS_DIR/${JOB_NAME}.sh"
    cat > "$SLURM_SCRIPT" << EOF
#!/bin/bash
module load python/3.11.6 petsc/3.21.1-nocomplex py-cython/3.0.8 py-mpi4py/4.0.1-py3.11.6 py-numpy/1.25.2 py-h5py/3.12.1
source /software/projects/pawsey1147/pyvenv/uw3-test/bin/activate
export OMP_NUM_THREADS=1
cd $RESULTS_DIR
# cp $SCRIPT $RESULTS_DIR
srun python3 $SCRIPT -uw_csize $csize -uw_U_degree $U_degree -uw_CFL_fac $cfl -uw_profile $profile -uw_zircon_size $zsize
EOF
  sbatch --account=pawsey1147 --job-name=$JOB_NAME --ntasks=$NPROCS --cpus-per-task=1 --time=24:00:00 --partition=work --mem-per-cpu=20G --output=$RESULTS_DIR/${JOB_NAME}.out --error=$RESULTS_DIR/${JOB_NAME}.err "$SLURM_SCRIPT"
  sleep 30
  done
done
