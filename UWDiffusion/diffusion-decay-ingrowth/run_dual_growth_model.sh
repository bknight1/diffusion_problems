#!/bin/bash
# Simple runner to submit a single SLURM job that runs
# Ex-Diffusion-decay-ingrowth-U-Pb_test-dualGrowth.py with the requested options.
# Usage: ./run_ex_diffusion_dualgrowth.sh
#
# Sets:
#   CFL_fac = 1
#   csize   = 0.01
#   U_degree= 2
#   T_initial = 800
#   T_pulse   = 850
#   T_base    = 750

# Load modules and activate environment
module load python/3.11.6 petsc/3.21.1-nocomplex py-cython/3.0.8 py-mpi4py/4.0.1-py3.11.6 py-numpy/1.25.2 py-h5py/3.12.1
source /software/projects/pawsey1147/pyvenv/uw3-test/bin/activate

# Path to the python script to run
SCRIPT="/home/bknight/models/Diffusion_models/diffusion-decay-ingrowth/zircon-U-Pb_test-dualGrowth.py"

# Results directory (adjust MYSCRATCH if necessary)
RESULTS_BASE="${MYSCRATCH}/Diffusion_models"
RESULTS_DIR="${RESULTS_BASE}/dualGrowth_run"
mkdir -p "$RESULTS_DIR"

# Job parameters (change as needed)
NPROCS=1
CPUS=1
TIME="12:00:00"
PARTITION="work"
MEM_PER_CPU="20G"
ACCOUNT="pawsey1147"

# Model options requested by user
CFL_fac=1
csize=0.01
U_degree=2
T_initial=800
T_pulse=850
T_base=750

JOB_NAME="ExDiff_dualgrowth_csize${csize}_Udeg${U_degree}_CFL${CFL_fac}"

SLURM_SCRIPT="$RESULTS_DIR/${JOB_NAME}.sh"

cat > "$SLURM_SCRIPT" << EOF
#!/bin/bash
#SBATCH --account=${ACCOUNT}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --ntasks=${NPROCS}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --time=${TIME}
#SBATCH --partition=${PARTITION}
#SBATCH --mem-per-cpu=${MEM_PER_CPU}
#SBATCH --output=${RESULTS_DIR}/${JOB_NAME}.out
#SBATCH --error=${RESULTS_DIR}/${JOB_NAME}.err

module load python/3.11.6 petsc/3.21.1-nocomplex py-cython/3.0.8 py-mpi4py/4.0.1-py3.11.6 py-numpy/1.25.2 py-h5py/3.12.1
source /software/projects/pawsey1147/pyvenv/uw3-test/bin/activate
export OMP_NUM_THREADS=1

cd ${RESULTS_DIR}

# Copy the script for provenance (optional)
cp ${SCRIPT} ${RESULTS_DIR}/

# Run the model. Pass both -uw_ style flags (used in previous scripts)
# and long-form flags (in case the script parses those).
srun python3 ${SCRIPT} \
  -uw_CFL_fac ${CFL_fac} -uw_csize ${csize} -uw_U_degree ${U_degree} \
  -uw_Temp_initial ${T_initial} -uw_Temp_pulse ${T_pulse} -uw_Temp_base ${T_base} 
EOF

# Submit the job
sbatch "$SLURM_SCRIPT"
# echo "Submitted ${JOB_NAME} (SLURM script: ${SLURM_SCRIPT})"
