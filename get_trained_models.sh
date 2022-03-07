#!/bin/bash -l
#SBATCH --ntasks 1 # The number of cores you need...
#SBATCH --array=1-5
#SBATCH -p cosma6 #or some other partition, e.g. cosma, cosma6, etc.
#SBATCH -A dp004
#SBATCH --cpus-per-task=16
#SBATCH -J SVM-color-color #Give it something meaningful.
#SBATCH -o logs/output_job.%A_%a.out
#SBATCH -e logs/error_job.%A_%a.err
#SBATCH -t 02:00:00

# Run the job from the following directory - change this to point to your own personal space on /lustre
cd /cosma7/data/dp004/dc-rope1/Observational/colorcolorSVM/

module purge
#load the modules used to build your program.
module load python/3.9.1-C7

source ../obs-env/bin/activate

i=$(($SLURM_ARRAY_TASK_ID - 1))

# Run the program
python color-color_svm.py 0 $i 1

echo "Job done, info follows..."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,MaxRSS,Elapsed,ExitCode
exit

