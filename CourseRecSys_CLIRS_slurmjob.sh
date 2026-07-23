#!/bin/bash

#SBATCH --job-name=CourseRecSys_CLIRS
#SBATCH --output=CourseRecSys_run.stdout
#SBATCH --error=CourseRecSys_run.stderr
#SBATCH --mem=44G
#SBATCH --cpus-per-task=6
#SBATCH --gpus=slice
#SBATCH --time=24:00:00
#SBATCH --mail-user=alesage@i3s.unice.fr
#SBATCH --mail-type=ALL


eval "$(mamba shell hook --shell bash)"
mamba activate myenv
python pipelines/run_pipeline.py --Config Config/run_dqn_k2_clirs.json
mamba deactivate