#!/bin/bash

#SBATCH --job-name=clirs_ppo_k2
#SBATCH --output=out/clirs_ppo_k2.stdout
#SBATCH --error=out/clirs_ppo_k2.stderr
#SBATCH --mem=44G
#SBATCH --cpus-per-task=6
#SBATCH --gpus=slice
#SBATCH --time=16:00:00
#SBATCH --mail-user=alesage@i3s.unice.fr
#SBATCH --mail-type=ALL

eval "$(mamba shell hook --shell bash)"
mamba activate myenv
python pipelines/run_pipeline.py --Config Config/run_ppo_k2_clirs.json
mamba deactivate