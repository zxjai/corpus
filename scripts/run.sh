mkdir -p slurm_logs

TS=$(date +%Y%m%d-%H%M%S)

sbatch --output="slurm_logs/${TS}-%x-%j.out" \
       --error="slurm_logs/${TS}-%x-%j.err" \
        scripts/ray.slurm