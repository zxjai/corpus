srun --nodes=1 --cpus-per-task=8 --mem=30G --time=3:00:00 --pty bash

srun --gres=gpu:h100:1 --cpus-per-task=8 --mem=80G --time=2:00:00 --pty bash

uv sync --all-extras 


uv run --no-sync --active <python file>
uv run --no-project  <python file>

uv run ray start --head --include-dashboard yes --temp-dir $(pwd)/ray_tmp
uv run ray start --address=''


ssh -N -L 8265:localhost:8265 $USER@c123
ssh c123 "cd $PWD && source .venv/bin/activate && python corpus/backends/cluster.py"

# Check running GPU jobs 
squeue -t R -o "%.12i %.20a %.40b %.5D %.15N %.2t %.11L %.4C %.6P %.20e %.20S" | \
    awk 'NR==1 || $2 ~ /^(rrg|def)-.*_gpu$/'

# check pending GPU jobs 
squeue -t PD -o "%.12i %.20a %.40b %.5D %.2t %.11L %.4C %.6P %.15r %.20S" | \
    awk 'NR==1 || $2 ~ /^(rrg|def)-.*_gpu$/' > all_pending_jobs.log