srun --nodes=1 --cpus-per-task=8 --mem=16G --time=1:00:00 --pty bash

ssh -N -L 8265:localhost:8265 $USER@c

uv run --no-sync --active <python file>
uv run --no-project  <python file>

uv run ray start --head --include-dashboard yes --temp-dir $(pwd)/ray_tmp
uv run ray start --address=''
