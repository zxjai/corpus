srun --cpus-per-task=32 --mem=80G --time=02:00:00 --pty bash
cd $SLURM_TMPDIR
export APPTAINER_CACHEDIR=$PWD/.apptainer_cache
export APPTAINER_TMPDIR=$PWD/.apptainer_tmp
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR
module load apptainer
apptainer pull nemo_26.04.sif docker://nvcr.io/nvidia/nemo:26.04

apptainer pull rocm_vllm_latest.sif docker://vllm/vllm-openai-rocm:latest