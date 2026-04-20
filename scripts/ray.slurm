#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=15G
#SBATCH --time=00:15:00

#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -x

source .venv/bin/activate

# --- 1. Resolve nodes ---
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}

# Get the routable IP of the head node (hostname alone can resolve to loopback)
head_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')

port=6379
ip_head=$head_ip:$port
export ip_head
echo "IP Head: $ip_head"

# echo "ssh -N -L 8265:localhost:8265 $USER@$head_node" > tunnel.sh

cat > tunnel.sh <<EOF
echo "Tunneling ... starting dashboard at localhost:8265"
ssh -N -L 8265:localhost:8265 $USER@$head_node
EOF
chmod +x tunnel.sh


# --- 2. Start head ---
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head \
        --node-ip-address="$head_ip" \
        --port=$port \
        --num-cpus="${SLURM_CPUS_PER_TASK}" \
        --block &
sleep 10

# --- 3. Start workers on remaining nodes ---
worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting worker on $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address="$ip_head" \
            --num-cpus="${SLURM_CPUS_PER_TASK}" \
            --block &
    sleep 5
done

# --- 4. Hold the allocation ---
echo "Ray cluster up at $ip_head"
echo "Attach with: srun --jobid=$SLURM_JOB_ID --overlap --pty -w $head_node bash"
wait