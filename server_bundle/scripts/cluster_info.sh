#!/usr/bin/env bash
# Run once on the OSC GPU cluster login node (10.10.65.6) to learn what the bundle has to
# fit: scheduler, GPU partitions and limits, container runtime, storage and network.
# Read-only; submits nothing.
#
#   bash server_bundle/scripts/cluster_info.sh 2>&1 | tee logs/cluster_info.txt
#
# The values it prints go into the SLURM_* lines of server_bundle/config.env.
set -u

section() { printf '\n=== %s ===\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }

section "host"
echo "user=$(id -un)  host=$(hostname -f 2>/dev/null || hostname)  date=$(date -Is)"
echo "home=$HOME  pwd=$(pwd)"
grep -E '^(PRETTY_NAME)=' /etc/os-release 2>/dev/null
echo "bash=$BASH_VERSION"

section "scheduler"
for c in sbatch srun squeue sinfo qsub qstat bsub condor_submit; do
    have "$c" && echo "found: $c -> $(command -v "$c")"
done
if have sinfo; then
    sinfo --version
    echo "--- partitions (name, avail, time limit, nodes, gres, mem MB, cpus) ---"
    sinfo -o '%P %a %l %D %G %m %c' 2>/dev/null
    echo "--- nodes with GPUs ---"
    sinfo -N -o '%N %P %G %t' 2>/dev/null | grep -i gpu | sort -u | head -40
    echo "--- my associations / QOS (limits on jobs, GPUs, walltime) ---"
    have sacctmgr && sacctmgr -nP show assoc user="$(id -un)" format=account,partition,qos,maxjobs,maxsubmit,maxwall,grptres 2>/dev/null
    have sacctmgr && sacctmgr -nP show qos format=name,maxwall,maxtrespu,maxjobspu 2>/dev/null | head -20
    echo "--- defaults from slurm.conf ---"
    scontrol show config 2>/dev/null | grep -Ei '^(DefMemPer|MaxJobCount|GresTypes|SelectTypeParameters)'
fi
if have qstat && ! have sinfo; then
    qstat -Q 2>/dev/null; qstat -q 2>/dev/null
fi

section "modules"
if have module || [ -n "${MODULESHOME:-}" ]; then
    (module avail 2>&1 || true) | grep -Ei 'singularity|apptainer|cuda|python' | head -30
else
    echo "no environment modules"
fi

section "container runtime"
for c in singularity apptainer; do
    if have "$c"; then
        echo "$c: $($c --version 2>&1)"
        "$c" exec --help 2>&1 | grep -q -- '--env' && echo "  exec --env supported" || echo "  no exec --env (pre-3.6)"
        "$c" remote list 2>&1 | head -10
    fi
done
have singularity || have apptainer || echo "NO singularity/apptainer on the login node (may exist on compute nodes only; check with srun)"

section "GPU on this node"
if have nvidia-smi; then nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; else echo "no nvidia-smi (normal on a login node)"; fi

section "storage"
df -h "$HOME" 2>/dev/null
for d in /scratch /work /data /project /projects /lustre /beegfs "/scratch/$(id -un)" "/work/$(id -un)"; do
    [ -d "$d" ] && df -h "$d" 2>/dev/null | tail -1 | sed "s|^|$d: |"
done
have quota && quota -s 2>/dev/null
have lfs && lfs quota -h -u "$(id -un)" "$HOME" 2>/dev/null
echo "bundle needs ~40 GB: 14 GB archives + 15 GB extracted + ~6 GB container.sif (+ outputs)"

section "outbound network (singularity build / pip need it; runs do not)"
for url in https://cloud.sylabs.io https://huggingface.co https://pypi.org https://registry-1.docker.io; do
    code=$(curl -s -o /dev/null -m 8 -w '%{http_code}' "$url" 2>/dev/null)
    echo "$url -> ${code:-no curl}"
done

section "done"
echo "Send this output back (or paste it into the Claude session)."
