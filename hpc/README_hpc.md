# DICC HPC Helpers

## Purpose

This folder contains DICC Slurm helper scripts for the final experiment.

## Manual environment activation

```bash
module load miniconda/24.11.1
conda activate lastmile
```

## Input check command

```bash
cd /home/user/chenziliang/dissertation/amazon-last-mile-sequence-optimization
python scripts/00_check_final_inputs.py --config config/config_dicc.yaml
```

## Submit a 20-route Slurm smoke test

```bash
sbatch --export=ALL,MAX_ROUTES_PER_SPLIT=20,JOB_SUFFIX=smoke_20 hpc/run_full_pairwise.sh
```

## Submit the full pairwise job

```bash
sbatch hpc/run_full_pairwise.sh
```

## Check job status

```bash
squeue -u chenziliang
```
