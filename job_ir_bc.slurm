#!/bin/bash
#SBATCH -J Jupyter
#SBATCH --job-name=notebook_explore       # Название задачи
#SBATCH --output=output.log               # Выходной файл
#SBATCH --error=error.log                 # Файл ошибок
#SBATCH --partition=dgx   # <-or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --gres=gpu:1                       # Использование одной GPU
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

source ~/miniconda3/bin/activate myenv  # Aktivácia Conda prostredia

cd /home/ir739wb/ilyarekun/bc_project/notebooks/ || { echo "Ошибка: Не удалось перейти в директорию"; exit 1; }

srun jupyter-notebook --no-browser --ip=0.0.0.0