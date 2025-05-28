#!/bin/bash
#SBATCH -J Jupyter
#SBATCH --job-name=smallres     
#SBATCH --output=output.log              
#SBATCH --error=error.log                 
#SBATCH --partition=dgx   
#SBATCH --gres=gpu:2                       
#SBATCH --time=100:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

source ~/miniconda3/bin/activate myenv  
cd /home/ir739wb/ilyarekun/bc_project/
srun jupyter-notebook --no-browser --ip=0.0.0.0