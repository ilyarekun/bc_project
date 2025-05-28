# BASIC PRINCIPLES OF FEDERATED LEARNING USING NEURAL NETWORKS

This is a documentation for bachelor thesis with centralized and federated learning implementations for classifying brain tumors using MRI images. The centralized learning approach is implemented with separate data preprocessing and model files, while the federated learning experiments explore algorithms such as FedAvg, FedMA, FedProx, and Scaffold, with results for both IID and non-IID data distributions.

## Project Structure
.  
├── README.md  
├── centralized-learning  
│   ├── notebooks  
│   │   └── exploration.ipynb  
│   └── src  
│       ├── data_preprocessing.py  
│       └── model.py  
├── error.log  
├── federated-learning  
│   ├── fed_learning_notebook-fedavg.ipynb  
│   ├── fed_learning_notebook-fedma.ipynb  
│   ├── fed_learning_notebook-fedprox.ipynb  
│   ├── fed_learning_notebook-scaffold.ipynb  
│   └── outputs  
├── irjob2.slurm  
├── output.log  
└── requirements.txt  

## Directory and File Descriptions

centralized-learning: Directory containing the centralized learning experiment.  

notebooks/exploration.ipynb: Jupyter notebook for running the centralized learning experiment, integrating data preprocessing and model training.  
src/data_preprocessing.py: Script for loading and preprocessing MRI image data for the centralized learning experiment.  
src/model.py: Defines the machine learning model used in the centralized learning experiment.


federated-learning: Directory containing federated learning experiments.  

fed_learning_notebook-fedavg.ipynb: Notebook for the Federated Averaging (FedAvg) experiment.  
fed_learning_notebook-fedma.ipynb: Notebook for the Federated Matching (FedMA) experiment.  
fed_learning_notebook-fedprox.ipynb: Notebook for the Federated Proximal (FedProx) experiment.  
fed_learning_notebook-scaffold.ipynb: Notebook for the Scaffold experiment.  
outputs: Directory storing output files (e.g., graphs and metrics) from the federated learning experiments.  


error.log: Log file capturing errors encountered during the experiments.

irjob2.slurm: Slurm script for running the experiments on a server with Slurm workload manager.

output.log: Log file containing output from the experiments.

requirements.txt: List of Python dependencies required for the project.


## Installation and Running Instructions
Prerequisites

Python Version: Python 3.12.9 (tested and recommended).
Server Environment: This project was run on a server with Slurm workload manager.

Setup Steps

Clone the Repository:
 - git clone https://github.com/ilyarekun/bc_project.git
 - cd bc_project


Create and Activate a Conda Environment:
 - conda create --name myenv python=3.12.9
 - conda activate myenv


Install Dependencies:
 - pip install -r requirements.txt


Run Experiments Using Slurm:

Submit the Slurm job:
 - sbatch irjob2.slurm


Check the error.log file for a link to the Jupyter server.


Access and Run Jupyter Notebooks:

Open VS Code.  
Go to "Kernel" > "Select Another Kernel" > "Existing Jupyter Server".  
Copy the link from error.log and paste it into the prompt.  
Select the created Conda environment (myenv).  
Open and run the desired notebook from the federated-learning or centralized-learning/notebooks directory.  



## Notes

This project was designed and executed on a server with Slurm. If running locally, you may need to modify the Slurm script or execute the notebooks directly.
Ensure the Conda environment (myenv) is activated before running any commands.
The outputs directory in federated-learning contains graphs and metrics files, providing insights into the performance of each federated learning algorithm under IID and non-IID conditions.

