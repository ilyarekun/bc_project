

# Diploma Thesis  
  
### Basic Principles of Federated Learning Using Convolutional Neural Networks

the name of the thesis should be a short version for those questions

#### **Main Question:**  
**How does the choice of aggregation method (FedAvg, FedProx, FedMA) affect the model performance in federated learning compared to the centralized approach?**  

#### **Additional Questions:**  
1. How does federated learning perform compared to centralized learning on KPE tasks under different data distribution scenarios (IID, Non-IID)?  
2. Which aggregation methods are most robust to data heterogeneity (e.g. FedProx vs. FedAvg)?  




## Thesis preparation instructions:

### Structure 

1. Provide an [Introduction](#1-introduction) describing Federated Learning, its benefits over Centralized Learning, and key challenges. 
2. Provide [the State of the Art](#2-the-state-of-the-art) and Applications by reviewing recent Advancements, key Applications, and ongoing Research Challenges in Federated Learning.
3. Provide a [Comparison of Aggregation Methods](#3-comparison-of-aggregation-methods) by analyzing FedAvg, FedProx, and FedMA in terms of Efficiency, Robustness, and handling of Heterogeneous Data. 
4. Provide an [Experimental Setup](#4-experimental-setup) that describes Datasets, and Methodology for comparing Federated Learning Methods with Centralized Learning of Neural Networks
5. Provide [Results and Analysis](#5-results-and-analysis) by presenting Performance Metrics, comparing Methods, and suggesting Best Practices for Federated Learning Aggregation. 
6. Follow the Guidance of the Thesis Supervisor to ensure Research aligns with Supervisory Recommendations.

---
---

## 1. Introduction 
*[Structure](#structure)*

will be written in the end


---
---

## 2. The State of the art
*[Structure](#structure)*
how many articles should be cyted?
tell about articles in the field of FL


---
---

## 3. Comparison of Aggregation Methods
*[Structure](#structure)*


### In this thesis, the following approaches are compared:

1. #### **Centralized Learning:**  
   Used as the baseline. Here, data is collected in one location, which allows for the most consistent model updates. However, this approach often violates privacy and security requirements.

2. #### **FedAvg (Federated Averaging):**  
   A basic method in which client weight updates are averaged. It is simple to implement, but it is sensitive to strong data heterogeneity.

3. #### **FedProx (Federated Proximal):**  
   A modification of FedAvg that adds a proximal term to the loss function. This helps to address discrepancies in local updates and improves convergence in the presence of non-uniform data distributions.

4. #### **FedMA (Federated Matched Averaging):**  
   An approach focused on aligning neural network architectures, which allows for accounting for differences in model structures. This method is particularly useful when there is significant heterogeneity.

---
---

## 4. Experimental Setup
*[Structure](#structure)*

is one model enough?

### **1. Model Overview:**  

### **1. ResNet-18**

ResNet-18 is a deep convolutional neural network (CNN) from the **Residual Network (ResNet)** family. It introduced **skip connections** (residual blocks) to solve the vanishing gradient problem in deep networks.  

#### **Proposers**  
- **Authors**: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (Microsoft Research, 2015).  
- **Paper**: ["Deep Residual Learning for Image Recognition"](https://arxiv.org/abs/1512.03385) (CVPR 2016).  

#### **Architecture Description**  
ResNet-18 has **18 layers** (including skip connections): 
0. **Input Size** 
   - 224x224 RGB images.
1. **Initial Convolution**:  
   - `Conv1`: 7x7 kernel, 64 filters, stride=2 → output size: 112x112.  
   - `MaxPool`: 3x3 kernel, stride=2 → output size: 56x56.  
2. **Residual Blocks**:  
   - **Block1**: 2 layers, 64 filters.  
   - **Block2**: 2 layers, 128 filters.  
   - **Block3**: 2 layers, 256 filters.  
   - **Block4**: 2 layers, 512 filters.  
   Each block uses **3x3 convolutions** and **batch normalization**.  
3. **Global Average Pooling**: Reduces spatial dimensions to 1x1.  
4. **Fully Connected Layer**: Maps features to class probabilities (e.g., 1000 classes for ImageNet).  

 #### **Amount of Weights**  
- **Total Parameters**: ~11.7 million.  
- Breakdown:  
  - Convolution layers: ~11.2M (95% of total).  
  - BatchNorm layers: ~0.4M.  
  - Fully connected layer: ~0.5M.  

 will be used **transfer learning** (pre-trained ResNet-18).  


#### **Datasets for Fine-Tuning on Medical Data**  


#### Most likely 1st one

| **Dataset**         | **Task**                     | **Size**      | **Link**                                                                 |  
|----------------------|------------------------------|---------------|--------------------------------------------------------------------------|  
| **COVIDx**          | COVID-19 detection           | 13,975 images  | [COVIDx](https://github.com/lindawangg/COVID-Net/blob/master/docs/COVIDx.md) |  
| **CheXpert**         | Chest X-ray classification   | 224,316 images | [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/)     |  
| **ISIC**            | Skin lesion classification   | 25,331 images  | [ISIC](https://www.isic-archive.com)                                    |  
| **NIH ChestX-ray14**| Thoracic disease detection   | 112,120 images | [NIH](https://nihcc.app.box.com/v/ChestXray-NIHCC)                      |  

---
#### **Why ResNet-18 for Medical Imaging?**  
- **Efficiency**: Lightweight compared to deeper models (ResNet-50/101).  
- **Transfer Learning**: Leverages features learned from natural images (edges, textures).  
- **Robustness**: Skip connections prevent overfitting on small datasets.  

---


### **Centralized Learning vs. Federated Learning**  

#### **Centralized Learning Experiment**  
- Model will be trained on full dataset and FL experiments will be compared to it

#### **Federated Learning Experiments**  

- The same dataset will be partitioned among multiple clients to simulate a federated environment.  
 - Data is partitioned both IID and Non-IID, so that each client has a different topical focus or imbalanced sizes, simulating real-world heterogeneity.
 - The amount of clients will be the same for each experiment, according to other researches.

**Federated Algorithms:**  
- **FedAvg:** Standard averaging approach for model updates.  
- **FedProx:** FedAvg extension adding a proximal term for stability with heterogeneous data.
- **FedMA:**  A study of the efficiency of matching model structures under significant heterogeneity.

**Training Procedure:**  
1. Each client fine-tunes the local model on its subset of data for a fixed number of epochs.  
2. A central server aggregates the updates using the selected FL method.  
3. The global model is updated and redistributed to clients for the next round.  

---
---

## 5. Results and Analysis
*[Structure](#structure)*

### Experiments  

| Aggregation methods                                                          | metrics   | IID    | Non-IID |
| ---------------------------------------------------------------------------- | --------- | ------ | ------- |
| 1. [Centralized Learning](#centralized-learning)                             | Accuracy  | values | -       |
|                                                                              | F1        | values | -       |
|                                                                              | Recall    | values | -       |
|                                                                              | Precision | values | -       |
|                                                                              | metrics   | values | -       |
|                                                                              | metrics   | values | -       |
| 2. [FedAvg (Federated Averaging)](#fedavg-federated-averaging)               | Accuracy  | values | values  |
|                                                                              | F1        | values | values  |
|                                                                              | Recall    | values | values  |
|                                                                              | Precision | values | values  |
|                                                                              | metrics   | values | values  |
|                                                                              | metrics   | values | values  |
| 3. [FedProx (Federated Proximal)](#fedprox-federated-proximal)               | Accuracy  | values | values  |
|                                                                              | F1        | values | values  |
|                                                                              | Recall    | values | values  |
|                                                                              | Precision | values | values  |
|                                                                              | metrics   | values | values  |
|                                                                              | metrics   | values | values  |
| 4. [FedMA (Federated Matched Averaging)](#fedma-federated-matched-averaging) | Accuracy  | values | values  |
|                                                                              | F1        | values | values  |
|                                                                              | Recall    | values | values  |
|                                                                              | Precision | values | values  |
|                                                                              | metrics   | values | values  |
|                                                                              | metrics   | values | values  |


### What Will Be Evaluated in the Experiments  

- **Training Efficiency:**  
  The speed of convergence and the final accuracy of the model.  

- **Robustness to Data Heterogeneity:**  
  How well the method handles uneven data distribution across devices.  

- **Communication Costs:**  
  The amount of transmitted data, update delays, and impact on network bandwidth.  

Each experiment will highlight the strengths and weaknesses of each aggregation method compared to centralized learning, allowing for the formulation of recommendations for applying federated learning in real-world products.

- Compare performance metrics.  
- Discuss trade-offs between centralized and federated learning.  
- Determine when federated learning yields competitive performance.  
- Analyze how **data distribution (IID vs. Non-IID)** affects performance.  
- Evaluate whether FL methods mitigate **non-homogeneity issues**.  









