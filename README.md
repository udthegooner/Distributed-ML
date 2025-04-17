# Distributed Training with AllReduce, DDP, and Custom Gather/Scatter

This repository contains an implementation of distributed training techniques using PyTorch, including **AllReduce**, **Distributed Data Parallel (DDP)**, and custom **Gather/Scatter** operations. The code is designed for training models using the CIFAR-10 dataset with a **VGG11** architecture. The setup is intended for **CPU-only** execution.

## Files

### 1. `allreduce.py`
Contains the implementation of the **AllReduce** technique for averaging gradients across multiple nodes in a distributed setup. This helps synchronize the model weights during training.

### 2. `ddp.py`
Implements **Distributed Data Parallel (DDP)**, a PyTorch feature that improves training efficiency by distributing the model across multiple processes and updating the gradients in parallel. This file sets up the environment for DDP training.

### 3. `gather_scatter.py`
Includes custom **Gather** and **Scatter** operations, which collect and distribute model gradients across nodes. This allows for synchronized updates to the model parameters during training.

### 4. `model.py`
Defines the **VGG11** model architecture, which is used for training on the CIFAR-10 dataset. The model consists of several convolutional layers followed by fully connected layers for classification.

## Requirements
- Python 3.x
- PyTorch
- NumPy
- CIFAR-10 dataset (automatically downloaded)

## Setup
To set up and run the code:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/distributed-training.git
   cd distributed-training
   ```

2. Install the required dependencies:
   ```bash
   pip install torch torchvision numpy
   ```

## Running the Code

### Starting the Distributed Training

The code uses **PyTorch's Distributed** framework. To start training on multiple nodes, run the `ddp.py` script with the appropriate arguments. Example for running on 2 nodes:

```bash
python ddp.py --master-ip "127.0.0.1" --num-nodes 2 --rank 0
```

- `--master-ip` is the IP address of the master node.
- `--num-nodes` specifies the number of nodes in the distributed setup.
- `--rank` is the rank of the current node (0 for the master node).

### Training Flow
1. **Gather/Scatter Operations**: The model's parameters are updated during training using custom gather and scatter functions, ensuring that all gradients are synchronized across nodes.
2. **DDP Setup**: The Distributed Data Parallel framework ensures efficient training by parallelizing the process and updating gradients in parallel.
3. **Model**: The VGG11 model is trained on the CIFAR-10 dataset using the **CrossEntropyLoss** criterion and **SGD optimizer**.

### Model and Dataset

- **Model**: VGG11 - A deep neural network with convolutional layers, batch normalization, and ReLU activation.
- **Dataset**: CIFAR-10 - A benchmark dataset for image classification with 60,000 32x32 color images in 10 classes.

## Output
- The code will output the training loss and accuracy after each epoch.
- For each node, the model will synchronize gradients using the AllReduce or custom gather-scatter operations.