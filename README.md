# GCN4MOL

GCN4MOL is a project that demonstrates how to use Graph Convolutional Network (GCN) for molecular property prediction. The repository contains three main scripts: `smiles2graph.py`, `train_gcn.py`, and `test_gcn.py`. These scripts are used to convert SMILES strings to graph data, train a GCN model, and evaluate the model's performance, respectively.

## Overview

1. **smiles2graph.py**: This script converts SMILES strings into graph data objects using RDKit and PyTorch Geometric. Each molecule is represented as a graph where atoms are nodes, and bonds are edges. The script also handles feature extraction for each atom.

2. **train_gcn.py**: This script trains a simple Graph Convolutional Network (GCN) model using the graph data objects generated from `smiles2graph.py`. The model is trained to predict molecular properties, and the training process includes data loading, model initialization, loss computation, and optimization.

3. **test_gcn.py**: This script evaluates the trained GCN model on a test dataset. It computes performance metrics such as Mean Absolute Error (MAE) and R² score and generates a scatter plot comparing actual versus predicted values.

## File Descriptions

### smiles2graph.py

This script converts SMILES strings into graph data objects. Key functionalities include:

- **Atom Feature Extraction**: Extracts features for each atom, such as atom type, degree, formal charge, hybridization, aromaticity, atomic mass, van der Waals radius, covalent radius, chirality, and hydrogen count.
- **Graph Creation**: Creates graph data objects from SMILES strings, including atom features and edge indices.
- **Data Saving**: Saves the generated graph data objects to a file for later use.

### train_gcn.py

This script trains a GCN model using the graph data objects. Key functionalities include:

- **Model Definition**: Defines a simple GCN model with two GCN layers and a linear output layer.
- **Data Loading**: Loads the graph data objects and splits them into training and testing datasets.
- **Training Process**: Trains the GCN model using Mean Squared Error (MSE) loss and Adam optimizer.
- **Model Saving**: Saves the trained model to a file for later use.

### test_gcn.py

This script evaluates the trained GCN model on a test dataset. Key functionalities include:

- **Model Loading**: Loads the trained GCN model and ensures it is in evaluation mode.
- **Data Loading**: Loads the test dataset for evaluation.
- **Prediction and Evaluation**: Makes predictions using the GCN model, collects actual and predicted values, and computes performance metrics.
- **Visualization**: Plots a scatter plot comparing actual versus predicted values, including a diagonal line representing perfect predictions.

## Usage

1. **Generate Graph Data**:
   ```sh
   python smiles2graph.py
   ```
2. **Train the GCN Model**:
   ```sh
   python train_gcn.py
   ```
3. **Evaluate the Model**:
   ```sh
   python test_gcn.py
   ```
## Dependencies
- Python 3.x
- RDKit
- PyTorch

1. Install torch for macOS
```
$ MACOSX_DEPLOYMENT_TARGET=12.5 CC=clang CXX=clang++ python -m pip --no-cache-dir install torch torchvision torchaudio
```
Here you can replace MACOSX_DEPLOYMENT_TARGET=12.X with the correct OS version you have

2. Check which torch version you have
```
$ python -c "import torch; print(torch.__version__)"
>>> 2.3.1 (must be 1.12.0+)
```
It does not matter which version of torch you have as long as it is 1.12.0 and up.

- PyTorch Geometric
  
1. Install torch-geometric
```
$ MACOSX_DEPLOYMENT_TARGET=12.5 CC=clang CXX=clang++ python -m pip --no-cache-dir  install  torch-scatter -f https://data.pyg.org/whl/torch-2.3.1+${cpu}.html
$ MACOSX_DEPLOYMENT_TARGET=12.5 CC=clang CXX=clang++ python -m pip --no-cache-dir  install  torch-sparse -f https://data.pyg.org/whl/torch-2.3.1+${cpu}.html
$ MACOSX_DEPLOYMENT_TARGET=12.5 CC=clang CXX=clang++ python -m pip --no-cache-dir  install  torch-cluster -f https://data.pyg.org/whl/torch-2.3.1+${cpu}.html
$ MACOSX_DEPLOYMENT_TARGET=12.5 CC=clang CXX=clang++ python -m pip --no-cache-dir  install  torch-geometric
```
Make sure to specify the right version of MACOSX_DEPLOYMENT_TARGET and the right torch version in link to the wheels build https://data.pyg.org/whl/torch-2.3.1+${cpu}.html.

- scikit-learn
- matplotlib
- pandas
- numpy
  
