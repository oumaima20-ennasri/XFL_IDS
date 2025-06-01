# XFL_IDS — eXtreme Federated Learning (XFL) for Intrusion Detection

This repository contains the source code used in our scientific paper on the use of **eXtreme Federated Learning (XFL):** for **Intrusion Detection Systems (IDS)**.

## Authors

- **Oumaima ENNASRI**
- **Manal ABBASSI**
- **Yann BEN MAISSA**
- **Rachid EL MOKADEM**
  
## Project Structure

The code is organized into two neural network architectures:
- `MLP/` : Implementation using Multi-Layer Perceptron
- `RNN/` : Implementation using Recurrent Neural Network

Each folder includes:
- `client_FedAvg.py` : Client-side implementation using FedAvg
- `server_FedAvg.py` : Server-side aggregation using FedAvg
- `client_xfl.py` : Client-side implementation using XFL
- `server_xfl.py` : Server-side XFL implementation

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/your-username/XFL_IDS.git
cd XFL_IDS


