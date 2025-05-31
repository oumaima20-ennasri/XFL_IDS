import flwr as fl
from typing import List, Tuple, Dict, Optional
from flwr.common import Scalar

def weighted_average(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}
def main():
    # Start Flower server with FedAvg strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1,  # Fraction of clients used in each round (optional)
        fraction_evaluate=1,  # Fraction of clients used for evaluation (optional)
        min_fit_clients=8,  # Minimum number of clients to participate in training (optional)
        min_evaluate_clients=8,  # Minimum number of clients for evaluation (optional)
        min_available_clients=8,  # Minimum number of clients that need to be connected (optional)
        evaluate_metrics_aggregation_fn=weighted_average,  # Use weighted average for accuracy
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",  # Server's IP address
        config=fl.server.ServerConfig(num_rounds=20),  # Number of training rounds
        strategy=strategy,  # Use FedAvg strategy
    )

if __name__ == "__main__":
    main()
