import flwr as fl
from typing import List, Tuple, Dict, Optional
from flwr.common import Scalar, Parameters, parameters_to_ndarrays, ndarrays_to_parameters
import numpy as np

def weighted_average(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    total_examples = sum(num_examples for num_examples, _ in metrics)
    accuracy = sum(num_examples * m["accuracy"] for num_examples, m in metrics) / total_examples
    return {"accuracy": accuracy}

class LayerWiseStrategy(fl.server.strategy.FedAvg):
    def __init__(self, num_layers: int, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.global_weights = [None] * num_layers

    def configure_fit(self, server_round, parameters, client_manager):
        sample_clients = client_manager.sample(num_clients=self.min_fit_clients)

        fit_ins_list = []
        for i, client in enumerate(sample_clients):
            config = {
                "server_round": server_round,
                "client_id": i,
            }
            fit_ins_list.append((client, fl.common.FitIns(parameters, config)))

        return fit_ins_list


    def aggregate_fit(self, server_round, results, failures):
        aggregated = [None] * self.num_layers
        counts = [0] * self.num_layers

        for _, fit_res in results:
            layer_index = int(fit_res.metrics["layer_index"])
            num_examples = fit_res.num_examples
            layer_params = parameters_to_ndarrays(fit_res.parameters)  

            param_pos = layer_index * 2
            for i in range(2):
                if aggregated[param_pos + i] is None:
                    aggregated[param_pos + i] = layer_params[i] * num_examples
                else:
                    aggregated[param_pos + i] += layer_params[i] * num_examples
            counts[layer_index] += num_examples

        for layer_index in range(self.num_layers // 2):
            param_pos = layer_index * 2
            if counts[layer_index] > 0:
                for i in range(2):
                    aggregated[param_pos + i] = aggregated[param_pos + i] / counts[layer_index]
            else:
                aggregated[param_pos] = self.global_weights[param_pos]
                aggregated[param_pos + 1] = self.global_weights[param_pos + 1]

        self.global_weights = aggregated
        return ndarrays_to_parameters(aggregated), {}


def main():
    strategy = LayerWiseStrategy(
        num_layers=6,  # nombre de poids/biais dans le modèle MLP (Dense → 2 poids par couche)
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=9,
        min_available_clients=9,
        min_evaluate_clients=9,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()