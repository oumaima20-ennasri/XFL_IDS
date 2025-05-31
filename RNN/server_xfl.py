import flwr as fl
from typing import List, Tuple, Dict, Optional
from flwr.common import Scalar, Parameters, parameters_to_ndarrays, ndarrays_to_parameters
import numpy as np

# Agrégation des métriques
def weighted_average(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    total_examples = sum(num_examples for num_examples, _ in metrics)
    accuracy = sum(num_examples * m["accuracy"] for num_examples, m in metrics) / total_examples
    return {"accuracy": accuracy}

# Stratégie personnalisée
class LayerWiseStrategy(fl.server.strategy.FedAvg):
    def __init__(self, num_layers: int, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.global_weights = [None] * num_layers

        # Define the layer slices for the RNN model (3 parameters per RNN layer)
        self.layer_slices = {
            0: (0, 3),   # RNN 1 - kernel, recurrent_kernel, bias
            1: (3, 6),   # RNN 2 - kernel, recurrent_kernel, bias
            2: (6, 9),   # RNN 3 - kernel, recurrent_kernel, bias
            3: (9, 11),  # Dense - kernel, bias
        }

    def configure_fit(self, server_round, parameters, client_manager):
        sample_clients = client_manager.sample(num_clients=self.min_fit_clients)

        fit_ins_list = []
        for i, client in enumerate(sample_clients):
            config = {
                "server_round": server_round,
                "client_id": i,  # ← injecter un ID unique pour chaque client dans ce round
            }
            fit_ins_list.append((client, fl.common.FitIns(parameters, config)))
            #fit_ins_list.append((client, fl.common.FitIns(ndarrays_to_parameters(self.global_weights), config)))


        return fit_ins_list

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        # Initialize global weights from full model if it's the first round
        if self.global_weights is None:
            print("Initializing global_weights from first result")
            full_model_weights = parameters_to_ndarrays(results[0][1].parameters)
            self.global_weights = [np.copy(w) for w in full_model_weights]

        aggregated = [np.copy(w) for w in self.global_weights]
        layer_counts = {idx: 0 for idx in self.layer_slices}

        for _, fit_res in results:
            layer_index = int(fit_res.metrics["layer_index"])
            num_examples = fit_res.num_examples
            layer_params = parameters_to_ndarrays(fit_res.parameters)

            # Get the slice for this layer (kernel, recurrent_kernel, bias)
            start, end = self.layer_slices[layer_index]

            if len(layer_params) != (end - start):
                raise ValueError(f"Expected {end - start} params for layer {layer_index}, got {len(layer_params)}")

            # Aggregate the layer parameters
            for i in range(start, end):
                if layer_counts[layer_index] == 0:
                    aggregated[i] = layer_params[i - start] * num_examples
                else:
                    aggregated[i] += layer_params[i - start] * num_examples
            layer_counts[layer_index] += num_examples

        # Normalize the aggregated weights by the number of examples for each layer
        for layer_index, (start, end) in self.layer_slices.items():
            count = layer_counts[layer_index]
            if count > 0:
                for i in range(start, end):
                    aggregated[i] = aggregated[i] / count
            else:
                # Use the previous global weights if there were no updates for this layer
                for i in range(start, end):
                    aggregated[i] = self.global_weights[i]

        # Update global weights
        self.global_weights = aggregated
        return ndarrays_to_parameters(aggregated), {}

def main():
    strategy = LayerWiseStrategy(
        num_layers=11,  # Total number of parameters (3 for each RNN + 2 for Dense)
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=8,
        min_available_clients=8,
        min_evaluate_clients=8,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
