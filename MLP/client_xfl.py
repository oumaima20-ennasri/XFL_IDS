import flwr as fl
import tensorflow as tf
import numpy as np
import copy
import os
import pandas as pd
import time
import psutil
import json

# Exemple simple : modèle MLP 3 couches (poids/biais)
def get_model(input_shape):

    model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(256, activation="relu", input_shape=input_shape),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(9, activation="softmax")
        ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# Fonction pour charger les données d'un client donné depuis les fichiers CSV
def load_data(client_id):
    client_dir = f"client_mpl_{client_id}"
    
    # Charger les données d'entraînement
    x_train = pd.read_csv(os.path.join(client_dir, "X_train.csv")).values
    y_train = pd.read_csv(os.path.join(client_dir, "y_train.csv")).values
    
    # Charger les données de test
    x_test = pd.read_csv(os.path.join(client_dir, "X_test.csv")).values
    y_test = pd.read_csv(os.path.join(client_dir, "y_test.csv")).values
    
    return x_train, y_train, x_test, y_test

class LayerWiseClient(fl.client.NumPyClient):
    def __init__(self, cid):
        self.cid = int(cid)
        self.model = get_model((20,))
        self.x_train, self.y_train, self.x_test, self.y_test = load_data(self.cid)
        self.history = {"rounds": [], "model_type": "rnn"}


    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)

        # Récupérer ID du client injecté par le serveur
        client_id = config["client_id"]
        server_round = config["server_round"]

        total_layers = len(self.model.get_weights()) // 2
        layer_index = (server_round + client_id) % total_layers
        start = layer_index * 2

        print(f"Client {client_id}: training layer {layer_index}")
        start_time = time.time() 
        history = self.model.fit(self.x_train, self.y_train, epochs=10, batch_size=64, verbose=1)
        
        train_time = time.time() - start_time
        round_metrics = {
            "round": len(self.history["rounds"]) + 1,
            "train_loss": history.history["loss"][-1],
            "train_accuracy": history.history["accuracy"][-1],
            "train_time": train_time,
            "cpu_usage": psutil.cpu_percent(),
            "ram_usage": psutil.virtual_memory().percent,
            "network_sent_MB": psutil.net_io_counters().bytes_sent / (1024 * 1024),
            "network_received_MB": psutil.net_io_counters().bytes_recv / (1024 * 1024),
            "model_size_MB": sum([param.nbytes for param in self.model.get_weights()]) / (1024 * 1024),
        }
        self.history["rounds"].append(round_metrics)
        self.save_training_history()
        
        updated_params = self.model.get_weights()[start:start+2]
        return updated_params, len(self.x_train), {"layer_index": layer_index}



    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        start_time = time.time()
        loss, acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        eval_time = time.time() - start_time
        # Enregistrer les métriques d'évaluation
        if self.history["rounds"]:  # Si la liste n'est pas vide
            self.history["rounds"][-1]["test_loss"] = loss
            self.history["rounds"][-1]["test_accuracy"] = acc
            self.history["rounds"][-1]["eval_time"] = eval_time
            self.save_training_history()
        return loss, len(self.x_test), {"accuracy": acc}
    
    def save_training_history(self):
        """Sauvegarde l'historique en JSON."""
        history_file = f"client_{self.cid}_mlp_history_iid.json"
        with open(history_file, "w") as f:
            json.dump(self.history, f, indent=4)
        print(f"Historique sauvegardé dans {history_file}")
if __name__ == "__main__":

    cid = int(input("Enter your client rank (0 for Client 1, 1 for Client 2, etc.): "))
    fl.client.start_numpy_client(server_address="192.168.1.129:8080", client=LayerWiseClient(cid))
