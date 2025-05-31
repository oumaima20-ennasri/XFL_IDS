import flwr as fl
import tensorflow as tf
import numpy as np
import os
import pandas as pd
import time
import psutil
import json

# Définition du modèle RNN avec 3 couches RNN
def get_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Reshape((1, input_shape[0]), input_shape=input_shape),
        tf.keras.layers.SimpleRNN(256, return_sequences=True, name="rnn_1"),
        tf.keras.layers.SimpleRNN(128, return_sequences=True, name="rnn_2"),
        tf.keras.layers.SimpleRNN(64, name="rnn_3"),
        tf.keras.layers.Dense(9, activation="softmax", name="output")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# Chargement des données
def load_data(client_id):
    client_dir = f"client_{client_id}"
    x_train = pd.read_csv(os.path.join(client_dir, "X_train.csv")).values
    y_train = pd.read_csv(os.path.join(client_dir, "y_train.csv")).values
    x_test = pd.read_csv(os.path.join(client_dir, "X_test.csv")).values
    y_test = pd.read_csv(os.path.join(client_dir, "y_test.csv")).values
    return x_train, y_train, x_test, y_test

class LayerWiseRNNClient(fl.client.NumPyClient):
    def __init__(self, cid):
        self.cid = cid
        self.model = get_model((20,))
        self.x_train, self.y_train, self.x_test, self.y_test = load_data(self.cid)
        self.history = {"rounds": [], "model_type": "rnn"}

        # Liste des couches ciblées (nom + index dans get_weights)
        self.layer_map = {
            0: ("rnn_1", 0, 3),  # 3 poids pour SimpleRNN
            1: ("rnn_2", 3, 6),
            2: ("rnn_3", 6, 9),
            3: ("output", 9, 11)  # Dense layer (poids + biais)
        }

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        # Charger les poids du serveur
        self.model.set_weights(parameters)

        client_id = self.cid
        server_round = config["server_round"]

        # Index manuel de couche
        layer_index = (server_round + client_id) % len(self.layer_map)
        layer_name, start, end = self.layer_map[layer_index]

        print(f"Client {client_id}: training layer '{layer_name}' (params {start} to {end})")
        start_time = time.time()        

        # Entraîner tout le modèle
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

        # Extraire les poids de la couche ciblée (index dans la layer_map)
        updated_params = self.model.get_weights()[start:end]

        # Retourner uniquement les poids de la couche sélectionnée
        return updated_params, len(self.x_train), {"layer_name": layer_name, "layer_index": layer_index}


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
        history_file = f"client_{self.cid}_rnn_history_xfl.json"
        with open(history_file, "w") as f:
            json.dump(self.history, f, indent=4)
        print(f" Historique sauvegardé dans {history_file}")

if __name__ == "__main__":

    cid = int(input("Enter your client rank (0 for Client 1, 1 for Client 2, etc.): "))
    fl.client.start_numpy_client(server_address="192.168.1.129:8080", client=LayerWiseRNNClient(cid))
