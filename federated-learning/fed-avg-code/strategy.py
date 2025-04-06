from flwr.server.strategy import FedAvg
import task

class CustomFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_history = {
            "val_loss": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": []
        }
        self.final_parameters = None
    
    def evaluate(self, server_round, parameters):
        result = super().evaluate(server_round, parameters)
        if result:
            metrics = result[1]
            for key in self.metrics_history:
                self.metrics_history[key].append(metrics[key])
            self.final_parameters = parameters  # Store the latest parameters
        return result

strategy = CustomFedAvg(
    fraction_fit=1.0,
    min_fit_clients=4,
    min_available_clients=4,
    evaluate_fn=task.evaluate_fn,
    on_fit_config_fn=task.fit_config
)