import flwr as fl
from client_app import client_fn
from strategy import strategy

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=4,
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
)