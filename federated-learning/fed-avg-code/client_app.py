import torch
from flwr.client import NumPyClient
import task

class FlowerClient(NumPyClient):
    def __init__(self, train_loader):
        self.train_loader = train_loader
        self.model = task.get_model()
        self.optimizer = task.get_optimizer(self.model)
        self.criterion = task.get_loss_function()
        self.device = task.device

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        local_epochs = config["local_epochs"]
        for epoch in range(local_epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(config), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        # Local evaluation is optional and not implemented here
        pass

def client_fn(cid: str) -> NumPyClient:
    train_loader = task.client_train_loaders[int(cid)]
    return FlowerClient(train_loader)