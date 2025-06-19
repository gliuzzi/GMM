import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from time import time

from LoadDatasets import get_dataset


class MLP(nn.Module):
    def __init__(self, input_size=784, width=128, output_size=1, activation=F.tanh):
        super().__init__()
        self.activation=activation
        self.fc1 = nn.Linear(input_size, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, output_size)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return self.fc3(x)
    

class MLPProblem:
    def __init__(self, problem_name, l_reg=0.001, device="cpu"):

        problem_name_splitted = problem_name.split("_")
        dataset_name = ''
        for d_n_parts in problem_name_splitted[:-3]:
            dataset_name += d_n_parts + "_"
        dataset_name = dataset_name[:-1]
        model_name = problem_name_splitted[-3]
        activation_name = problem_name_splitted[-2]
        n_seed = int(problem_name_splitted[-1])
        self.name = problem_name

        # print(dataset_name, model_name, activation_name)

        self.device = device if device in ["cpu", "cuda"] else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Device:", self.device)

        self.data_loader, input_size, _ = get_dataset(dataset_name)

        self.dataset_len = len(self.data_loader.dataset)

        if 'TANH' == activation_name:
            activation = F.tanh
        else:
            raise NotImplementedError("No such activation function")

        if model_name == 'MLP':
            torch.manual_seed(n_seed)
            self.model = MLP(input_size=input_size,
                             output_size=1,
                             activation=activation).to(self.device)
        else:
            raise NotImplementedError("No such model")

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.shapes = [p.shape for p in self.model.parameters()]
        self.n = sum(p.numel() for p in self.model.parameters())
        print("N_params", self.n)
        self.x0 = np.ascontiguousarray(self._flatten_params(self.model.parameters()), dtype=np.float32)
        self.l_reg = l_reg

    def _flatten_params(self, params):
        return torch.cat([p.data.view(-1) for p in params]).detach().cpu().numpy()

    def _assign_params(self, flat_x):
        # Assign flat_x to model parameters

        flat_x = np.ascontiguousarray(flat_x, dtype=np.float32)

        with torch.no_grad():
            flat_tensor = torch.tensor(flat_x, dtype=torch.float32, device=self.device)
            idx = 0
            for p in self.model.parameters():
                numel = p.numel()
                p.data.copy_(flat_tensor[idx:idx+numel].view(p.shape))
                idx += numel

    def f(self, x):
        self._assign_params(x)
        self.model.eval()

        total_loss = 0.0
        with torch.no_grad():
            for data, target in self.data_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.loss_fn(output, target)
                
                total_loss += loss * len(data)
                
            total_loss /= self.dataset_len

            for p in self.model.parameters():
                total_loss += (self.l_reg / 2) * (p ** 2).sum()

        return total_loss.item()

    def g(self, x, return_also_f=False):

        self._assign_params(x)
        self.model.train()
        self.model.zero_grad()

        total_loss = 0.0
        for data, target in self.data_loader:
            data, target = data.to(self.device), target.to(self.device)

            output = self.model(data)
            loss = self.loss_fn(output, target)
            
            total_loss += loss * len(data)
            
        total_loss /= self.dataset_len

        for p in self.model.parameters():
            total_loss += (self.l_reg / 2) * (p ** 2).sum()
        

        total_loss.backward()

        grads = torch.cat([p.grad.view(-1) for p in self.model.parameters()])

        if return_also_f:
            return total_loss.item(), grads.detach().cpu().numpy()
        else:
            return grads.detach().cpu().numpy()

    def f_g(self, x):
        return self.g(x, True)

    def get_x0(self):
        print(sum(self.x0))
        return self.x0.copy()

    def get_n(self):
        return self.n

    def hess(self, x):
        raise NotImplementedError("Hessian not implemented for MLPProblem")
