import torch
import torch.nn as nn
import torch.nn.functional as F


class Projector(nn.Module):
    """
    A flexible MLP Projector module used to map embeddings between different spaces.

    This class implements a Multi-Layer Perceptron with configurable depth, dimensions,
    activation functions, and dropout.

    Args:
        input_dim (int): Dimensionality of the input features.
        output_dim (int): Dimensionality of the output features.
        hidden_dim (int, optional): Dimensionality of the hidden layers. 
            If not provided, it defaults to the output_dim.
        num_layers (int, optional): Number of linear layers. Default is 2.
        dropout (float, optional): Dropout probability applied after activation. Default is 0.1.
        activation (str, optional): Activation function name ('relu', 'gelu', 'tanh', 'sigmoid', 'leaky_relu'). 
            Default is 'relu'.
    """

    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dim: int = None, 
        num_layers: int = 2, 
        dropout: float = 0.1, 
        activation: str = 'gelu'
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        if self.num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        # Resolve activation function
        activation_map = {
            'relu': nn.ReLU,
            'gelu': nn.GELU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid,
            'leaky_relu': nn.LeakyReLU
        }
        
        if activation.lower() not in activation_map:
             raise ValueError(f"Unsupported activation function: {activation}. Supported: {list(activation_map.keys())}")
        
        act_layer = activation_map[activation.lower()]

        layers = []
        
        # If only 1 layer, it's just a linear projection
        if num_layers == 1:
            layers.append(nn.Linear(self.input_dim, self.output_dim))
        else:
            # Input layer
            layers.append(nn.Linear(self.input_dim, self.hidden_dim))
            layers.append(nn.LayerNorm(self.hidden_dim))
            layers.append(act_layer())
            layers.append(nn.Dropout(self.dropout))
            
            # Hidden layers
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                layers.append(nn.LayerNorm(self.hidden_dim))
                layers.append(act_layer())
                layers.append(nn.Dropout(self.dropout))
            
            # Output layer
            layers.append(nn.Linear(self.hidden_dim, self.output_dim))
            
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the projector.

        Args:
            x (torch.Tensor): Input tensor of shape (..., input_dim)

        Returns:
            torch.Tensor: Projected tensor of shape (..., output_dim)
        """
        return self.net(x)