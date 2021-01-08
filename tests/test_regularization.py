import torch
import torch.nn as nn

from torchfitter.regularization import (
    L1Regularization, 
    ElasticNetRegularization
)

from torchfitter.testing import change_model_params


model = nn.Linear(2,2)

# change weights and biases
weights = torch.Tensor([[0.5675, 0.8352], [0.2056, 0.5932]]).float()
biases = torch.Tensor([-0.2723,  0.1896]).float()

change_model_params(model, weights, biases)


def test_L1Regularization():
    regularizer = L1Regularization(regularization_rate=0.01, biases=False)
    
    obtained_term = regularizer(model.named_parameters()).item()
    expected_term = 0.022014999762177467
    
    msg = "Error in L1 regularization penalty"
    assert obtained_term == expected_term, msg


def test_ElasticNetRegularization():
    regularizer = ElasticNetRegularization(
        l1_lambda=0.01, 
        l2_lambda=0.05, 
        biases=False
    )
    
    obtained_term = regularizer(model.named_parameters()).item()
    expected_term = 0.08146609365940094
    
    msg = "Error in Elastic net regularization penalty"
    assert obtained_term == expected_term, msg