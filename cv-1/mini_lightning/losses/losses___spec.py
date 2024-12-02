import pytest
import torch
import torch.nn.functional as F
import numpy as np

# Import your implementations
from losses.binary_cross_entropy import binary_cross_entropy_loss
from losses.kl_divergence import kl_divergence_loss
from losses.l1 import l1_loss
from losses.mse import mse_loss

@pytest.fixture
def predictions():
    # Create random predictions between 0 and 1
    torch.manual_seed(42)
    return torch.rand(10, 1)

@pytest.fixture
def targets():
    # Create random binary targets
    torch.manual_seed(42)
    return torch.randint(0, 2, (10, 1)).float()

@pytest.fixture
def continuous_targets():
    # Create random continuous targets
    torch.manual_seed(42)
    return torch.rand(10, 1)

class TestBinaryCrossEntropy:
    def test_bce_perfect_prediction(self):
        # Test BCE with perfect predictions
        targets = torch.tensor([[0.], [1.], [0.], [1.]])
        predictions = torch.tensor([[0.], [1.], [0.], [1.]])
        
        loss = binary_cross_entropy_loss(predictions, targets)
        pytorch_loss = F.binary_cross_entropy(predictions, targets)
        
        assert torch.isclose(loss, pytorch_loss, atol=1e-5)
        assert loss < 1e-5  # Should be very close to 0

    def test_bce_worst_prediction(self):
        # Test BCE with completely wrong predictions
        targets = torch.tensor([[0.], [1.], [0.], [1.]])
        predictions = torch.tensor([[1.], [0.], [1.], [0.]])
        
        loss = binary_cross_entropy_loss(predictions, targets)
        assert loss > 0  # Loss should be positive
        assert torch.isfinite(loss)  # Loss should be finite

    def test_bce_random_data(self, predictions, targets):
        loss = binary_cross_entropy_loss(predictions, targets)
        pytorch_loss = F.binary_cross_entropy(predictions, targets)
        
        assert torch.isclose(loss, pytorch_loss, atol=1e-5)

    def test_bce_input_validation(self):
        with pytest.raises(ValueError):
            # Test predictions outside [0,1] range
            invalid_pred = torch.tensor([[1.2], [-0.1]])
            valid_targets = torch.tensor([[1.], [0.]])
            binary_cross_entropy_loss(invalid_pred, valid_targets)

class TestKLDivergence:
    def test_kld_identical_distributions(self):
        # Test KLD between identical distributions
        p = torch.tensor([[0.5, 0.5], [0.3, 0.7]])
        q = p.clone()
        
        loss = kl_divergence_loss(p, q)
        pytorch_loss = F.kl_div(p.log(), q, reduction='sum')
        
        assert torch.isclose(loss, pytorch_loss, atol=1e-5)
        assert loss < 1e-5  # Should be very close to 0

    def test_kld_different_distributions(self):
        p = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        q = torch.tensor([[0.1, 0.9], [0.9, 0.1]])
        
        loss = kl_divergence_loss(p, q)
        assert loss > 0  # KLD should be positive
        assert torch.isfinite(loss)

    def test_kld_uniform_distribution(self):
        # Test KLD with uniform distribution
        p = torch.ones(4, 4) / 16
        q = torch.ones(4, 4) / 16
        
        loss = kl_divergence_loss(p, q)
        assert loss < 1e-5  # Should be very close to 0

    def test_kld_input_validation(self):
        with pytest.raises(ValueError):
            # Test non-normalized probabilities
            invalid_p = torch.tensor([[0.6, 0.6]])  # Sum > 1
            valid_q = torch.tensor([[0.5, 0.5]])
            kl_divergence_loss(invalid_p, valid_q)

class TestL1Loss:
    def test_l1_perfect_prediction(self):
        # Test L1 with perfect predictions
        predictions = torch.tensor([[1.], [2.], [3.]])
        targets = predictions.clone()
        
        loss = l1_loss(predictions, targets)
        pytorch_loss = F.l1_loss(predictions, targets)
        
        assert torch.isclose(loss, pytorch_loss, atol=1e-5)
        assert loss < 1e-5  # Should be very close to 0

    def test_l1_random_data(self, predictions, continuous_targets):
        loss = l1_loss(predictions, continuous_targets)
        pytorch_loss = F.l1_loss(predictions, continuous_targets)
        
        assert torch.isclose(loss, pytorch_loss, atol=1e-5)

    def test_l1_negative_values(self):
        predictions = torch.tensor([[-1.], [2.], [-3.]])
        targets = torch.tensor([[1.], [-2.], [3.]])
        
        loss = l1_loss(predictions, targets)
        pytorch_loss = F.l1_loss(predictions, targets)
        
        assert torch.isclose(loss, pytorch_loss, atol=1e-5)

    def test_l1_scale_invariance(self):
        pred1 = torch.randn(10, 1)
        targets1 = torch.randn(10, 1)
        loss1 = l1_loss(pred1, targets1)
        
        # Scale inputs by a factor
        scale = 5.0
        loss2 = l1_loss(scale * pred1, scale * targets1)
        
        assert torch.isclose(loss2, scale * loss1, atol=1e-5)

class TestMSELoss:
    def test_mse_perfect_prediction(self):
        # Test MSE with perfect predictions
        predictions = torch.tensor([[1.], [2.], [3.]])
        targets = predictions.clone()
        
        loss = mse_loss(predictions, targets)
        pytorch_loss = F.mse_loss(predictions, targets)
        
        assert torch.isclose(loss, pytorch_loss, atol=1e-5)
        assert loss < 1e-5  # Should be very close to 0

    def test_mse_random_data(self, predictions, continuous_targets):
        loss = mse_loss(predictions, continuous_targets)
        pytorch_loss = F.mse_loss(predictions, continuous_targets)
        
        assert torch.isclose(loss, pytorch_loss, atol=1e-5)

    def test_mse_negative_values(self):
        predictions = torch.tensor([[-1.], [2.], [-3.]])
        targets = torch.tensor([[1.], [-2.], [3.]])
        
        loss = mse_loss(predictions, targets)
        pytorch_loss = F.mse_loss(predictions, targets)
        
        assert torch.isclose(loss, pytorch_loss, atol=1e-5)

    def test_mse_squared_property(self):
        # Test that MSE is the square of L1 for single-element tensors
        pred = torch.tensor([[3.]])
        target = torch.tensor([[1.]])
        
        mse = mse_loss(pred, target)
        l1 = l1_loss(pred, target)
        
        assert torch.isclose(mse, l1 * l1, atol=1e-5)

@pytest.mark.parametrize("loss_fn,pytorch_fn", [
    (binary_cross_entropy_loss, F.binary_cross_entropy),
    (l1_loss, F.l1_loss),
    (mse_loss, F.mse_loss)
])
def test_loss_matches_pytorch(loss_fn, pytorch_fn, predictions, continuous_targets):
    # For BCE, we need binary targets
    if loss_fn == binary_cross_entropy_loss:
        targets = (continuous_targets > 0.5).float()
    else:
        targets = continuous_targets
    
    custom_loss = loss_fn(predictions, targets)
    pytorch_loss = pytorch_fn(predictions, targets)
    
    assert torch.isclose(custom_loss, pytorch_loss, atol=1e-5)

def test_all_losses_gradients():
    # Test that all losses produce valid gradients
    predictions = torch.randn(10, 1, requires_grad=True)
    targets = torch.randn(10, 1)
    binary_targets = (targets > 0).float()
    
    # Test BCE
    bce_loss = binary_cross_entropy_loss(torch.sigmoid(predictions), binary_targets)
    bce_loss.backward()
    assert predictions.grad is not None
    
    # Reset gradients
    predictions.grad = None
    
    # Test L1
    l1 = l1_loss(predictions, targets)
    l1.backward()
    assert predictions.grad is not None
    
    # Reset gradients
    predictions.grad = None
    
    # Test MSE
    mse = mse_loss(predictions, targets)
    mse.backward()
    assert predictions.grad is not None