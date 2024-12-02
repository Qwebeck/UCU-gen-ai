import pytest
import torch
import piq
import numpy as np
from torchvision import transforms

# Import your implementations
from metrics.frechet_inception_distance import calculate_fid
from metrics.lpips import calculate_lpips
from metrics.inception_score import calculate_is
from metrics.kl_divergence import calculate_kld
from metrics.mode_collapse import calculate_mode_collapse

@pytest.fixture
def fake_images():
    # Create random fake images batch
    return torch.randn(10, 3, 32, 32)

@pytest.fixture
def real_images():
    # Create random real images batch
    return torch.randn(10, 3, 32, 32)

class TestFID:
    def test_fid_perfect_match(self):
        # Test FID between identical distributions
        images = torch.randn(100, 3, 32, 32)
        fid_score = calculate_fid(images, images)
        assert abs(fid_score) < 1e-5  # Should be very close to 0

    def test_fid_different_distributions(self, fake_images, real_images):
        # Compare with PIQ implementation
        fid_score_ours = calculate_fid(fake_images, real_images)
        fid_score_piq = piq.FrechetInceptionDistance()(fake_images, real_images)
        
        # Allow for small differences due to implementation details
        assert abs(fid_score_ours - fid_score_piq) < 1.0

    def test_fid_input_validation(self):
        with pytest.raises(ValueError):
            # Test invalid input dimensions
            invalid_images = torch.randn(10, 1, 32, 32)  # Wrong number of channels
            calculate_fid(invalid_images, invalid_images)

class TestLPIPS:
    def test_lpips_identical_images(self):
        images = torch.randn(10, 3, 32, 32)
        lpips_score = calculate_lpips(images, images)
        assert abs(lpips_score) < 1e-5  # Should be very close to 0

    def test_lpips_different_images(self, fake_images, real_images):
        lpips_score_ours = calculate_lpips(fake_images, real_images)
        lpips_score_piq = piq.LPIPS()(fake_images, real_images)
        
        # Compare with PIQ implementation
        assert abs(lpips_score_ours - lpips_score_piq) < 0.1

    def test_lpips_range(self, fake_images, real_images):
        score = calculate_lpips(fake_images, real_images)
        assert 0 <= score <= 1.0  # LPIPS should be between 0 and 1

class TestInceptionScore:
    def test_is_perfect_distribution(self):
        # Create fake logits that would result in perfect distribution
        perfect_logits = torch.ones(100, 1000) / 1000  # Uniform distribution
        is_score = calculate_is(perfect_logits)
        assert abs(is_score - 1.0) < 1e-5  # Should be close to 1

    def test_is_range(self, fake_images):
        is_score = calculate_is(fake_images)
        assert 1.0 <= is_score <= 10.0  # Typical range for IS

    def test_is_batch_independence(self):
        # Score should be similar for different batch sizes
        images1 = torch.randn(50, 3, 32, 32)
        images2 = torch.randn(100, 3, 32, 32)
        
        score1 = calculate_is(images1)
        score2 = calculate_is(images2)
        
        # Scores should be relatively close despite different batch sizes
        assert abs(score1 - score2) < 1.0

class TestKLDivergence:
    def test_kld_identical_distributions(self):
        dist = torch.distributions.Normal(0, 1)
        samples = dist.sample((1000,))
        kld = calculate_kld(samples, samples)
        assert abs(kld) < 1e-5  # Should be very close to 0

    def test_kld_different_distributions(self):
        dist1 = torch.distributions.Normal(0, 1)
        dist2 = torch.distributions.Normal(2, 1)
        samples1 = dist1.sample((1000,))
        samples2 = dist2.sample((1000,))
        
        kld = calculate_kld(samples1, samples2)
        assert kld > 0  # KL divergence should be positive

    def test_kld_non_negativity(self, fake_images, real_images):
        kld = calculate_kld(fake_images, real_images)
        assert kld >= 0  # KL divergence should always be non-negative

class TestModeCollapse:
    def test_mode_collapse_identical_samples(self):
        # Create batch of identical images
        images = torch.ones(10, 3, 32, 32)
        collapse_score = calculate_mode_collapse(images)
        assert collapse_score > 0.9  # Should indicate high mode collapse

    def test_mode_collapse_diverse_samples(self):
        # Create batch of diverse images
        images = torch.randn(10, 3, 32, 32)
        collapse_score = calculate_mode_collapse(images)
        assert collapse_score < 0.5  # Should indicate low mode collapse

    def test_mode_collapse_range(self, fake_images):
        score = calculate_mode_collapse(fake_images)
        assert 0 <= score <= 1.0  # Score should be normalized between 0 and 1

@pytest.mark.parametrize("metric_fn,expected_range", [
    (calculate_fid, (0, float('inf'))),
    (calculate_lpips, (0, 1)),
    (calculate_is, (1, 10)),
    (calculate_kld, (0, float('inf'))),
    (calculate_mode_collapse, (0, 1))
])
def test_metric_ranges(metric_fn, expected_range, fake_images, real_images):
    if metric_fn in [calculate_fid, calculate_lpips]:
        score = metric_fn(fake_images, real_images)
    else:
        score = metric_fn(fake_images)
    
    assert expected_range[0] <= score <= expected_range[1], \
        f"{metric_fn.__name__} returned score outside expected range"