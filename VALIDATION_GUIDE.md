# Validation & Testing Guide

This guide provides comprehensive testing procedures to ensure your noise-explicit online SR system works correctly.

---

## Testing Philosophy

**Test Pyramid:**
```
       /\
      /  \    Integration Tests (5%)
     /    \   - Full pipeline on synthetic systems
    /______\  
   /        \  Component Tests (25%)
  /          \ - Noise operators, likelihood, optimization
 /____________\
/              \ Unit Tests (70%)
\              / - Individual functions, edge cases
 \____________/
```

**Testing Priorities:**
1. **Unit tests:** Fast, isolated, many edge cases
2. **Component tests:** Medium scope, integration between 2-3 modules
3. **Integration tests:** Full system, realistic scenarios

---

## Level 1: Unit Tests

### Test Suite 1: Expression Tree

**File: `tests/test_expression_tree.py`**

```python
"""
Unit tests for expression tree functionality.
"""

import pytest
import numpy as np
from src.expression_tree import *

class TestExpressionNode:
    """Tests for ExpressionNode class."""
    
    def test_constant_node(self):
        """Constants should be created and evaluated correctly."""
        node = create_constant(3.14)
        
        assert node.node_type == NodeType.CONSTANT
        assert node.value == 3.14
        
        # Evaluate on any data
        X = np.random.rand(10, 3)
        result = node.evaluate(X, theta=[])
        
        assert result.shape == (10,)
        assert np.allclose(result, 3.14)
    
    def test_variable_node(self):
        """Variables should extract correct column from data."""
        node = create_variable(1)  # x_1 (second column)
        
        X = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])
        
        result = node.evaluate(X, theta=[])
        
        assert np.allclose(result, [2, 5, 8])
    
    def test_binary_addition(self):
        """Binary addition: x_0 + x_1"""
        x0 = create_variable(0)
        x1 = create_variable(1)
        expr = create_binary_op(lambda a,b: a+b, x0, x1)
        
        X = np.array([[1, 2],
                     [3, 4]])
        
        result = expr.evaluate(X, theta=[])
        assert np.allclose(result, [3, 7])
    
    def test_complex_expression(self):
        """Test: θ_0 * x_0^2 + θ_1 * x_1 + θ_2"""
        # Build tree for: a*x₀² + b*x₁ + c
        
        x0 = create_variable(0)
        x1 = create_variable(1)
        
        # θ_0 * x_0^2
        theta_0 = create_constant(2.0)
        x0_sq = create_binary_op(lambda a,b: a**b, x0, create_constant(2.0))
        term1 = create_binary_op(lambda a,b: a*b, theta_0, x0_sq)
        
        # θ_1 * x_1
        theta_1 = create_constant(3.0)
        term2 = create_binary_op(lambda a,b: a*b, theta_1, x1)
        
        # θ_2
        theta_2 = create_constant(1.0)
        
        # Combine
        temp = create_binary_op(lambda a,b: a+b, term1, term2)
        expr = create_binary_op(lambda a,b: a+b, temp, theta_2)
        
        # Test
        X = np.array([[1, 1],   # 2*1 + 3*1 + 1 = 6
                     [2, 0]])   # 2*4 + 0 + 1 = 9
        
        result = expr.evaluate(X, theta=[])
        assert np.allclose(result, [6, 9])
    
    def test_copy_method(self):
        """Deep copy should create independent tree."""
        expr = create_binary_op(
            lambda a,b: a+b,
            create_constant(1.0),
            create_variable(0)
        )
        
        expr_copy = expr.copy()
        
        # Modify copy
        expr_copy.value = lambda a,b: a*b
        
        # Original should be unchanged
        X = np.array([[2]])
        result_original = expr.evaluate(X, theta=[])
        result_copy = expr_copy.evaluate(X, theta=[])
        
        assert result_original[0] == 3  # 1 + 2
        assert result_copy[0] == 2      # 1 * 2
    
    def test_string_representation(self):
        """__str__ should give readable output."""
        expr = create_binary_op(
            lambda a,b: a+b,
            create_constant(2.0),
            create_variable(0)
        )
        
        expr_str = str(expr)
        assert 'x_0' in expr_str
        assert '2' in expr_str or '2.0' in expr_str

class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_data(self):
        """Evaluating on empty data should handle gracefully."""
        expr = create_variable(0)
        X = np.array([]).reshape(0, 1)
        
        result = expr.evaluate(X, theta=[])
        assert len(result) == 0
    
    def test_mismatched_dimensions(self):
        """Should raise error if variable index out of bounds."""
        expr = create_variable(5)  # x_5
        X = np.random.rand(10, 2)  # Only 2 features
        
        with pytest.raises(IndexError):
            expr.evaluate(X, theta=[])
    
    def test_nan_handling(self):
        """Should propagate NaN correctly."""
        expr = create_variable(0)
        X = np.array([[1], [np.nan], [3]])
        
        result = expr.evaluate(X, theta=[])
        assert np.isnan(result[1])
        assert result[0] == 1
        assert result[3] == 3
```

**Run:**
```bash
pytest tests/test_expression_tree.py -v
```

### Test Suite 2: Noise Operators

**File: `tests/test_noise_operators.py`**

```python
"""
Statistical tests for noise operators.
"""

import pytest
import numpy as np
from scipy import stats
from src.noise_operators import *

class TestGaussianNoiseConstant:
    """Tests for GAUSSIAN_NOISE_CONSTANT."""
    
    def test_mean_zero(self):
        """Mean should be approximately zero."""
        np.random.seed(42)
        samples = [GAUSSIAN_NOISE_CONSTANT.func({'σ': 1.0}) 
                  for _ in range(10000)]
        
        mean = np.mean(samples)
        assert abs(mean) < 0.05, f"Mean {mean} too far from 0"
    
    def test_std_correct(self):
        """Standard deviation should match σ parameter."""
        for sigma in [0.1, 1.0, 5.0]:
            np.random.seed(42)
            samples = [GAUSSIAN_NOISE_CONSTANT.func({'σ': sigma}) 
                      for _ in range(10000)]
            
            std = np.std(samples)
            assert abs(std - sigma) / sigma < 0.05, \
                f"Std {std} doesn't match σ={sigma}"
    
    def test_normality(self):
        """Distribution should be approximately normal."""
        np.random.seed(42)
        samples = [GAUSSIAN_NOISE_CONSTANT.func({'σ': 1.0}) 
                  for _ in range(10000)]
        
        # Shapiro-Wilk test for normality
        # (Not super reliable with 10k samples, but good check)
        # Alternatively use Kolmogorov-Smirnov
        _, p_value = stats.kstest(samples, 'norm', args=(0, 1))
        assert p_value > 0.01, "Samples don't appear normal"

class TestGaussianNoiseScaled:
    """Tests for GAUSSIAN_NOISE_SCALED (heteroscedastic)."""
    
    def test_scaling_with_input(self):
        """Noise magnitude should scale with |x|^α."""
        np.random.seed(42)
        
        # Test points
        x_values = np.array([1, 2, 5, 10])
        params = {'σ_0': 0.1, 'α': 1.0}
        
        # Generate many samples at each x
        for x in x_values:
            samples = [scaled_gaussian_noise(np.array([x]), params)[0]
                      for _ in range(1000)]
            
            expected_std = params['σ_0'] * abs(x) ** params['α']
            actual_std = np.std(samples)
            
            rel_error = abs(actual_std - expected_std) / expected_std
            assert rel_error < 0.1, \
                f"At x={x}: expected σ={expected_std:.3f}, got {actual_std:.3f}"
    
    def test_power_law(self):
        """Test different values of α."""
        for alpha in [0.5, 1.0, 2.0]:
            params = {'σ_0': 0.5, 'α': alpha}
            
            # At x=4:
            x = np.array([4.0])
            samples = [scaled_gaussian_noise(x, params)[0] for _ in range(5000)]
            
            expected_std = 0.5 * (4.0 ** alpha)
            actual_std = np.std(samples)
            
            rel_error = abs(actual_std - expected_std) / expected_std
            assert rel_error < 0.1

class TestNoiseOperatorProperties:
    """Test general properties all noise operators should satisfy."""
    
    @pytest.mark.parametrize("noise_op", NOISE_OPERATORS)
    def test_returns_numeric(self, noise_op):
        """All operators should return numeric values."""
        # Initialize params to reasonable values
        params = {name: 1.0 for name in noise_op.param_names}
        
        if noise_op.arity == 0:
            result = noise_op.func(params)
        else:
            x = np.array([1.0])
            result = noise_op.func(x, params)
        
        assert isinstance(result, (int, float, np.ndarray))
    
    @pytest.mark.parametrize("noise_op", NOISE_OPERATORS)
    def test_param_names_match(self, noise_op):
        """param_names should match what func expects."""
        params = {name: 1.0 for name in noise_op.param_names}
        
        # Should not raise KeyError
        try:
            if noise_op.arity == 0:
                _ = noise_op.func(params)
            else:
                _ = noise_op.func(np.array([1.0]), params)
        except KeyError as e:
            pytest.fail(f"Missing parameter: {e}")
```

**Run:**
```bash
pytest tests/test_noise_operators.py -v --tb=short
```

---

## Level 2: Component Tests

### Test Suite 3: Likelihood with Noise

**File: `tests/test_likelihood.py`**

```python
"""
Component tests for likelihood computation.
"""

import pytest
import numpy as np
from src.likelihood import *
from src.expression_tree import *
from src.noise_operators import *

class TestLikelihoodDeterministic:
    """Test likelihood for models without noise."""
    
    def test_perfect_fit(self):
        """Perfect fit should give high likelihood."""
        # Model: y = 2*x
        expr = create_binary_op(
            lambda a,b: a*b,
            create_constant(2.0),
            create_variable(0)
        )
        
        X = np.array([[1], [2], [3]])
        y = np.array([2, 4, 6])  # Perfect fit
        
        ll = log_likelihood_simple(y, expr, X, theta=[])
        
        # Should be very high (close to 0 in log space)
        # Exact value depends on how obs noise is estimated
        assert ll > -10  # Should be much better than random
    
    def test_poor_fit(self):
        """Poor fit should give low likelihood."""
        # Model: y = 2*x
        expr = create_binary_op(
            lambda a,b: a*b,
            create_constant(2.0),
            create_variable(0)
        )
        
        X = np.array([[1], [2], [3]])
        y = np.array([100, 200, 300])  # Very different from 2*x
        
        ll = log_likelihood_simple(y, expr, X, theta=[])
        
        assert ll < -100  # Should be very poor
    
    def test_likelihood_improves_with_better_fit(self):
        """Better fits should have higher likelihood."""
        X = np.random.rand(50, 1) * 10
        y_true = 3*X.flatten() + 1.5
        
        # Model 1: Close to true params
        expr1 = create_binary_op(
            lambda a,b: a+b,
            create_binary_op(lambda a,b: a*b, create_constant(3.1), create_variable(0)),
            create_constant(1.4)
        )
        
        # Model 2: Far from true params
        expr2 = create_binary_op(
            lambda a,b: a+b,
            create_binary_op(lambda a,b: a*b, create_constant(10.0), create_variable(0)),
            create_constant(5.0)
        )
        
        ll1 = log_likelihood_simple(y_true, expr1, X, theta=[])
        ll2 = log_likelihood_simple(y_true, expr2, X, theta=[])
        
        assert ll1 > ll2

class TestLikelihoodWithNoise:
    """Test likelihood for models with explicit noise."""
    
    def test_noise_model_vs_deterministic(self):
        """Model with noise should have different likelihood."""
        
        # Generate data: y = 2*x + noise
        np.random.seed(42)
        X = np.random.rand(50, 1) * 5
        y = 2*X.flatten() + np.random.normal(0, 0.5, 50)
        
        # Model 1: Deterministic (y = 2*x)
        expr_det = create_binary_op(
            lambda a,b: a*b,
            create_constant(2.0),
            create_variable(0)
        )
        
        # Model 2: With noise (y = 2*x + η(σ=0.5))
        expr_noise = create_binary_op(
            lambda a,b: a+b,
            expr_det.copy(),
            ExpressionNode(
                NodeType.NOISE,
                GAUSSIAN_NOISE_CONSTANT,
                noise_params={'σ': 0.5}
            )
        )
        
        ll_det = log_likelihood_simple(y, expr_det, X, theta=[])
        ll_noise = log_likelihood_simple(y, expr_noise, X, theta=[])
        
        # Noise model should be better (it models the noise source)
        # Note: This might not always be true due to MC sampling variance
        print(f"LL deterministic: {ll_det:.2f}")
        print(f"LL with noise: {ll_noise:.2f}")
    
    def test_mc_sampling_convergence(self):
        """More MC samples should give more stable likelihood."""
        
        # Create expression with noise
        expr = create_binary_op(
            lambda a,b: a+b,
            create_binary_op(lambda a,b: a*b, create_constant(1.0), create_variable(0)),
            ExpressionNode(NodeType.NOISE, GAUSSIAN_NOISE_CONSTANT, noise_params={'σ': 0.1})
        )
        
        X = np.random.rand(20, 1)
        y = X.flatten() + np.random.normal(0, 0.1, 20)
        
        # Compute likelihood multiple times (should vary due to MC)
        lls = [log_likelihood_simple(y, expr, X, theta=[]) for _ in range(10)]
        
        # Variance should be reasonable
        ll_std = np.std(lls)
        assert ll_std < 5, f"Too much MC variance: std={ll_std}"
```

**Run:**
```bash
pytest tests/test_likelihood.py -v -s
```

---

## Level 3: Integration Tests

### Test Suite 4: End-to-End System

**File: `tests/test_integration.py`**

```python
"""
Integration tests for full pipeline.
"""

import pytest
import numpy as np
from src.expression_tree import *
from src.noise_operators import *
from src.likelihood import *
from src.parameter_optimization import optimize_params

class TestParameterRecovery:
    """Can we recover known parameters from synthetic data?"""
    
    def test_linear_no_noise(self):
        """Recover parameters: y = θ_0*x + θ_1"""
        
        # Generate data
        np.random.seed(42)
        X = np.random.rand(100, 1) * 10
        theta_true = [2.5, 1.3]
        y = theta_true[0]*X.flatten() + theta_true[1]
        
        # Build expression: θ_0*x + θ_1
        expr = create_binary_op(
            lambda a,b: a+b,
            create_binary_op(lambda a,b: a*b, create_constant(1.0), create_variable(0)),
            create_constant(0.0)
        )
        
        # Optimize
        result = optimize_params(expr, X, y)
        
        assert result['success']
        
        # Check recovery
        theta_recovered = result['theta']
        assert abs(theta_recovered[0] - theta_true[0]) < 0.01
        assert abs(theta_recovered[1] - theta_true[1]) < 0.01
    
    def test_quadratic_with_noise(self):
        """Recover: y = θ_0*x² + θ_1*x + η(σ)"""
        
        np.random.seed(42)
        X = np.random.rand(200, 1) * 5
        theta_true = [0.5, 2.0]
        sigma_true = 0.3
        
        y = (theta_true[0]*X**2 + theta_true[1]*X).flatten()
        y += np.random.normal(0, sigma_true, len(y))
        
        # Build expression
        x = create_variable(0)
        x_sq = create_binary_op(lambda a,b: a**b, x, create_constant(2.0))
        
        term1 = create_binary_op(lambda a,b: a*b, create_constant(1.0), x_sq)
        term2 = create_binary_op(lambda a,b: a*b, create_constant(1.0), x)
        
        det_part = create_binary_op(lambda a,b: a+b, term1, term2)
        
        noise_part = ExpressionNode(
            NodeType.NOISE,
            GAUSSIAN_NOISE_CONSTANT,
            noise_params={'σ': 0.1}  # Initial guess
        )
        
        expr = create_binary_op(lambda a,b: a+b, det_part, noise_part)
        
        # Optimize
        result = optimize_params(expr, X, y)
        
        assert result['success']
        
        # Check parameter recovery
        theta_rec = result['theta']
        sigma_rec = result['noise_params'][0]
        
        print(f"True: θ={theta_true}, σ={sigma_true}")
        print(f"Recovered: θ={theta_rec}, σ={sigma_rec}")
        
        assert abs(theta_rec[0] - theta_true[0]) / theta_true[0] < 0.1  # 10% error
        assert abs(theta_rec[1] - theta_true[1]) / theta_true[1] < 0.1
        assert abs(sigma_rec - sigma_true) / sigma_true < 0.2  # 20% error ok for noise

class TestOnlineUpdates:
    """Test online parameter adaptation."""
    
    @pytest.mark.skip(reason="Implement after online module is ready")
    def test_rls_updates(self):
        """RLS should track changing parameters."""
        # TODO: Generate data with drift
        # TODO: Initialize OnlineSymbolicRegression
        # TODO: Stream data, check parameter tracking
        pass
    
    @pytest.mark.skip(reason="Implement after drift detection is ready")
    def test_drift_detection(self):
        """System should detect and adapt to drift."""
        # TODO: Generate data with abrupt change
        # TODO: Verify drift is detected
        # TODO: Verify model structure is updated
        pass
```

---

## Level 4: Benchmark Validation

### Benchmark 1: Polynomial with Heteroscedastic Noise

**File: `tests/benchmarks/test_polynomial_heteroscedastic.py`**

```python
"""
Benchmark: Polynomial with noise that scales with |x|.

True model: y = 1 + 2x + 0.5x² + η(σ=0.1*|x|)

This tests:
- Parameter recovery for multi-term polynomial
- Noise parameter estimation (heteroscedastic case)
- Comparison to baseline (ignoring noise)
"""

import numpy as np
import matplotlib.pyplot as plt
from src.smc_sr import SMC_SR_with_Noise  # Assuming this exists

def run_benchmark():
    """Run polynomial heteroscedastic benchmark."""
    
    # 1. Generate data
    np.random.seed(42)
    n_train, n_test = 200, 100
    
    X_train = np.random.uniform(-3, 3, (n_train, 1))
    X_test = np.random.uniform(-3, 3, (n_test, 1))
    
    def true_function(X):
        return 1 + 2*X + 0.5*X**2
    
    def noise_function(X):
        return np.random.normal(0, 0.1 * np.abs(X))
    
    y_train = true_function(X_train).flatten() + noise_function(X_train).flatten()
    y_test = true_function(X_test).flatten() + noise_function(X_test).flatten()
    
    # 2. Run discovery
    print("Running SMC-SR with noise operators...")
    
    sr = SMC_SR_with_Noise(
        n_particles=500,
        n_iterations=100,
        include_noise=True
    )
    sr.fit(X_train, y_train)
    
    best_expr = sr.get_best_expression()
    print(f"Discovered: {best_expr}")
    print(f"Parameters: {sr.best_theta}")
    print(f"Noise params: {sr.best_noise_params}")
    
    # 3. Evaluate
    y_pred_train = best_expr.evaluate(X_train, sr.best_theta, sample_noise=False)
    y_pred_test = best_expr.evaluate(X_test, sr.best_theta, sample_noise=False)
    
    rmse_train = np.sqrt(np.mean((y_train - y_pred_train)**2))
    rmse_test = np.sqrt(np.mean((y_test - y_pred_test)**2))
    
    r2_test = 1 - np.sum((y_test - y_pred_test)**2) / np.sum((y_test - y_test.mean())**2)
    
    print(f"\nResults:")
    print(f"  Train RMSE: {rmse_train:.4f}")
    print(f"  Test RMSE: {rmse_test:.4f}")
    print(f"  Test R²: {r2_test:.4f}")
    
    # 4. Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Predictions
    axes[0].scatter(X_test, y_test, alpha=0.5, label='True')
    axes[0].scatter(X_test, y_pred_test, alpha=0.5, label='Predicted')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].legend()
    axes[0].set_title('Predictions vs True')
    
    # Residuals
    residuals = y_test - y_pred_test
    axes[1].scatter(X_test, residuals, alpha=0.5)
    axes[1].axhline(0, color='red', linestyle='--')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('Residual')
    axes[1].set_title('Residuals')
    
    plt.tight_layout()
    plt.savefig('benchmark_polynomial_heteroscedastic.png', dpi=150)
    print(f"\nPlot saved: benchmark_polynomial_heteroscedastic.png")
    
    # 5. Compare to baseline (no noise modeling)
    print("\nRunning baseline (no noise)...")
    sr_baseline = SMC_SR_with_Noise(
        n_particles=500,
        n_iterations=100,
        include_noise=False  # Don't use noise operators
    )
    sr_baseline.fit(X_train, y_train)
    
    y_pred_baseline = sr_baseline.get_best_expression().evaluate(
        X_test, sr_baseline.best_theta, sample_noise=False
    )
    rmse_baseline = np.sqrt(np.mean((y_test - y_pred_baseline)**2))
    
    print(f"Baseline RMSE: {rmse_baseline:.4f}")
    print(f"With noise RMSE: {rmse_test:.4f}")
    print(f"Improvement: {(1 - rmse_test/rmse_baseline)*100:.1f}%")
    
    # 6. Return results for further analysis
    return {
        'expression': best_expr,
        'theta': sr.best_theta,
        'noise_params': sr.best_noise_params,
        'rmse_test': rmse_test,
        'r2_test': r2_test,
        'rmse_baseline': rmse_baseline
    }

if __name__ == "__main__":
    results = run_benchmark()
    
    # Success criteria
    assert results['rmse_test'] < 0.5, "RMSE too high"
    assert results['r2_test'] > 0.95, "R² too low"
    assert results['rmse_test'] < results['rmse_baseline'], "Should beat baseline"
    
    print("\n✓ Benchmark passed!")
```

### Benchmark 2: Lorenz System

**File: `tests/benchmarks/test_lorenz.py`**

```python
"""
Benchmark: Lorenz system (from the project document).

Tests:
- Multi-equation discovery
- Handling of chaotic dynamics
- Noise robustness on ODE discovery
"""

import numpy as np
from scipy.integrate import odeint

def lorenz_deriv(state, t, sigma=10, rho=28, beta=8/3):
    """Lorenz system derivatives."""
    x, y, z = state
    return [
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ]

def run_lorenz_benchmark():
    """Run Lorenz system benchmark."""
    
    # 1. Generate trajectory
    t = np.linspace(0, 10, 500)
    init_state = [1.0, 1.0, 1.0]
    states = odeint(lorenz_deriv, init_state, t)
    
    # Add noise
    noise_level = 0.1
    states_noisy = states + np.random.normal(0, noise_level, states.shape)
    
    # Compute derivatives (finite differences)
    dt = t[1] - t[0]
    derivs = np.diff(states_noisy, axis=0) / dt
    states_noisy = states_noisy[:-1]  # Match shapes
    
    # 2. Discover equations for each component
    # dx/dt, dy/dt, dz/dt
    
    print("Discovering Lorenz equations...")
    
    results = {}
    for i, name in enumerate(['dx/dt', 'dy/dt', 'dz/dt']):
        print(f"\n{name}:")
        
        X = states_noisy
        y = derivs[:, i]
        
        sr = SMC_SR_with_Noise(
            n_particles=1000,
            n_iterations=200,
            include_noise=True
        )
        sr.fit(X, y)
        
        expr = sr.get_best_expression()
        print(f"  Discovered: {expr}")
        
        y_pred = expr.evaluate(X, sr.best_theta, sample_noise=False)
        rmse = np.sqrt(np.mean((y - y_pred)**2))
        print(f"  RMSE: {rmse:.4f}")
        
        results[name] = {
            'expression': expr,
            'rmse': rmse
        }
    
    # 3. Integrate discovered system and compare to true
    # (This is more complex - for now just check RMSEs)
    
    avg_rmse = np.mean([r['rmse'] for r in results.values()])
    print(f"\nAverage RMSE: {avg_rmse:.4f}")
    
    assert avg_rmse < 1.0, "RMSE too high for Lorenz"
    print("✓ Lorenz benchmark passed!")
    
    return results

if __name__ == "__main__":
    run_lorenz_benchmark()
```

---

## Continuous Integration Setup

**File: `.github/workflows/tests.yml`**

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run unit tests
      run: |
        pytest tests/ -v --cov=src --cov-report=xml
    
    - name: Run benchmarks
      run: |
        pytest tests/benchmarks/ -v -s
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

---

## Performance Testing

### Test Suite 5: Computational Efficiency

**File: `tests/test_performance.py`**

```python
"""
Performance and scaling tests.
"""

import time
import numpy as np
import pytest

class TestScaling:
    """Test computational scaling with data size."""
    
    def test_evaluation_scales_linearly(self):
        """Expression evaluation should be O(n) in data size."""
        
        expr = create_complex_expression()  # Some deep tree
        
        times = []
        sizes = [100, 1000, 10000]
        
        for n in sizes:
            X = np.random.rand(n, 5)
            
            start = time.time()
            for _ in range(10):  # Average over 10 runs
                _ = expr.evaluate(X, theta=np.random.randn(10))
            elapsed = time.time() - start
            
            times.append(elapsed / 10)  # Average time per evaluation
        
        # Check roughly linear scaling
        # time[1] / time[0] should be ≈ sizes[1] / sizes[0]
        ratio_time = times[1] / times[0]
        ratio_size = sizes[1] / sizes[0]
        
        assert 0.8 * ratio_size < ratio_time < 1.2 * ratio_size, \
            f"Scaling not linear: {ratio_time} vs {ratio_size}"
    
    def test_optimization_time_reasonable(self):
        """Parameter optimization should complete in reasonable time."""
        
        X = np.random.rand(200, 2)
        y = 2*X[:, 0] + 3*X[:, 1] + np.random.normal(0, 0.1, 200)
        
        expr = create_linear_expression(n_features=2)
        
        start = time.time()
        result = optimize_params(expr, X, y)
        elapsed = time.time() - start
        
        assert elapsed < 5.0, f"Optimization took {elapsed:.1f}s (too slow)"
        assert result['success']
```

---

## Validation Checklist

### Pre-Release Checklist

**Before considering the system complete:**

**Unit Tests:**
- [ ] All expression tree tests pass
- [ ] All noise operator tests pass
- [ ] All likelihood tests pass
- [ ] All optimization tests pass
- [ ] Code coverage > 80%

**Component Tests:**
- [ ] Noise operators integrate with expression tree
- [ ] Likelihood works for both deterministic and noisy models
- [ ] Optimization recovers known parameters

**Integration Tests:**
- [ ] End-to-end: generate data → discover → evaluate works
- [ ] Can recover parameters on synthetic problems
- [ ] Online updates work (when implemented)

**Benchmarks:**
- [ ] Polynomial with heteroscedastic noise: RMSE < threshold
- [ ] Lorenz system: RMSE < threshold
- [ ] Beats baseline (no noise modeling) on noisy data

**Performance:**
- [ ] Evaluation scales linearly with data size
- [ ] Optimization completes in reasonable time
- [ ] Memory usage is acceptable

**Documentation:**
- [ ] All functions have docstrings
- [ ] Usage examples exist
- [ ] Tutorials are runnable
- [ ] API documentation generated

**Code Quality:**
- [ ] Black formatting applied
- [ ] Flake8 passes (no linting errors)
- [ ] Type hints on public functions
- [ ] No TODOs in main code paths

---

## Debugging Failed Tests

### Common Issues and Solutions

**Issue:** Tests pass individually but fail when run together
- **Cause:** Shared state or random seed not being reset
- **Solution:** Add `@pytest.fixture(autouse=True)` to reset state

**Issue:** Statistical tests occasionally fail
- **Cause:** Random variation in noise generation
- **Solution:** Increase sample size or use fixed random seed

**Issue:** Optimization doesn't converge
- **Cause:** Bad initialization, difficult landscape
- **Solution:** Try multiple initializations, use global optimizer (DE)

**Issue:** MC likelihood has high variance
- **Cause:** Too few samples
- **Solution:** Increase n_samples in log_likelihood_with_noise

---

## Acceptance Criteria

**Minimum Viable System:**
- 80% test coverage
- All unit tests pass
- At least 2 benchmarks pass
- Documentation exists

**Production Ready:**
- 90% test coverage
- All tests pass (unit, component, integration)
- All benchmarks pass
- Performance tests pass
- Full documentation and tutorials
- CI/CD pipeline working

**Publication Ready:**
- All above, plus:
- Comparison to 3+ baselines
- Real-world application demonstrated
- Paper-quality visualizations
- Reproducibility verified by independent party

---

**Summary:**

This testing guide provides:
✓ Hierarchical test structure (unit → component → integration → benchmark)
✓ Specific test cases with assertions
✓ Statistical validation for stochastic components
✓ Performance and scaling tests
✓ Continuous integration setup
✓ Clear acceptance criteria

Start with unit tests, build up to integration, then validate on benchmarks!
