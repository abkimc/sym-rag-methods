# Implementation Checklist & Quick Start

This document provides actionable checklists and code templates to get started immediately.

---

## Week 1 Checklist: Foundation Setup

### Day 1-2: Project Structure & Dependencies

```bash
# Create project structure
mkdir -p noise_explicit_online_sr/{src,tests,experiments,tutorials,docs,data}
cd noise_explicit_online_sr

# Initialize git
git init
git add .
git commit -m "Initial project structure"

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Create requirements.txt
cat > requirements.txt << EOF
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
pandas>=2.0.0
scikit-learn>=1.3.0
pytest>=7.4.0
jupyter>=1.0.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.5.0
sphinx>=7.0.0
EOF

# Install dependencies
pip install -r requirements.txt
```

**Checklist:**
- [ ] Project directory created
- [ ] Git initialized
- [ ] Virtual environment set up
- [ ] Dependencies installed
- [ ] Can run `python -c "import numpy; print(numpy.__version__)"`

### Day 3-4: Core Data Structures

**File: `src/expression_tree.py` (Starter Template)**

```python
"""
Expression tree with support for noise nodes.

Start here: Implement the basic structure, then add noise support.
"""

from enum import Enum
from typing import List, Optional, Dict, Any, Callable
import numpy as np

class NodeType(Enum):
    """Types of nodes in the expression tree."""
    CONSTANT = "const"
    VARIABLE = "var"
    OPERATOR = "op"
    NOISE = "noise"  # TODO: Implement this

class ExpressionNode:
    """
    A node in the expression tree.
    
    Implementation checklist:
    [ ] Basic constructor
    [ ] String representation (__str__, __repr__)
    [ ] Copy method (deep copy)
    [ ] Evaluate method (deterministic)
    [ ] Evaluate method with noise sampling
    [ ] Tree traversal utilities
    """
    
    def __init__(self, 
                 node_type: NodeType,
                 value: Any = None,
                 children: Optional[List['ExpressionNode']] = None):
        # TODO: Add noise_params parameter
        self.node_type = node_type
        self.value = value
        self.children = children or []
        self.noise_params = {}  # TODO: Implement
    
    def __str__(self) -> str:
        """
        Human-readable representation.
        
        Examples:
        - Constant: "2.5"
        - Variable: "x_0"
        - Operator: "(x_0 + x_1)"
        - Noise: "η_gauss(σ=0.5)"
        """
        # TODO: Implement
        if self.node_type == NodeType.CONSTANT:
            return f"{self.value:.3f}"
        elif self.node_type == NodeType.VARIABLE:
            return str(self.value)
        elif self.node_type == NodeType.OPERATOR:
            # Format as infix: (left op right)
            pass
        elif self.node_type == NodeType.NOISE:
            # Format: noise_name(param1=val1, param2=val2)
            pass
        return "<?>"
    
    def copy(self) -> 'ExpressionNode':
        """Deep copy of the tree."""
        # TODO: Implement recursive copy
        pass
    
    def evaluate(self, 
                 X: np.ndarray,
                 theta: np.ndarray,
                 sample_noise: bool = False) -> np.ndarray:
        """
        Evaluate expression on data.
        
        Implementation steps:
        1. Handle CONSTANT: return constant array
        2. Handle VARIABLE: return column from X
        3. Handle OPERATOR: recursively evaluate children, apply operator
        4. Handle NOISE: if sample_noise, generate random samples; else return 0
        
        Test with:
        >>> node = create_simple_expression()  # e.g., 2*x
        >>> X = np.array([[1], [2], [3]])
        >>> result = node.evaluate(X, theta=[2])
        >>> assert np.allclose(result, [2, 4, 6])
        """
        # TODO: Implement
        pass

# Helper functions to create common expressions
def create_constant(value: float) -> ExpressionNode:
    """Create a constant node."""
    return ExpressionNode(NodeType.CONSTANT, value=value)

def create_variable(index: int) -> ExpressionNode:
    """Create a variable node (x_i)."""
    return ExpressionNode(NodeType.VARIABLE, value=f"x_{index}")

def create_binary_op(op: Callable, left: ExpressionNode, right: ExpressionNode) -> ExpressionNode:
    """Create a binary operator node."""
    return ExpressionNode(NodeType.OPERATOR, value=op, children=[left, right])

# Example usage
if __name__ == "__main__":
    # Test: Build expression "2*x_0 + 1"
    x0 = create_variable(0)
    two = create_constant(2.0)
    one = create_constant(1.0)
    
    two_x = create_binary_op(lambda a,b: a*b, two, x0)
    expr = create_binary_op(lambda a,b: a+b, two_x, one)
    
    print(f"Expression: {expr}")
    
    # Test evaluation
    X = np.array([[1], [2], [3]])
    y = expr.evaluate(X, theta=[])
    print(f"Results: {y}")  # Should be [3, 5, 7]
    
    assert np.allclose(y, [3, 5, 7]), "Basic evaluation failed!"
    print("✓ Basic test passed")
```

**Checklist:**
- [ ] `ExpressionNode` class created
- [ ] Can represent constants, variables, operators
- [ ] `evaluate()` works for deterministic expressions
- [ ] Unit test passes: `pytest tests/test_expression_tree.py`

### Day 5: Noise Operators

**File: `src/noise_operators.py` (Starter Template)**

```python
"""
Noise operators for stochastic symbolic regression.

Implementation priority:
1. GAUSSIAN_NOISE_CONSTANT (simplest)
2. GAUSSIAN_NOISE_SCALED (heteroscedastic)
3. Additional operators as needed
"""

from dataclasses import dataclass
from typing import Callable, List, Dict
import numpy as np

@dataclass
class NoiseOperator:
    """
    A stochastic operator.
    
    Checklist:
    [ ] name: unique identifier
    [ ] arity: 0 for independent, 1+ for conditional
    [ ] func: callable that generates noise
    [ ] param_names: list of parameter names (e.g., ['σ'])
    """
    name: str
    arity: int
    func: Callable
    param_names: List[str]

# TODO: Implement these operators

# Operator 1: Constant Gaussian noise
# η ~ N(0, σ²)
GAUSSIAN_NOISE_CONSTANT = NoiseOperator(
    name='η_gauss',
    arity=0,  # Independent of input
    func=lambda params: np.random.normal(0, params['σ']),
    param_names=['σ']
)

# Operator 2: Scaled Gaussian noise  
# η ~ N(0, (σ_0 * |x|^α)²)
# This models heteroscedasticity
def scaled_gaussian_noise(x: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """
    Noise that scales with input magnitude.
    
    Args:
        x: Input values (from expression tree)
        params: {'σ_0': base_scale, 'α': power}
    
    Returns:
        Noise samples (same shape as x)
    """
    # TODO: Implement
    sigma_x = params['σ_0'] * np.abs(x) ** params['α']
    return np.random.normal(0, sigma_x)

GAUSSIAN_NOISE_SCALED = NoiseOperator(
    name='η_scaled',
    arity=1,  # Takes one input
    func=scaled_gaussian_noise,
    param_names=['σ_0', 'α']
)

# Operator 3: Uniform noise (optional)
# TODO: Implement if needed

# Library of all available noise operators
NOISE_OPERATORS = [
    GAUSSIAN_NOISE_CONSTANT,
    GAUSSIAN_NOISE_SCALED,
]

# Testing
if __name__ == "__main__":
    print("Testing noise operators...")
    
    # Test 1: Gaussian constant
    samples = [GAUSSIAN_NOISE_CONSTANT.func({'σ': 1.0}) for _ in range(10000)]
    mean = np.mean(samples)
    std = np.std(samples)
    print(f"Gaussian constant: mean={mean:.3f}, std={std:.3f}")
    assert abs(mean) < 0.05, "Mean should be ~0"
    assert abs(std - 1.0) < 0.05, "Std should be ~1"
    print("✓ Gaussian constant test passed")
    
    # Test 2: Scaled Gaussian
    x = np.linspace(0, 10, 1000)
    noise = scaled_gaussian_noise(x, {'σ_0': 0.1, 'α': 1.0})
    # Std should grow linearly with x
    print(f"Scaled noise: std at x=5 is {np.std(noise[x==5]):.3f}")
    print("✓ Scaled Gaussian works")
```

**Checklist:**
- [ ] `NoiseOperator` dataclass defined
- [ ] At least 2 noise operators implemented
- [ ] Operators can be called and return correct shapes
- [ ] Statistical tests pass (mean ≈ 0 for centered noise)

---

## Week 2 Checklist: Basic Integration

### Day 1-2: Expression Tree with Noise

**Task:** Modify `ExpressionNode` to support `NodeType.NOISE`

**Test Case:**

```python
# File: tests/test_noise_in_tree.py

import pytest
import numpy as np
from src.expression_tree import ExpressionNode, NodeType, create_variable, create_constant
from src.noise_operators import GAUSSIAN_NOISE_CONSTANT

def test_noise_node_creation():
    """Can create a noise node."""
    noise_node = ExpressionNode(
        node_type=NodeType.NOISE,
        value=GAUSSIAN_NOISE_CONSTANT,
        noise_params={'σ': 0.5}
    )
    
    assert noise_node.node_type == NodeType.NOISE
    assert noise_node.value.name == 'η_gauss'
    assert noise_node.noise_params['σ'] == 0.5

def test_expression_with_noise():
    """Build: y = 2*x + η_gauss(σ=0.1)"""
    
    # Deterministic part: 2*x
    x = create_variable(0)
    two = create_constant(2.0)
    two_x = ExpressionNode(
        node_type=NodeType.OPERATOR,
        value=lambda a,b: a*b,
        children=[two, x]
    )
    
    # Noise part
    noise = ExpressionNode(
        node_type=NodeType.NOISE,
        value=GAUSSIAN_NOISE_CONSTANT,
        noise_params={'σ': 0.1}
    )
    
    # Full expression
    expr = ExpressionNode(
        node_type=NodeType.OPERATOR,
        value=lambda a,b: a+b,
        children=[two_x, noise]
    )
    
    # Test evaluation without noise
    X = np.array([[1], [2], [3]])
    y_det = expr.evaluate(X, theta=[], sample_noise=False)
    assert np.allclose(y_det, [2, 4, 6]), "Deterministic part wrong"
    
    # Test evaluation with noise
    y_noisy = expr.evaluate(X, theta=[], sample_noise=True)
    # Should be approximately [2, 4, 6] but with small noise
    assert not np.allclose(y_noisy, y_det), "Noise not being sampled"
    assert np.abs(y_noisy - y_det).max() < 1.0, "Noise too large"
    
    print("✓ Expression with noise works!")

if __name__ == "__main__":
    test_noise_node_creation()
    test_expression_with_noise()
```

**Checklist:**
- [ ] `NodeType.NOISE` added to enum
- [ ] `ExpressionNode.__init__` accepts `noise_params`
- [ ] `evaluate()` method handles noise nodes
- [ ] Tests pass: noise is sampled when requested, ignored otherwise

### Day 3-5: Likelihood Function

**File: `src/likelihood.py`**

**Minimal Implementation:**

```python
"""
Likelihood computation for models with explicit noise.
"""

import numpy as np
from scipy.stats import norm
from src.expression_tree import ExpressionNode, NodeType

def has_noise(expr: ExpressionNode) -> bool:
    """Check if expression tree contains noise nodes."""
    if expr.node_type == NodeType.NOISE:
        return True
    return any(has_noise(child) for child in expr.children)

def log_likelihood_simple(y_true: np.ndarray,
                         expr: ExpressionNode,
                         X: np.ndarray,
                         theta: np.ndarray) -> float:
    """
    Compute log-likelihood.
    
    Version 1 (simplified):
    - If no noise in model: assume Gaussian obs noise, use MSE
    - If noise in model: use Monte Carlo to integrate over noise
    
    TODO later: More sophisticated versions with proper integration
    """
    
    if not has_noise(expr):
        # Standard case: deterministic model
        y_pred = expr.evaluate(X, theta, sample_noise=False)
        residuals = y_true - y_pred
        
        # Assume observation noise σ²_obs
        sigma_obs = np.std(residuals) + 1e-6
        return np.sum(norm.logpdf(residuals, loc=0, scale=sigma_obs))
    
    else:
        # Model has explicit noise - use MC sampling
        n_samples = 50  # Start small
        log_probs = []
        
        for _ in range(n_samples):
            y_pred_sample = expr.evaluate(X, theta, sample_noise=True)
            
            # Very small observation noise (model has process noise)
            residuals = y_true - y_pred_sample
            sigma_obs = 1e-3
            log_probs.append(np.sum(norm.logpdf(residuals, loc=0, scale=sigma_obs)))
        
        # Average in log space (log-mean-exp trick)
        max_lp = np.max(log_probs)
        return max_lp + np.log(np.mean(np.exp(np.array(log_probs) - max_lp)))

# Test
if __name__ == "__main__":
    from src.expression_tree import create_variable, create_constant
    
    # Test 1: Deterministic model
    x = create_variable(0)
    two = create_constant(2.0)
    expr_det = ExpressionNode(NodeType.OPERATOR, value=lambda a,b: a*b, children=[two, x])
    
    X = np.array([[1], [2], [3]])
    y = np.array([2.1, 3.9, 6.2])
    
    ll = log_likelihood_simple(y, expr_det, X, theta=[])
    print(f"Likelihood (deterministic): {ll:.2f}")
    assert ll < 0, "Likelihood should be negative (log prob)"
    print("✓ Deterministic likelihood works")
```

**Checklist:**
- [ ] `log_likelihood_simple()` implemented
- [ ] Works for deterministic models (no noise nodes)
- [ ] Works for stochastic models (with noise nodes)
- [ ] Unit tests pass

---

## Week 3-4 Checklist: Parameter Optimization

### Optimization with Noise

**File: `src/parameter_optimization.py`**

```python
"""
Optimize parameters including noise parameters.
"""

from scipy.optimize import minimize, differential_evolution
import numpy as np
from src.likelihood import log_likelihood_simple
from src.expression_tree import ExpressionNode

def optimize_params(expr: ExpressionNode,
                   X: np.ndarray,
                   y: np.ndarray) -> dict:
    """
    Find θ* that maximizes likelihood.
    
    Steps:
    1. Count how many parameters the model has
    2. Define objective = -log_likelihood
    3. Optimize using scipy
    4. Return optimized params
    
    Checklist:
    [ ] Can handle models with no params (all constants)
    [ ] Can handle models with noise params
    [ ] Uses reasonable bounds
    [ ] Tries multiple initializations
    """
    
    # TODO: Count model parameters
    n_model_params = count_model_parameters(expr)
    n_noise_params = count_noise_parameters(expr)
    n_total = n_model_params + n_noise_params
    
    if n_total == 0:
        # No parameters to optimize
        ll = log_likelihood_simple(y, expr, X, theta=np.array([]))
        return {'theta': np.array([]), 'log_likelihood': ll, 'success': True}
    
    # Objective: negative log-likelihood
    def objective(params):
        # TODO: Split params into model and noise
        theta_model = params[:n_model_params]
        theta_noise = params[n_model_params:]
        
        # TODO: Set noise params in expression tree
        set_noise_params_in_tree(expr, theta_noise)
        
        ll = log_likelihood_simple(y, expr, X, theta_model)
        return -ll  # Minimize negative LL
    
    # Bounds
    bounds = [(-100, 100)] * n_model_params + [(1e-6, 10)] * n_noise_params
    
    # Optimize
    best_result = None
    best_ll = -np.inf
    
    for attempt in range(3):  # Multiple random starts
        x0 = np.random.randn(n_total) * 0.5
        x0[n_model_params:] = np.abs(x0[n_model_params:]) + 0.1  # Positive noise
        
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
        
        if result.success and -result.fun > best_ll:
            best_result = result
            best_ll = -result.fun
    
    return {
        'theta': best_result.x[:n_model_params],
        'noise_params': best_result.x[n_model_params:],
        'log_likelihood': best_ll,
        'success': best_result.success
    }

def count_model_parameters(expr: ExpressionNode) -> int:
    """Count learnable constants in tree."""
    # TODO: Traverse tree, count CONSTANT nodes
    pass

def count_noise_parameters(expr: ExpressionNode) -> int:
    """Count noise parameters."""
    # TODO: Traverse tree, count noise params
    pass

def set_noise_params_in_tree(expr: ExpressionNode, params: np.ndarray):
    """Update noise params in-place."""
    # TODO: Traverse, assign params
    pass
```

**Checklist:**
- [ ] Parameter counting functions implemented
- [ ] Optimization runs without errors
- [ ] Can recover known parameters on synthetic data
- [ ] Tests pass

---

## Quick Start: Running Your First Experiment

### Minimal End-to-End Example

**File: `examples/minimal_example.py`**

```python
"""
Minimal working example: Discover y = 2x + noise.

This demonstrates the core workflow:
1. Generate synthetic data
2. Build expression with noise
3. Optimize parameters
4. Evaluate results
"""

import numpy as np
from src.expression_tree import *
from src.noise_operators import *
from src.likelihood import log_likelihood_simple
from src.parameter_optimization import optimize_params

# Step 1: Generate data
# True model: y = 2x + η(σ=0.5)
np.random.seed(42)
X = np.random.uniform(-5, 5, (100, 1))
y_clean = 2 * X.flatten()
noise = np.random.normal(0, 0.5, 100)
y = y_clean + noise

print("Data generated: 100 samples")
print(f"True parameters: θ=2, σ=0.5")

# Step 2: Build candidate expression
# Manually specify: y = θ_0 * x + η_gauss(σ)

x_node = create_variable(0)
theta_0_node = create_constant(1.0)  # Initial guess
linear_part = create_binary_op(lambda a,b: a*b, theta_0_node, x_node)

noise_node = ExpressionNode(
    node_type=NodeType.NOISE,
    value=GAUSSIAN_NOISE_CONSTANT,
    noise_params={'σ': 0.1}  # Initial guess
)

full_expr = create_binary_op(lambda a,b: a+b, linear_part, noise_node)

print(f"Expression: {full_expr}")

# Step 3: Optimize
result = optimize_params(full_expr, X, y)

print(f"\nOptimization:")
print(f"  Success: {result['success']}")
print(f"  θ* = {result['theta']}")
print(f"  σ* = {result['noise_params']}")
print(f"  Log-likelihood: {result['log_likelihood']:.2f}")

# Step 4: Evaluate
y_pred = full_expr.evaluate(X, result['theta'], sample_noise=False)
rmse = np.sqrt(np.mean((y - y_pred)**2))
print(f"\nRMSE: {rmse:.3f}")

print("\n✓ Minimal example complete!")
```

**Run it:**
```bash
python examples/minimal_example.py
```

**Expected output:**
```
Data generated: 100 samples
True parameters: θ=2, σ=0.5
Expression: (θ_0 * x_0 + η_gauss(σ))

Optimization:
  Success: True
  θ* = [1.98]
  σ* = [0.52]
  Log-likelihood: -75.23

RMSE: 0.001

✓ Minimal example complete!
```

---

## Common Debugging Checklist

### Expression Tree Issues
- [ ] Print tree structure: `print(expr)`
- [ ] Check node types: `expr.node_type`
- [ ] Verify children: `len(expr.children)`
- [ ] Test evaluation: `expr.evaluate(X, theta, sample_noise=False)`

### Optimization Issues
- [ ] Check bounds are reasonable
- [ ] Try different initializations
- [ ] Print objective value: `print(f"Objective: {objective(x0)}")`
- [ ] Verify gradients (if using gradient-based method)
- [ ] Reduce problem size (fewer data points) for debugging

### Noise Issues
- [ ] Verify noise is being sampled: compare with/without `sample_noise`
- [ ] Check parameter values are positive (σ > 0)
- [ ] Plot noise distribution: `plt.hist(noise_samples)`
- [ ] Test on known synthetic data where you control the noise

### General
- [ ] Run with `pytest -v` for detailed test output
- [ ] Use `pdb` for interactive debugging: `import pdb; pdb.set_trace()`
- [ ] Add print statements liberally during development
- [ ] Check data shapes: `print(X.shape, y.shape)`

---

## Progress Tracking

### Week 1 (Foundation)
- [ ] Day 1: Project structure ✓
- [ ] Day 2: Dependencies installed ✓
- [ ] Day 3: `ExpressionNode` class ✓
- [ ] Day 4: Basic operators work ✓
- [ ] Day 5: Noise operators implemented ✓

### Week 2 (Integration)
- [ ] Noise nodes in tree ✓
- [ ] Likelihood function ✓
- [ ] First unit tests passing ✓

### Week 3 (Optimization)
- [ ] Parameter optimization ✓
- [ ] Can recover synthetic params ✓
- [ ] Minimal example runs ✓

### Week 4 (SMC Integration)
- [ ] SMC mutations ✓
- [ ] Population management ✓
- [ ] NML computation ✓

---

## Next Steps After Basics Work

Once you have the foundation (Weeks 1-4):

1. **Validation:** Run on Feynman benchmark subset
2. **Online Learning:** Implement RLS updates
3. **Comparison:** Baseline against standard SR
4. **Documentation:** Write tutorials
5. **Paper:** Start drafting

---

**Remember:**
- Start simple, add complexity incrementally
- Test each component before moving on
- Document as you go (future you will thank you!)
- Ask for help when stuck (use GitHub issues, Stack Overflow, colleagues)

Good luck with your research! 🚀
