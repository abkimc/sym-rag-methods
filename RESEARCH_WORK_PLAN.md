# Research Work Plan: Noise-Explicit Online Symbolic Regression

## Executive Summary

This plan outlines the development of a symbolic regression system that:
1. **Explicitly models noise** as stochastic components within equations (not just residuals)
2. **Updates online** as new data streams in, adapting both structure and parameters
3. **Builds on existing SMC-SR framework** to maintain Bayesian uncertainty quantification

---

## Phase 1: Foundation & Architecture Design (Weeks 1-3)

### 1.1 Literature Review & Method Selection

**Objectives:**
- Deep dive into noise modeling approaches (Schmidt & Lipson 2007, NRSR 2025)
- Study online learning frameworks (BRSL 2026, Sym-Q 2024)
- Identify integration points with existing SMC-SR

**Deliverables:**
- Literature matrix comparing approaches
- Technical decision document
- Architecture diagrams

**Key Decisions to Make:**
```
Decision 1: Noise Representation
├── Option A: Stochastic symbols (η_i, ξ_i) in expression tree
├── Option B: Parametric noise models (Gaussian, Student-t, etc.)
└── Recommendation: Hybrid - start with Gaussian, extend to symbols

Decision 2: Online Update Mechanism
├── Option A: Recursive Least Squares (RLS) for coefficients only
├── Option B: Bayesian filtering (Kalman/particle) for θ updates
├── Option C: SMC re-weighting with forgetting factor
└── Recommendation: Option C + Option B for parameter tracking
```

### 1.2 Architecture Design

**Core Components:**

```
┌─────────────────────────────────────────────────────┐
│                 Online SR System                     │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌────────────────────────────────────────────┐   │
│  │   1. Expression Discovery Module           │   │
│  │      (SMC-SR with noise-aware operators)   │   │
│  └────────────────────────────────────────────┘   │
│                      │                              │
│                      ▼                              │
│  ┌────────────────────────────────────────────┐   │
│  │   2. Noise Modeling Layer                  │   │
│  │      • Noise symbol injection              │   │
│  │      • Heteroscedastic variance models     │   │
│  └────────────────────────────────────────────┘   │
│                      │                              │
│                      ▼                              │
│  ┌────────────────────────────────────────────┐   │
│  │   3. Parameter Estimation                  │   │
│  │      • Maximum Likelihood (θ*)             │   │
│  │      • Noise parameters (σ²(x), etc.)      │   │
│  └────────────────────────────────────────────┘   │
│                      │                              │
│                      ▼                              │
│  ┌────────────────────────────────────────────┐   │
│  │   4. Online Update Engine                  │   │
│  │      • RLS for θ updates                   │   │
│  │      • SMC re-weighting with λ_forget      │   │
│  │      • Drift detection & re-discovery      │   │
│  └────────────────────────────────────────────┘   │
│                      │                              │
│                      ▼                              │
│  ┌────────────────────────────────────────────┐   │
│  │   5. Model Selection & Validation          │   │
│  │      • Cross-validation on windows         │   │
│  │      • Information criteria (AIC, BIC)     │   │
│  └────────────────────────────────────────────┘   │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## Phase 2: Noise Modeling Implementation (Weeks 4-7)

### 2.1 Noise-Aware Expression Tree

**Objective:** Extend expression representation to include noise components

**Implementation Steps:**

#### Step 2.1.1: Extend Grammar with Noise Operators

```python
# File: src/noise_operators.py

"""
Noise-aware operators for symbolic regression.

This module extends the standard operator set with stochastic elements
that can model heteroscedastic or structured noise.
"""

from dataclasses import dataclass
from typing import Callable, List
import numpy as np

@dataclass
class NoiseOperator:
    """
    Represents a stochastic operator in the expression tree.
    
    Attributes:
        name: Operator identifier (e.g., 'gaussian_noise', 'uniform_noise')
        arity: Number of inputs (0 for pure random, 1+ for conditional)
        func: Function that generates noise given inputs
        param_names: Names of learnable noise parameters
    """
    name: str
    arity: int
    func: Callable
    param_names: List[str]
    
# Example 1: Additive Gaussian noise with constant variance
GAUSSIAN_NOISE_CONSTANT = NoiseOperator(
    name='η_gauss',
    arity=0,  # Pure random, no dependence
    func=lambda params: np.random.normal(0, params['σ']),
    param_names=['σ']
)

# Example 2: Heteroscedastic noise (variance grows with x)
GAUSSIAN_NOISE_SCALED = NoiseOperator(
    name='η_scaled',
    arity=1,  # Takes one input to scale noise
    func=lambda x, params: np.random.normal(0, params['σ_0'] * np.abs(x)**params['α']),
    param_names=['σ_0', 'α']
)

# Example 3: Uniform noise
UNIFORM_NOISE = NoiseOperator(
    name='ξ_uniform',
    arity=0,
    func=lambda params: np.random.uniform(-params['a'], params['a']),
    param_names=['a']
)

# Complete operator library
NOISE_OPERATORS = [
    GAUSSIAN_NOISE_CONSTANT,
    GAUSSIAN_NOISE_SCALED,
    UNIFORM_NOISE,
]
```

#### Step 2.1.2: Modify Expression Tree Structure

```python
# File: src/expression_tree.py

"""
Expression tree that supports both deterministic and stochastic nodes.
"""

from enum import Enum
from typing import Union, Dict, Any

class NodeType(Enum):
    """Type of node in expression tree."""
    CONSTANT = "const"
    VARIABLE = "var"
    OPERATOR = "op"
    NOISE = "noise"  # NEW: stochastic node

class ExpressionNode:
    """
    Node in symbolic expression tree.
    
    Can represent:
    - Constants: θ_i
    - Variables: x_j  
    - Operators: +, -, *, /, sin, exp, etc.
    - Noise: η_gauss, ξ_uniform, etc. (NEW)
    """
    
    def __init__(self, 
                 node_type: NodeType, 
                 value: Any = None,
                 children: List['ExpressionNode'] = None,
                 noise_params: Dict[str, float] = None):
        """
        Initialize expression node.
        
        Args:
            node_type: Type of node (constant, variable, operator, noise)
            value: For constants (float), variables (str), operators (callable), 
                   or noise (NoiseOperator)
            children: Child nodes for operators/noise
            noise_params: Parameters for noise operators (σ, α, etc.)
        """
        self.node_type = node_type
        self.value = value
        self.children = children or []
        self.noise_params = noise_params or {}
        
    def evaluate(self, X: np.ndarray, 
                 theta: np.ndarray,
                 sample_noise: bool = True) -> np.ndarray:
        """
        Evaluate expression on data.
        
        Args:
            X: Input data (n_samples, n_features)
            theta: Parameter vector
            sample_noise: If True, sample stochastic nodes; 
                         if False, return expected value (0 for noise)
        
        Returns:
            Evaluated expression (n_samples,)
            
        Example:
            # Expression: y = θ_0 * x_1^2 + η_gauss(σ=0.1)
            node = create_expression_tree()
            y_pred = node.evaluate(X, theta=[2.0], sample_noise=True)
        """
        if self.node_type == NodeType.CONSTANT:
            return np.full(len(X), self.value)
            
        elif self.node_type == NodeType.VARIABLE:
            var_idx = int(self.value.split('_')[1])  # 'x_0' -> 0
            return X[:, var_idx]
            
        elif self.node_type == NodeType.OPERATOR:
            child_results = [child.evaluate(X, theta, sample_noise) 
                           for child in self.children]
            return self.value(*child_results)
            
        elif self.node_type == NodeType.NOISE:
            if not sample_noise:
                return np.zeros(len(X))  # Expected value of noise is 0
            
            # Sample noise for each data point
            noise_op = self.value  # NoiseOperator instance
            
            if noise_op.arity == 0:
                # Independent noise
                return np.array([noise_op.func(self.noise_params) 
                               for _ in range(len(X))])
            else:
                # Conditional noise (depends on inputs)
                child_results = [child.evaluate(X, theta, sample_noise=False) 
                               for child in self.children]
                return noise_op.func(*child_results, params=self.noise_params)
```

#### Step 2.1.3: Noise-Aware Likelihood Function

```python
# File: src/likelihood.py

"""
Likelihood functions that account for explicit noise modeling.
"""

import numpy as np
from scipy.stats import norm, t as student_t

def log_likelihood_with_noise(y_true: np.ndarray,
                               expression: ExpressionNode,
                               X: np.ndarray,
                               theta: np.ndarray,
                               n_samples: int = 100) -> float:
    """
    Compute log-likelihood when noise is explicit in the model.
    
    Uses Monte Carlo integration to marginalize over noise:
    p(y|X,θ,M) = ∫ p(y|f(X,θ,η)) p(η) dη ≈ (1/S) Σ p(y|f(X,θ,η_s))
    
    Args:
        y_true: Observed outputs (n_data,)
        expression: Expression tree with potential noise nodes
        X: Input features (n_data, n_features)
        theta: Parameter vector
        n_samples: Number of Monte Carlo samples for noise integration
        
    Returns:
        Log-likelihood value
        
    Example:
        # Model: y = θ_0*x + θ_1 + η_gauss(σ=θ_2)
        # We need to integrate over the noise distribution
        ll = log_likelihood_with_noise(y, expr, X, theta=[2.0, 1.0, 0.5])
    """
    n_data = len(y_true)
    
    # Check if expression has noise nodes
    has_noise = _tree_has_noise(expression)
    
    if not has_noise:
        # Standard deterministic model - use simple MSE
        y_pred = expression.evaluate(X, theta, sample_noise=False)
        residuals = y_true - y_pred
        
        # Assume Gaussian observation noise with learned σ
        # (last theta parameter if provided, else estimate)
        if len(theta) > _count_params(expression):
            sigma_obs = theta[-1]
        else:
            sigma_obs = np.std(residuals)
            
        return np.sum(norm.logpdf(residuals, loc=0, scale=sigma_obs))
    
    else:
        # Model includes explicit noise - use MC integration
        log_probs = []
        
        for _ in range(n_samples):
            # Sample one realization of the noisy expression
            y_pred_sample = expression.evaluate(X, theta, sample_noise=True)
            
            # Tiny observation noise (model already has process noise)
            sigma_obs = 1e-6
            log_probs.append(np.sum(norm.logpdf(y_true - y_pred_sample, 
                                                loc=0, scale=sigma_obs)))
        
        # Log-sum-exp trick for numerical stability
        max_log_prob = np.max(log_probs)
        return max_log_prob + np.log(np.mean(np.exp(np.array(log_probs) - max_log_prob)))

def _tree_has_noise(node: ExpressionNode) -> bool:
    """Recursively check if tree contains noise nodes."""
    if node.node_type == NodeType.NOISE:
        return True
    return any(_tree_has_noise(child) for child in node.children)

def _count_params(node: ExpressionNode) -> int:
    """Count number of parameters in deterministic part."""
    # Implementation: traverse tree, count constants
    pass
```

### 2.2 Noise Parameter Optimization

**Objective:** Learn noise parameters jointly with model parameters

```python
# File: src/parameter_optimization.py

"""
Joint optimization of model and noise parameters.
"""

from scipy.optimize import minimize, differential_evolution
import numpy as np

def optimize_parameters_with_noise(expression: ExpressionNode,
                                   X: np.ndarray,
                                   y: np.ndarray,
                                   method: str = 'L-BFGS-B') -> Dict[str, Any]:
    """
    Optimize both model parameters (θ) and noise parameters jointly.
    
    For expression: y = θ_0*x^θ_1 + η_gauss(σ)
    Optimizes: θ_0, θ_1, σ to maximize likelihood
    
    Args:
        expression: Expression tree with noise nodes
        X: Input data (n_samples, n_features)
        y: Target outputs (n_samples,)
        method: Optimization method ('L-BFGS-B', 'DE', 'CMA-ES')
        
    Returns:
        Dictionary with:
            - 'theta': Optimized model parameters
            - 'noise_params': Optimized noise parameters  
            - 'log_likelihood': Final log-likelihood
            - 'success': Whether optimization converged
            
    Example:
        expr = create_expression("θ_0*x + η_gauss(σ)")
        result = optimize_parameters_with_noise(expr, X, y)
        print(f"Learned noise std: {result['noise_params']['σ']}")
    """
    
    # Extract parameter structure
    n_model_params = _count_model_parameters(expression)
    n_noise_params = _count_noise_parameters(expression)
    n_total = n_model_params + n_noise_params
    
    # Define objective (negative log-likelihood)
    def objective(params):
        theta_model = params[:n_model_params]
        theta_noise = params[n_model_params:]
        
        # Update noise parameters in expression tree
        _set_noise_parameters(expression, theta_noise)
        
        # Compute likelihood
        ll = log_likelihood_with_noise(y, expression, X, theta_model)
        return -ll  # Minimize negative LL
    
    # Set bounds
    # Model parameters: typically [-100, 100]
    # Noise parameters: typically [1e-6, 10] (must be positive)
    bounds = (
        [(-100, 100)] * n_model_params +  # Model params
        [(1e-6, 10)] * n_noise_params      # Noise params (σ, α, etc.)
    )
    
    # Initialize
    x0 = np.random.randn(n_total) * 0.1
    x0[n_model_params:] = np.abs(x0[n_model_params:]) + 0.1  # Ensure positive noise
    
    # Optimize
    if method == 'L-BFGS-B':
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
    elif method == 'DE':
        result = differential_evolution(objective, bounds, seed=42)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Parse results
    theta_opt = result.x[:n_model_params]
    noise_params_opt = result.x[n_model_params:]
    
    return {
        'theta': theta_opt,
        'noise_params': _parse_noise_params(expression, noise_params_opt),
        'log_likelihood': -result.fun,
        'success': result.success
    }

def _count_model_parameters(expr: ExpressionNode) -> int:
    """Count learnable constants in expression."""
    # Traverse tree, count CONSTANT nodes
    pass

def _count_noise_parameters(expr: ExpressionNode) -> int:
    """Count noise parameters across all noise nodes."""
    # Traverse tree, sum len(node.noise_params) for NOISE nodes
    pass

def _set_noise_parameters(expr: ExpressionNode, params: np.ndarray):
    """Update noise parameters in tree."""
    # Traverse, assign params to each NOISE node
    pass

def _parse_noise_params(expr: ExpressionNode, params: np.ndarray) -> Dict:
    """Convert param array to named dict."""
    # Return {'σ': params[0], 'α': params[1], ...}
    pass
```

### 2.3 Testing & Validation

**Test Cases:**

```python
# File: tests/test_noise_modeling.py

"""
Unit tests for noise-aware symbolic regression.
"""

import numpy as np
import pytest

def test_gaussian_noise_operator():
    """Test that Gaussian noise has correct distribution."""
    op = GAUSSIAN_NOISE_CONSTANT
    params = {'σ': 1.0}
    
    samples = [op.func(params) for _ in range(10000)]
    
    assert np.abs(np.mean(samples)) < 0.05  # Mean ≈ 0
    assert np.abs(np.std(samples) - 1.0) < 0.05  # Std ≈ 1
    
def test_heteroscedastic_noise():
    """Test noise that scales with input."""
    # True model: y = 2x + η(σ=0.1*|x|)
    np.random.seed(42)
    X = np.linspace(-5, 5, 100).reshape(-1, 1)
    
    noise_std = 0.1 * np.abs(X.flatten())
    noise = np.random.normal(0, noise_std)
    y = 2*X.flatten() + noise
    
    # Build expression with heteroscedastic noise
    expr = create_heteroscedastic_model()
    
    result = optimize_parameters_with_noise(expr, X, y)
    
    # Should recover θ_0≈2, σ_0≈0.1, α≈1
    assert np.abs(result['theta'][0] - 2.0) < 0.2
    assert np.abs(result['noise_params']['σ_0'] - 0.1) < 0.05
    assert np.abs(result['noise_params']['α'] - 1.0) < 0.2

def test_noise_vs_deterministic_likelihood():
    """Verify likelihood computation differs for noisy models."""
    # Create two versions: deterministic and noisy
    expr_det = create_expression("θ_0*x")
    expr_noise = create_expression("θ_0*x + η_gauss(σ)")
    
    X = np.random.randn(50, 1)
    y = 2*X.flatten() + np.random.normal(0, 0.5, 50)
    
    ll_det = log_likelihood_with_noise(y, expr_det, X, theta=[2.0])
    ll_noise = log_likelihood_with_noise(y, expr_noise, X, 
                                         theta=[2.0], 
                                         noise_params={'σ': 0.5})
    
    # Noisy model should have higher likelihood (captures noise source)
    assert ll_noise > ll_det
```

**Deliverables:**
- `src/noise_operators.py` - Noise operator library
- `src/expression_tree.py` - Extended tree with noise nodes  
- `src/likelihood.py` - Noise-aware likelihood
- `src/parameter_optimization.py` - Joint optimization
- `tests/test_noise_modeling.py` - Comprehensive tests
- Documentation with usage examples

---

## Phase 3: SMC Integration for Noise Models (Weeks 8-10)

### 3.1 Modify SMC Proposal Distribution

**Objective:** Allow SMC to propose expressions with noise operators

#### Step 3.1.1: Extend Mutation Operators

```python
# File: src/smc_mutations.py

"""
Mutation operators for SMC that can add/remove/modify noise nodes.
"""

import random
from typing import List

class NoiseMutation:
    """Mutations that introduce or modify noise in expressions."""
    
    @staticmethod
    def insert_noise_node(expr: ExpressionNode, 
                         noise_op: NoiseOperator,
                         position: str = 'additive') -> ExpressionNode:
        """
        Insert a noise node into the expression.
        
        Positions:
        - 'additive': y = f(x) + η  (most common)
        - 'multiplicative': y = f(x) * (1 + η)
        - 'internal': Replace a subtree with f'(x) + η
        
        Args:
            expr: Current expression tree
            noise_op: Noise operator to add
            position: Where to insert noise
            
        Returns:
            New expression with noise node
            
        Example:
            # Before: y = θ_0*x + θ_1
            # After:  y = θ_0*x + θ_1 + η_gauss(σ)
            expr_new = NoiseMutation.insert_noise_node(
                expr, GAUSSIAN_NOISE_CONSTANT, position='additive'
            )
        """
        if position == 'additive':
            # Create: old_expr + noise
            noise_node = ExpressionNode(
                node_type=NodeType.NOISE,
                value=noise_op,
                noise_params={p: 0.1 for p in noise_op.param_names}  # Init
            )
            
            return ExpressionNode(
                node_type=NodeType.OPERATOR,
                value=lambda a, b: a + b,
                children=[expr.copy(), noise_node]
            )
            
        elif position == 'multiplicative':
            # Create: old_expr * (1 + noise)
            noise_node = ExpressionNode(
                node_type=NodeType.NOISE,
                value=noise_op,
                noise_params={p: 0.1 for p in noise_op.param_names}
            )
            one_node = ExpressionNode(node_type=NodeType.CONSTANT, value=1.0)
            one_plus_noise = ExpressionNode(
                node_type=NodeType.OPERATOR,
                value=lambda a, b: a + b,
                children=[one_node, noise_node]
            )
            
            return ExpressionNode(
                node_type=NodeType.OPERATOR,
                value=lambda a, b: a * b,
                children=[expr.copy(), one_plus_noise]
            )
            
        elif position == 'internal':
            # Replace random subtree with subtree + noise
            # (More complex - find insertion point, etc.)
            pass
    
    @staticmethod
    def remove_noise_node(expr: ExpressionNode) -> ExpressionNode:
        """
        Remove a noise node from expression.
        
        Example:
            # Before: y = θ_0*x + η_gauss
            # After:  y = θ_0*x
        """
        # Traverse tree, find NOISE nodes, remove them
        return _prune_noise_recursive(expr)
    
    @staticmethod
    def mutate_noise_params(expr: ExpressionNode, 
                           mutation_rate: float = 0.1) -> ExpressionNode:
        """
        Perturb noise parameters (σ, α, etc.) slightly.
        
        Args:
            expr: Expression with noise nodes
            mutation_rate: Fraction to perturb (e.g., 0.1 = ±10%)
        """
        expr_new = expr.copy()
        _perturb_noise_params_recursive(expr_new, mutation_rate)
        return expr_new

def _prune_noise_recursive(node: ExpressionNode) -> ExpressionNode:
    """Helper to remove noise nodes from tree."""
    # If this is a noise node, return None (to be pruned by parent)
    # Else recursively process children
    pass

def _perturb_noise_params_recursive(node: ExpressionNode, rate: float):
    """Helper to mutate noise parameters."""
    if node.node_type == NodeType.NOISE:
        for param_name in node.noise_params:
            old_val = node.noise_params[param_name]
            node.noise_params[param_name] = old_val * (1 + random.gauss(0, rate))
    
    for child in node.children:
        _perturb_noise_params_recursive(child, rate)
```

#### Step 3.1.2: Update SMC Proposal Function

```python
# File: src/smc_sr.py (modifications to existing)

"""
SMC-SR with noise-aware proposals.
"""

def propose_new_expression(current_expr: ExpressionNode,
                          mutation_probs: Dict[str, float]) -> ExpressionNode:
    """
    Propose a new expression via mutation.
    
    Mutations now include noise operations:
    - Standard: add/remove node, change operator, etc.
    - Noise-specific: insert_noise, remove_noise, mutate_noise_params
    
    Args:
        current_expr: Current expression in population
        mutation_probs: Dict of mutation type -> probability
                       e.g., {'add_node': 0.3, 'insert_noise': 0.1, ...}
    
    Returns:
        Mutated expression
    """
    mutation_type = random.choices(
        list(mutation_probs.keys()),
        weights=list(mutation_probs.values())
    )[0]
    
    if mutation_type == 'insert_noise':
        # Add a noise node
        noise_op = random.choice(NOISE_OPERATORS)
        return NoiseMutation.insert_noise_node(current_expr, noise_op)
        
    elif mutation_type == 'remove_noise':
        # Remove noise if present
        return NoiseMutation.remove_noise_node(current_expr)
        
    elif mutation_type == 'mutate_noise_params':
        # Perturb noise parameters
        return NoiseMutation.mutate_noise_params(current_expr)
        
    else:
        # Standard mutations (add node, change op, etc.)
        return standard_mutation(current_expr, mutation_type)
```

### 3.2 Noise-Aware NML Calculation

**Objective:** Compute normalized marginal likelihood accounting for noise

```python
# File: src/nml_noise.py

"""
Normalized Marginal Likelihood for models with explicit noise.
"""

def compute_nml_with_noise(expression: ExpressionNode,
                           X_train: np.ndarray,
                           y_train: np.ndarray,
                           n_attempts: int = 10) -> float:
    """
    Compute NML for expression with noise nodes.
    
    NML = p(D|M) = ∫ p(D|θ,M) p(θ|M) dθ
    
    With noise: θ includes both model params and noise params
    
    Args:
        expression: Expression tree (may have noise nodes)
        X_train: Training inputs
        y_train: Training outputs
        n_attempts: Number of random initializations for optimization
        
    Returns:
        Log NML (approximation via Laplace's method)
        
    Implementation:
        1. Find MAP estimate θ* = argmax p(D|θ,M)p(θ|M)
        2. Compute Hessian H at θ*
        3. NML ≈ log p(D|θ*,M) + log p(θ*|M) + 0.5*log(|2π H^-1|)
    """
    best_nml = -np.inf
    
    for attempt in range(n_attempts):
        # Optimize parameters (including noise params)
        result = optimize_parameters_with_noise(
            expression, X_train, y_train, method='L-BFGS-B'
        )
        
        if not result['success']:
            continue
        
        theta_star = np.concatenate([
            result['theta'],
            list(result['noise_params'].values())
        ])
        
        # Log-likelihood at θ*
        ll_star = result['log_likelihood']
        
        # Log-prior (assume Gaussian prior on all params)
        prior_mean = 0.0
        prior_std = 10.0  # Broad prior
        log_prior = np.sum(norm.logpdf(theta_star, loc=prior_mean, scale=prior_std))
        
        # Hessian (use finite differences)
        H = compute_hessian_fd(
            lambda theta: -log_likelihood_with_noise(y_train, expression, X_train, theta),
            theta_star
        )
        
        # Laplace approximation
        # log p(D|M) ≈ log p(D|θ*) + log p(θ*) + 0.5*log(det(2π H^-1))
        try:
            sign, logdet = np.linalg.slogdet(H)
            if sign <= 0:
                logdet = 0  # Hessian not positive definite
            
            nml = ll_star + log_prior + 0.5 * (len(theta_star) * np.log(2*np.pi) - logdet)
            
            if nml > best_nml:
                best_nml = nml
                
        except np.linalg.LinAlgError:
            continue
    
    return best_nml

def compute_hessian_fd(func, x, eps=1e-5):
    """Compute Hessian using finite differences."""
    n = len(x)
    H = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            # Central difference
            x_pp = x.copy(); x_pp[i] += eps; x_pp[j] += eps
            x_pm = x.copy(); x_pm[i] += eps; x_pm[j] -= eps
            x_mp = x.copy(); x_mp[i] -= eps; x_mp[j] += eps
            x_mm = x.copy(); x_mm[i] -= eps; x_mm[j] -= eps
            
            H[i, j] = (func(x_pp) - func(x_pm) - func(x_mp) + func(x_mm)) / (4*eps*eps)
            H[j, i] = H[i, j]
    
    return H
```

**Deliverables:**
- Modified SMC proposal with noise mutations
- NML calculation for noisy models
- Integration tests

---

## Phase 4: Online Learning Framework (Weeks 11-15)

### 4.1 Online Update Architecture

**Objective:** Adapt model as new data streams in

#### Step 4.1.1: Recursive Parameter Updates

```python
# File: src/online_updates.py

"""
Online parameter updates for symbolic regression.
"""

from dataclasses import dataclass
import numpy as np

@dataclass
class OnlineState:
    """
    State maintained for online learning.
    
    Attributes:
        expression: Current best symbolic model
        theta: Current parameter estimates
        noise_params: Current noise parameter estimates
        P_theta: Parameter covariance (for uncertainty)
        n_samples_seen: Total data points processed
        last_update_time: Timestamp of last update
    """
    expression: ExpressionNode
    theta: np.ndarray
    noise_params: Dict[str, float]
    P_theta: np.ndarray  # Covariance matrix
    n_samples_seen: int
    last_update_time: float

class OnlineSymbolicRegression:
    """
    Online symbolic regression with forgetting factor.
    
    Combines:
    - Recursive Least Squares (RLS) for parameter updates
    - SMC re-weighting for structural adaptation
    - Drift detection for model re-discovery
    """
    
    def __init__(self,
                 initial_expression: ExpressionNode,
                 forgetting_factor: float = 0.99,
                 drift_threshold: float = 2.0):
        """
        Initialize online SR system.
        
        Args:
            initial_expression: Starting symbolic model
            forgetting_factor: λ ∈ (0,1]. Lower = more forgetting
                              λ=1: no forgetting (standard RLS)
                              λ=0.95: forget ~5% of past at each step
            drift_threshold: Trigger re-discovery if error grows by this factor
        """
        self.forgetting_factor = forgetting_factor
        self.drift_threshold = drift_threshold
        
        # Initialize state
        n_params = _count_total_parameters(initial_expression)
        self.state = OnlineState(
            expression=initial_expression,
            theta=np.random.randn(n_params) * 0.1,
            noise_params=_extract_noise_params(initial_expression),
            P_theta=np.eye(n_params) * 10.0,  # Large initial uncertainty
            n_samples_seen=0,
            last_update_time=0.0
        )
        
        self.error_history = []
        
    def update(self, x_new: np.ndarray, y_new: float):
        """
        Update model with single new observation.
        
        Args:
            x_new: New input features (n_features,)
            y_new: New output observation (scalar)
            
        Process:
            1. Predict ŷ using current parameters
            2. Compute prediction error e = y_new - ŷ
            3. Update parameters via RLS
            4. Check for drift; if detected, trigger re-discovery
            
        Example:
            >>> online_sr = OnlineSymbolicRegression(initial_expr)
            >>> for x, y in data_stream:
            >>>     online_sr.update(x, y)
            >>>     print(f"Current params: {online_sr.state.theta}")
        """
        # 1. Predict with current model
        X_single = x_new.reshape(1, -1)
        y_pred = self.state.expression.evaluate(
            X_single, 
            self.state.theta,
            sample_noise=False  # Use expected value for prediction
        )[0]
        
        # 2. Prediction error
        error = y_new - y_pred
        
        # 3. Compute Jacobian (gradient of model w.r.t. parameters)
        jacobian = self._compute_jacobian(x_new)  # (n_params,)
        
        # 4. RLS update with forgetting factor
        # See: Ljung & Söderström (1983), "Theory and Practice of Recursive Identification"
        
        λ = self.forgetting_factor
        P = self.state.P_theta
        θ = self.state.theta
        φ = jacobian  # Regressor vector
        
        # Gain vector
        K = (P @ φ) / (λ + φ.T @ P @ φ)
        
        # Update parameters
        θ_new = θ + K * error
        
        # Update covariance
        P_new = (P - np.outer(K, φ.T @ P)) / λ
        
        # 5. Update state
        self.state.theta = θ_new
        self.state.P_theta = P_new
        self.state.n_samples_seen += 1
        
        # 6. Track error for drift detection
        self.error_history.append(abs(error))
        
        # 7. Check for drift
        if self._detect_drift():
            print(f"[DRIFT DETECTED at sample {self.state.n_samples_seen}]")
            self._trigger_rediscovery()
    
    def _compute_jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Compute ∂f/∂θ at point x.
        
        Uses finite differences.
        """
        n_params = len(self.state.theta)
        jacobian = np.zeros(n_params)
        eps = 1e-7
        
        X_single = x.reshape(1, -1)
        
        for i in range(n_params):
            theta_plus = self.state.theta.copy()
            theta_plus[i] += eps
            
            theta_minus = self.state.theta.copy()
            theta_minus[i] -= eps
            
            y_plus = self.state.expression.evaluate(X_single, theta_plus, sample_noise=False)[0]
            y_minus = self.state.expression.evaluate(X_single, theta_minus, sample_noise=False)[0]
            
            jacobian[i] = (y_plus - y_minus) / (2*eps)
        
        return jacobian
    
    def _detect_drift(self, window_size: int = 50) -> bool:
        """
        Detect if model performance is degrading (drift).
        
        Method: Compare recent error to baseline error.
        If recent_error / baseline_error > threshold, trigger re-discovery.
        """
        if len(self.error_history) < 2 * window_size:
            return False
        
        baseline_error = np.mean(self.error_history[-2*window_size:-window_size])
        recent_error = np.mean(self.error_history[-window_size:])
        
        drift_ratio = recent_error / (baseline_error + 1e-10)
        
        return drift_ratio > self.drift_threshold
    
    def _trigger_rediscovery(self):
        """
        Re-run SMC-SR on recent data to discover new structure.
        
        Process:
            1. Collect last N samples (sliding window)
            2. Run SMC-SR to find new expression
            3. Initialize RLS with new expression
            4. Reset error history
        """
        print("[Re-running symbolic discovery...]")
        
        # Collect recent data (last 500 points, for example)
        # (In real implementation, maintain a buffer)
        window_size = min(500, self.state.n_samples_seen)
        # X_recent, y_recent = ...  # Retrieve from buffer
        
        # Run SMC-SR
        # new_expression = run_smc_sr(X_recent, y_recent, ...)
        
        # Re-initialize
        # self.state.expression = new_expression
        # self.state.theta = ...  # Optimized params for new expr
        # self.state.P_theta = np.eye(n_params_new) * 10.0
        # self.error_history = []
        
        print("[Model structure updated!]")
```

#### Step 4.1.2: SMC Re-weighting

```python
# File: src/online_smc.py

"""
Online SMC with forgetting factor for structural adaptation.
"""

def update_smc_population_online(population: List[ExpressionNode],
                                 weights: np.ndarray,
                                 X_new: np.ndarray,
                                 y_new: np.ndarray,
                                 forgetting_factor: float = 0.99):
    """
    Update SMC particle weights with new data batch.
    
    Instead of re-running SMC from scratch, we can update weights:
    w_t(M) = w_{t-1}(M) * [p(D_new | M)]^λ
    
    Where λ is a forgetting factor to down-weight old data.
    
    Args:
        population: Current population of expressions
        weights: Current particle weights (sum to 1)
        X_new: New data batch (n_new, n_features)
        y_new: New targets (n_new,)
        forgetting_factor: How much to down-weight past
        
    Returns:
        Updated weights
    """
    n_particles = len(population)
    new_weights = np.zeros(n_particles)
    
    for i, expr in enumerate(population):
        # Optimize parameters on new data
        result = optimize_parameters_with_noise(expr, X_new, y_new)
        
        # Likelihood of new data
        ll_new = result['log_likelihood']
        
        # Update weight: w_new = w_old^λ * p(D_new|M)
        # (In log space: log w_new = λ*log w_old + log p(D_new|M))
        new_weights[i] = np.exp(
            forgetting_factor * np.log(weights[i] + 1e-300) + ll_new
        )
    
    # Normalize
    new_weights /= np.sum(new_weights)
    
    # Resample if effective sample size is low
    ess = 1.0 / np.sum(new_weights**2)
    if ess < n_particles / 2:
        indices = np.random.choice(n_particles, size=n_particles, p=new_weights)
        population = [population[i].copy() for i in indices]
        new_weights = np.ones(n_particles) / n_particles
    
    return population, new_weights
```

### 4.2 Drift Detection & Adaptation

```python
# File: src/drift_detection.py

"""
Methods for detecting concept drift and triggering adaptation.
"""

from collections import deque

class DriftDetector:
    """
    Monitors model performance to detect drift.
    
    Methods:
    - Statistical: ADWIN, DDM, EDDM
    - Error-based: Sliding window comparison
    - Explicit: User-provided change points
    """
    
    def __init__(self, 
                 method: str = 'sliding_window',
                 window_size: int = 100,
                 threshold: float = 2.0):
        self.method = method
        self.window_size = window_size
        self.threshold = threshold
        
        self.error_buffer = deque(maxlen=2*window_size)
        self.n_drifts_detected = 0
        
    def update(self, error: float) -> bool:
        """
        Update detector with new error value.
        
        Returns:
            True if drift detected, False otherwise
        """
        self.error_buffer.append(error)
        
        if self.method == 'sliding_window':
            return self._sliding_window_test()
        elif self.method == 'adwin':
            return self._adwin_test()
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _sliding_window_test(self) -> bool:
        """
        Compare recent error to baseline.
        
        Drift if: mean(recent) / mean(baseline) > threshold
        """
        if len(self.error_buffer) < 2*self.window_size:
            return False
        
        errors = list(self.error_buffer)
        baseline = errors[:self.window_size]
        recent = errors[self.window_size:]
        
        mean_baseline = np.mean(baseline)
        mean_recent = np.mean(recent)
        
        if mean_recent / (mean_baseline + 1e-10) > self.threshold:
            self.n_drifts_detected += 1
            self.error_buffer.clear()  # Reset
            return True
        
        return False
    
    def _adwin_test(self) -> bool:
        """
        Adaptive Windowing (ADWIN) algorithm.
        
        Reference: Bifet & Gavaldà (2007)
        """
        # Simplified version (full ADWIN is more complex)
        pass
```

**Deliverables:**
- `src/online_updates.py` - RLS parameter updates
- `src/online_smc.py` - Population re-weighting
- `src/drift_detection.py` - Drift detection methods
- Integration tests with synthetic drift scenarios

---

## Phase 5: Validation & Benchmarking (Weeks 16-18)

### 5.1 Benchmark Datasets

**Synthetic Systems:**

```python
# File: tests/benchmark_systems.py

"""
Synthetic dynamical systems for validation.
"""

import numpy as np

class BenchmarkSystem:
    """Base class for benchmark systems."""
    
    def generate_data(self, 
                      n_train: int,
                      n_test: int,
                      noise_level: float = 0.1,
                      drift_point: int = None) -> Dict:
        """Generate train/test data."""
        raise NotImplementedError

class LorenzSystem(BenchmarkSystem):
    """
    Chaotic Lorenz system with known equations:
    dx/dt = σ(y - x)
    dy/dt = x(ρ - z) - y
    dz/dt = xy - βz
    """
    
    def __init__(self, sigma=10, rho=28, beta=8/3):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        
    def generate_data(self, 
                      n_train: int,
                      n_test: int,
                      noise_level: float = 0.1,
                      drift_point: int = None):
        """
        Generate Lorenz trajectories with noise.
        
        Args:
            n_train: Number of training samples
            n_test: Number of test samples
            noise_level: Std of additive Gaussian noise
            drift_point: If provided, change parameters at this index
            
        Returns:
            Dict with X_train, y_train, X_test, y_test, derivatives
        """
        from scipy.integrate import odeint
        
        def lorenz_deriv(state, t, sigma, rho, beta):
            x, y, z = state
            return [
                sigma * (y - x),
                x * (rho - z) - y,
                x * y - beta * z
            ]
        
        # Generate clean trajectory
        t = np.linspace(0, 50, n_train + n_test)
        init_state = [1.0, 1.0, 1.0]
        
        # Optionally introduce drift
        if drift_point is not None:
            # First part: original params
            t1 = t[:drift_point]
            traj1 = odeint(lorenz_deriv, init_state, t1, 
                          args=(self.sigma, self.rho, self.beta))
            
            # Second part: changed params (e.g., rho *= 1.2)
            init_state2 = traj1[-1]
            t2 = t[drift_point:] - t[drift_point]
            traj2 = odeint(lorenz_deriv, init_state2, t2,
                          args=(self.sigma, self.rho*1.2, self.beta))
            
            states = np.vstack([traj1, traj2[1:]])
        else:
            states = odeint(lorenz_deriv, init_state, t,
                           args=(self.sigma, self.rho, self.beta))
        
        # Add noise
        noise = np.random.normal(0, noise_level, states.shape)
        states_noisy = states + noise
        
        # Compute derivatives (target for equation discovery)
        derivs = np.array([
            lorenz_deriv(s, 0, self.sigma, self.rho, self.beta) 
            for s in states
        ])
        
        # Split train/test
        X_train = states_noisy[:n_train]
        dX_train = derivs[:n_train]
        X_test = states_noisy[n_train:]
        dX_test = derivs[n_train:]
        
        return {
            'X_train': X_train,
            'y_train': dX_train,  # We're predicting derivatives
            'X_test': X_test,
            'y_test': dX_test,
            'states_clean': states,
            'drift_point': drift_point
        }

class NonlinearOscillator(BenchmarkSystem):
    """
    Duffing oscillator: ẍ + δẋ + αx + βx³ = γ cos(ωt)
    With heteroscedastic noise: noise scales with |x|
    """
    pass  # Similar implementation

class PolynomialWithNoise(BenchmarkSystem):
    """
    Simple polynomial with explicit noise model:
    y = θ0 + θ1*x + θ2*x² + η(σ = θ3*|x|)
    """
    
    def __init__(self, theta=[1, 2, 0.5], sigma_0=0.1, sigma_power=1.0):
        self.theta = theta
        self.sigma_0 = sigma_0
        self.sigma_power = sigma_power
        
    def generate_data(self, n_train, n_test, noise_level=None, drift_point=None):
        """Generate polynomial data with heteroscedastic noise."""
        n_total = n_train + n_test
        X = np.random.uniform(-3, 3, (n_total, 1))
        
        # Clean output
        y_clean = sum(self.theta[i] * X**i for i in range(len(self.theta)))
        
        # Heteroscedastic noise: σ(x) = σ_0 * |x|^α
        sigma_x = self.sigma_0 * np.abs(X.flatten())**self.sigma_power
        noise = np.random.normal(0, sigma_x)
        
        y = y_clean + noise
        
        return {
            'X_train': X[:n_train],
            'y_train': y[:n_train],
            'X_test': X[n_train:],
            'y_test': y[n_train:],
            'noise_std': sigma_x
        }
```

### 5.2 Evaluation Metrics

```python
# File: src/evaluation.py

"""
Metrics for evaluating symbolic regression with noise and online learning.
"""

def evaluate_model(model: ExpressionNode,
                  X: np.ndarray,
                  y_true: np.ndarray,
                  theta: np.ndarray,
                  noise_params: Dict) -> Dict[str, float]:
    """
    Comprehensive evaluation of a symbolic model.
    
    Metrics:
    - RMSE: Root mean squared error
    - R²: Coefficient of determination
    - Complexity: Number of nodes in expression tree
    - Parsimony: AIC, BIC
    - Noise accuracy: If noise is explicit, how well does it match true noise?
    
    Args:
        model: Expression tree
        X: Input features
        y_true: True outputs
        theta: Model parameters
        noise_params: Learned noise parameters
        
    Returns:
        Dict of metric_name -> value
    """
    # Predict (expected value, no noise)
    y_pred = model.evaluate(X, theta, sample_noise=False)
    
    # Prediction error
    residuals = y_true - y_pred
    rmse = np.sqrt(np.mean(residuals**2))
    
    # R²
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - ss_res / ss_tot
    
    # Model complexity
    n_nodes = _count_nodes(model)
    n_params = len(theta) + len(noise_params)
    
    # Information criteria
    n_samples = len(y_true)
    log_likelihood = -n_samples/2 * np.log(2*np.pi*rmse**2) - ss_res/(2*rmse**2)
    
    aic = 2*n_params - 2*log_likelihood
    bic = n_params*np.log(n_samples) - 2*log_likelihood
    
    metrics = {
        'rmse': rmse,
        'r2': r2,
        'n_nodes': n_nodes,
        'n_params': n_params,
        'aic': aic,
        'bic': bic,
        'log_likelihood': log_likelihood
    }
    
    # If model has explicit noise, evaluate noise prediction
    if _tree_has_noise(model):
        # Compare learned noise to actual residual distribution
        # (This requires knowing true noise model - only for synthetic data)
        pass
    
    return metrics

def evaluate_online_performance(online_sr: OnlineSymbolicRegression,
                                data_stream: List[Tuple[np.ndarray, float]],
                                eval_interval: int = 10) -> pd.DataFrame:
    """
    Evaluate online SR over a data stream.
    
    Records:
    - Prediction error at each step
    - Parameter values over time
    - Drift detection events
    - Model structure changes
    
    Returns:
        DataFrame with columns: [step, error, params, structure, drift_detected]
    """
    results = []
    
    for step, (x, y) in enumerate(data_stream):
        # Predict before update
        y_pred = online_sr.state.expression.evaluate(
            x.reshape(1, -1),
            online_sr.state.theta,
            sample_noise=False
        )[0]
        error = abs(y - y_pred)
        
        # Update
        drift_detected = online_sr._detect_drift() if step > 100 else False
        online_sr.update(x, y)
        
        # Record
        if step % eval_interval == 0:
            results.append({
                'step': step,
                'error': error,
                'params': online_sr.state.theta.copy(),
                'structure': str(online_sr.state.expression),
                'drift_detected': drift_detected,
                'n_samples': online_sr.state.n_samples_seen
            })
    
    return pd.DataFrame(results)
```

### 5.3 Experimental Protocol

```python
# File: experiments/run_benchmarks.py

"""
Main experimental script.
"""

def run_experiment(system_name: str,
                  noise_level: float,
                  drift_point: int = None,
                  online: bool = True):
    """
    Run full experiment on a benchmark system.
    
    Protocol:
    1. Generate data with known noise model
    2. Initial discovery: Run SMC-SR with noise operators
    3. (If online) Stream additional data, update parameters
    4. Evaluate: Compare learned model to ground truth
    5. Report metrics and visualizations
    
    Args:
        system_name: 'lorenz', 'duffing', 'polynomial', etc.
        noise_level: Std of noise (if applicable)
        drift_point: Sample index where dynamics change (None = no drift)
        online: Whether to test online adaptation
    """
    print(f"\n{'='*60}")
    print(f"Experiment: {system_name}")
    print(f"Noise level: {noise_level}")
    print(f"Drift: {'Yes' if drift_point else 'No'}")
    print(f"Online: {'Yes' if online else 'No'}")
    print(f"{'='*60}\n")
    
    # 1. Generate data
    system = create_benchmark_system(system_name)
    data = system.generate_data(
        n_train=500,
        n_test=500,
        noise_level=noise_level,
        drift_point=drift_point
    )
    
    # 2. Initial discovery
    print("[Phase 1: Initial Discovery]")
    smc_sr = SMC_SR_with_Noise(
        n_particles=1000,
        n_iterations=100,
        noise_operators=NOISE_OPERATORS,
        mutation_probs={
            'add_node': 0.3,
            'remove_node': 0.2,
            'change_op': 0.2,
            'insert_noise': 0.15,
            'remove_noise': 0.1,
            'mutate_noise': 0.05
        }
    )
    
    smc_sr.fit(data['X_train'], data['y_train'])
    
    best_expr = smc_sr.get_best_expression()
    print(f"Discovered model: {best_expr}")
    
    # Evaluate on test set
    metrics_static = evaluate_model(
        best_expr,
        data['X_test'],
        data['y_test'],
        smc_sr.best_theta,
        smc_sr.best_noise_params
    )
    print(f"Test RMSE: {metrics_static['rmse']:.4f}")
    print(f"Test R²: {metrics_static['r2']:.4f}")
    
    # 3. Online adaptation (if enabled)
    if online:
        print("\n[Phase 2: Online Adaptation]")
        
        # Initialize online SR with discovered model
        online_sr = OnlineSymbolicRegression(
            initial_expression=best_expr,
            forgetting_factor=0.99,
            drift_threshold=2.0
        )
        
        # Stream test data
        stream = list(zip(data['X_test'], data['y_test']))
        
        results = evaluate_online_performance(
            online_sr,
            stream,
            eval_interval=10
        )
        
        # Plot evolution
        plot_online_results(results, data)
        
    # 4. Comparison to baselines
    print("\n[Phase 3: Baseline Comparison]")
    
    # Baseline 1: Standard SMC-SR (no noise operators)
    # Baseline 2: Fixed model with RLS only
    # Baseline 3: PySINDy, gplearn, etc.
    
    # 5. Save results
    save_experiment_results(
        system_name=system_name,
        discovered_model=best_expr,
        metrics=metrics_static,
        online_results=results if online else None
    )
```

**Deliverables:**
- Benchmark systems with known noise models
- Evaluation metrics suite
- Experimental scripts
- Baseline comparisons
- Results visualization

---

## Phase 6: Documentation & Code Quality (Weeks 19-20)

### 6.1 Code Documentation

**Standards:**
- Every function: docstring with Args, Returns, Example
- Every class: docstring with Attributes, Methods overview
- Every module: module-level docstring explaining purpose
- Type hints throughout

**Example Template:**

```python
def function_name(arg1: Type1, arg2: Type2) -> ReturnType:
    """
    One-line summary of what this function does.
    
    More detailed explanation if needed. This can span multiple paragraphs
    and explain the algorithm, key assumptions, or mathematical background.
    
    Args:
        arg1: Description of arg1. Include units, ranges, or constraints.
              Can span multiple lines if needed.
        arg2: Description of arg2.
        
    Returns:
        Description of return value. If returning a complex structure,
        document each field.
        
    Raises:
        ValueError: When arg1 is negative
        RuntimeError: When optimization fails to converge
        
    Example:
        >>> result = function_name(5, "test")
        >>> print(result)
        42
        
    Notes:
        Additional information, mathematical formulas, references.
        
    References:
        [1] Author et al. (2025). "Paper title". Journal.
    """
    pass
```

### 6.2 Tutorial Notebooks

**Notebook 1: Basic Usage**
```python
# File: tutorials/01_basic_noise_modeling.ipynb

"""
Tutorial: Discovering Equations with Explicit Noise

This notebook demonstrates:
1. Creating a simple synthetic dataset with noise
2. Running SMC-SR with noise operators
3. Interpreting the learned noise model
4. Comparing to standard SR (no noise modeling)
"""

# [Cell 1: Setup]
import numpy as np
from src.smc_sr import SMC_SR_with_Noise
from src.noise_operators import NOISE_OPERATORS

# [Cell 2: Generate Data]
# True model: y = 2x + 1 + η(σ=0.5*|x|)
# This has heteroscedastic noise!

X = np.random.uniform(-5, 5, (200, 1))
y_clean = 2*X + 1
noise_std = 0.5 * np.abs(X)
y = y_clean + np.random.normal(0, noise_std)

# [Cell 3: Visualize]
import matplotlib.pyplot as plt

plt.scatter(X, y, alpha=0.5, label='Observed data')
plt.plot(X, y_clean, 'r-', label='True function')
plt.fill_between(X.flatten(), 
                 y_clean.flatten() - 2*noise_std.flatten(),
                 y_clean.flatten() + 2*noise_std.flatten(),
                 alpha=0.2, label='±2σ (true noise)')
plt.legend()
plt.show()

# [Cell 4: Run SMC-SR with noise]
sr = SMC_SR_with_Noise(n_particles=500, n_iterations=50)
sr.fit(X, y)

best_model = sr.get_best_expression()
print(f"Discovered: {best_model}")
print(f"Noise params: {sr.best_noise_params}")

# [Cell 5: Compare predictions]
# Plot learned noise vs true noise
# ...

# And so on with detailed explanations and visualizations
```

**Notebook 2: Online Learning**
**Notebook 3: Multi-Equation Systems**

### 6.3 API Reference

Generate Sphinx documentation:

```bash
# File: docs/conf.py
# Configure Sphinx for API docs

# File: docs/index.rst
# Main documentation page

# Generate HTML docs
cd docs
make html
```

---

## Phase 7: Paper Writing & Publication (Weeks 21-24)

### 7.1 Paper Outline

```
Title: Noise-Explicit Online Symbolic Regression via Bayesian Sequential Monte Carlo

Abstract
1. Introduction
   - Motivation: SR fails with noise, can't adapt online
   - Our contribution: Explicit noise + online updates
   
2. Related Work
   - Standard SR (GP, SINDy, etc.)
   - Noise modeling (Schmidt & Lipson 2007, NRSR 2025)
   - Online learning (BRSL 2026, Sym-Q 2024)
   - Our positioning

3. Method
   3.1 Noise-Aware Expression Trees
   3.2 SMC with Stochastic Operators
   3.3 Online Parameter Updates via RLS
   3.4 Drift Detection & Structural Adaptation

4. Experiments
   4.1 Synthetic Benchmarks
   4.2 Noise Recovery Analysis
   4.3 Online Adaptation Performance
   4.4 Comparison to Baselines

5. Results
   - Tables, figures, ablation studies
   
6. Discussion
   - When does explicit noise help?
   - Trade-offs of online adaptation
   - Limitations

7. Conclusion

Appendix
   - Derivations
   - Additional experiments
   - Hyperparameter settings
```

### 7.2 Figures & Tables

**Key Figures:**
1. Architecture diagram
2. Example expression tree with noise nodes
3. Lorenz recovery comparison (vs PySINDy, static SMC-SR)
4. Online adaptation with drift (error over time)
5. Noise parameter learning accuracy
6. Ablation study (noise ops vs no noise, online vs static)

**Key Tables:**
1. Benchmark results (RMSE, R², complexity)
2. Hyperparameter settings
3. Computational cost comparison

---

## Timeline Summary

| Phase | Weeks | Deliverable |
|-------|-------|-------------|
| 1. Foundation | 1-3 | Architecture design, decisions |
| 2. Noise Modeling | 4-7 | Noise operators, likelihood, tests |
| 3. SMC Integration | 8-10 | SMC mutations, NML for noise |
| 4. Online Learning | 11-15 | RLS updates, drift detection |
| 5. Validation | 16-18 | Benchmarks, metrics, comparisons |
| 6. Documentation | 19-20 | Code docs, tutorials, API ref |
| 7. Paper | 21-24 | Write, submit, revise |
| **Total** | **24 weeks** | **Complete research project** |

---

## Risk Mitigation

### Technical Risks

1. **Risk:** Noise parameter optimization may be unstable
   - **Mitigation:** Use constrained optimization, multiple initializations, robust priors

2. **Risk:** Online updates might diverge or overfit
   - **Mitigation:** Implement forgetting factor, cross-validation on windows, drift detection

3. **Risk:** Computational cost too high for real-time online
   - **Mitigation:** Profile code, use Cython/JAX, limit population size, smart re-discovery triggers

### Research Risks

1. **Risk:** Explicit noise modeling doesn't improve much over ignoring noise
   - **Mitigation:** Careful experimental design with systems where noise is truly heteroscedastic/structured

2. **Risk:** Online adaptation is unnecessary (static model works fine)
   - **Mitigation:** Test on drift scenarios, emphasize when it helps vs when static is sufficient

---

## Success Criteria

### Minimum Viable Product (MVP)
- [ ] Noise operators implemented and tested
- [ ] SMC-SR can discover models with noise nodes
- [ ] Online RLS parameter updates working
- [ ] One benchmark (polynomial with heteroscedastic noise) validated

### Target Goals
- [ ] Full system (noise + online) working on 3+ benchmarks
- [ ] Outperforms baselines on noisy data
- [ ] Online adaptation recovers from drift
- [ ] Clear documentation and tutorials

### Stretch Goals
- [ ] Real-world application (e.g., sensor data with drift)
- [ ] Multi-equation discovery with noise
- [ ] Published in top-tier venue (NeurIPS, ICML, etc.)
- [ ] Open-source release with community adoption

---

## Next Steps (Week 1)

1. **Read papers in detail**
   - Schmidt & Lipson 2007 (stochastic symbols)
   - BRSL 2026 (Bayesian online)
   - Review SMC-SR paper again

2. **Set up project structure**
   ```
   project/
   ├── src/
   │   ├── noise_operators.py
   │   ├── expression_tree.py
   │   ├── likelihood.py
   │   ├── smc_sr.py
   │   ├── online_updates.py
   │   └── ...
   ├── tests/
   ├── experiments/
   ├── tutorials/
   └── docs/
   ```

3. **Prototype noise operators**
   - Implement GAUSSIAN_NOISE_CONSTANT
   - Write unit tests
   - Verify distribution properties

4. **Weekly meeting**
   - Review progress
   - Adjust plan as needed
   - Discuss blockers

---

## References

1. Schmidt & Lipson (2007). "Distilling Free-Form Natural Laws from Experimental Data"
2. BRSL (2026). "Bayesian Regression-based Symbolic Learning"
3. NRSR (2025). "Noise-Resilient Symbolic Regression"
4. Sym-Q (2024). "Symbolic Regression via Q-Learning"
5. RAG-SR (2025). "Retrieval-Augmented Generation for Symbolic Regression"
6. SyMANTIC (2025). "An Efficient Symbolic Regression Method"
7. SMC-SR (2025). "Bayesian Symbolic Regression via Posterior Sampling" (this project!)

---

**End of Work Plan**

This plan provides:
✓ Clear phase-by-phase breakdown
✓ Concrete code examples with documentation
✓ Testing strategies
✓ Validation protocols
✓ Timeline with deliverables
✓ Risk mitigation
✓ Success criteria

Ready to start implementation!
