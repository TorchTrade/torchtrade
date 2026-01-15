# PR #94 Simplified Test Examples

This document shows concrete before/after examples of the most impactful simplifications.

---

## Example 1: Parametrized Batch Size Testing

### Before (test_grpo_loss.py lines 161-182)
```python
def test_loss_with_different_batch_sizes(self, actor_network, obs_dim, action_dim):
    """Test loss computation with different batch sizes."""
    loss = GRPOLoss(actor_network=actor_network)

    for batch_size in [1, 5, 32]:
        data = TensorDict(
            {
                "observation": torch.randn(batch_size, obs_dim),
                "action": torch.randint(0, action_dim, (batch_size,)),
                "action_log_prob": torch.randn(batch_size),
                "next": {
                    "reward": torch.randn(batch_size, 1),
                    "done": torch.zeros(batch_size, 1, dtype=torch.bool),
                    "terminated": torch.zeros(batch_size, 1, dtype=torch.bool),
                },
            },
            batch_size=[batch_size],
        )
        output = loss(data)
        assert "loss_objective" in output.keys()
        assert output["loss_objective"].requires_grad
```

**Issues**:
- Loop creates implicit subtests that aren't tracked separately by pytest
- Harder to identify which batch size failed if test fails
- More indentation increases cognitive load

### After (Recommended)
```python
@pytest.mark.parametrize("batch_size", [1, 5, 32])
def test_with_different_batch_sizes(self, actor_network, obs_dim, action_dim, batch_size):
    """Test loss computation with different batch sizes."""
    loss = GRPOLoss(actor_network=actor_network)

    data = TensorDict(
        {
            "observation": torch.randn(batch_size, obs_dim),
            "action": torch.randint(0, action_dim, (batch_size,)),
            "action_log_prob": torch.randn(batch_size),
            "next": {
                "reward": torch.randn(batch_size, 1),
                "done": torch.zeros(batch_size, 1, dtype=torch.bool),
                "terminated": torch.zeros(batch_size, 1, dtype=torch.bool),
            },
        },
        batch_size=[batch_size],
    )
    output = loss(data)
    assert output["loss_objective"].requires_grad
```

**Benefits**:
- 3 separate test runs visible in pytest output
- Clearer failure messages showing exact batch size
- Less indentation and no loop
- Removed redundant `in output.keys()` check
- Shorter test name (removed redundant "loss_" prefix)

---

## Example 2: Simplify Scalar Fixtures to Class Constants

### Before (test_grpo_loss.py lines 20-30)
```python
class TestGRPOLoss:
    """Test suite for GRPOLoss."""

    @pytest.fixture
    def obs_dim(self):
        return 4

    @pytest.fixture
    def action_dim(self):
        return 3

    @pytest.fixture
    def batch_size(self):
        return 10
```

**Issues**:
- Unnecessary fixture overhead for simple constants
- Values hidden from test method signatures
- Requires fixture injection for every test

### After (Recommended)
```python
class TestGRPOLoss:
    """Test suite for GRPOLoss."""

    OBS_DIM = 4
    ACTION_DIM = 3
    BATCH_SIZE = 10

    @pytest.fixture
    def actor_network(self):
        """Create a simple actor network for testing."""
        policy_net = TensorDictModule(
            nn.Sequential(
                nn.Linear(self.OBS_DIM, 64),
                nn.ReLU(),
                nn.Linear(64, self.ACTION_DIM),
            ),
            in_keys=["observation"],
            out_keys=["logits"],
        )
        # ... rest of setup
```

**Benefits**:
- Dimensions immediately visible at top of class
- No fixture injection needed
- Still accessible in all methods via `self.OBS_DIM`
- Reduces fixture count from 6 to 3

---

## Example 3: Remove Redundant Key Checks

### Before (Multiple locations)
```python
def test_forward_pass(self, actor_network, sample_data):
    """Test forward pass produces expected outputs."""
    loss = GRPOLoss(actor_network=actor_network)
    output = loss(sample_data)

    assert "loss_objective" in output.keys()
    assert output["loss_objective"].requires_grad
    assert output["loss_objective"].shape == torch.Size([])
```

**Issues**:
- First assertion is redundant - accessing the key directly will raise KeyError if missing
- `output.keys()` is unnecessary - can use `in output`

### After (Recommended)
```python
def test_forward_pass(self, actor_network, sample_data):
    """Test forward pass produces expected outputs."""
    loss = GRPOLoss(actor_network=actor_network)
    output = loss(sample_data)

    assert output["loss_objective"].requires_grad
    assert output["loss_objective"].shape == torch.Size([])
```

**Benefits**:
- One fewer assertion
- If key is missing, KeyError provides same information
- More concise and direct

---

## Example 4: Consolidate Functional Mode Tests

### Before (test_grpo_loss.py lines 239-250)
```python
def test_functional_mode(self, actor_network):
    """Test that functional mode works correctly."""
    loss = GRPOLoss(actor_network=actor_network, functional=True)
    assert loss.functional is True
    assert loss.actor_network_params is not None

def test_non_functional_mode(self, actor_network):
    """Test that non-functional mode works correctly."""
    loss = GRPOLoss(actor_network=actor_network, functional=False)
    assert loss.functional is False
    assert loss.actor_network_params is None
```

**Issues**:
- Two separate tests for opposite sides of same boolean flag
- Tests are essentially identical except for the flag value

### After (Recommended)
```python
@pytest.mark.parametrize("functional,has_params", [
    (True, True),
    (False, False),
])
def test_functional_mode(self, actor_network, functional, has_params):
    """Test functional mode setting."""
    loss = GRPOLoss(actor_network=actor_network, functional=functional)
    assert loss.functional is functional

    if has_params:
        assert loss.actor_network_params is not None
    else:
        assert loss.actor_network_params is None
```

**Alternative (simpler but less explicit)**:
```python
@pytest.mark.parametrize("functional", [True, False])
def test_functional_mode(self, actor_network, functional):
    """Test functional mode setting."""
    loss = GRPOLoss(actor_network=actor_network, functional=functional)
    assert loss.functional is functional
    assert (loss.actor_network_params is not None) == functional
```

**Benefits**:
- Reduces from 2 tests to 1 parametrized test
- Clearer relationship between functional flag and params
- Less code duplication

---

## Example 5: Simplify Boolean Assertions

### Before (Multiple locations)
```python
def test_loss_initialization(self, actor_network):
    """Test that loss module initializes correctly."""
    loss = GRPOLoss(actor_network=actor_network)
    assert loss is not None
    assert loss.actor_network is not None
    assert loss.epsilon_low == 0.2
    assert loss.epsilon_high == 0.2
    assert loss.entropy_bonus is True
    assert loss.reduction == "mean"
```

**Issues**:
- `assert loss is not None` is unnecessary (constructor would raise if failed)
- `assert loss.entropy_bonus is True` can be simplified

### After (Recommended)
```python
def test_initialization(self, actor_network):
    """Test loss module initializes correctly."""
    loss = GRPOLoss(actor_network=actor_network)
    assert loss.actor_network is not None
    assert loss.epsilon_low == 0.2
    assert loss.epsilon_high == 0.2
    assert loss.entropy_bonus
    assert loss.reduction == "mean"
```

**Benefits**:
- More Pythonic boolean assertions
- Removed redundant null check for loss object
- Shorter test name

---

## Example 6: Sinkhorn Iteration Testing

### Before (test_ctrl_loss.py lines 161-169)
```python
def test_sinkhorn_iterations(self):
    """Test Sinkhorn with different iteration counts."""
    scores = torch.randn(10, 20)

    for iters in [1, 3, 5, 10]:
        Q = CTRLLoss._sinkhorn(scores, iters=iters)
        assert Q.shape == scores.shape
        assert torch.isfinite(Q).all()
```

### After (Recommended)
```python
@pytest.mark.parametrize("iters", [1, 3, 5, 10])
def test_sinkhorn_iterations(self, iters):
    """Test Sinkhorn with different iteration counts."""
    scores = torch.randn(10, 20)
    Q = CTRLLoss._sinkhorn(scores, iters=iters)

    assert Q.shape == scores.shape
    assert torch.isfinite(Q).all()
```

**Benefits**:
- Each iteration count runs as separate test
- Less indentation
- Clearer test output if one iteration count fails

---

## Example 7: Reduction Mode Testing

### Before (test_grpo_loss.py lines 149-160)
```python
def test_loss_reduction_modes(self, actor_network, sample_data):
    """Test different reduction modes."""
    # Mean reduction
    loss_mean = GRPOLoss(actor_network=actor_network, reduction="mean")
    output_mean = loss_mean(sample_data)
    assert output_mean["loss_objective"].shape == torch.Size([])

    # Sum reduction
    loss_sum = GRPOLoss(actor_network=actor_network, reduction="sum")
    output_sum = loss_sum(sample_data)
    assert output_sum["loss_objective"].shape == torch.Size([])
```

### After (Recommended)
```python
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_reduction_modes(self, actor_network, sample_data, reduction):
    """Test different reduction modes."""
    loss = GRPOLoss(actor_network=actor_network, reduction=reduction)
    output = loss(sample_data)
    assert output["loss_objective"].shape == torch.Size([])
```

**Benefits**:
- Eliminates duplicate code
- Easier to add new reduction modes
- Each mode tested separately in pytest output

---

## Example 8: Helper Method for Gradient Checking

### Before (test_grpo_loss.py lines 133-148)
```python
def test_loss_backward_pass(self, actor_network, sample_data):
    """Test that gradients flow correctly through the loss."""
    loss = GRPOLoss(actor_network=actor_network)
    output = loss(sample_data)

    # Check that we can compute gradients
    total_loss = output["loss_objective"]
    if "loss_entropy" in output.keys():
        total_loss = total_loss + output["loss_entropy"]

    total_loss.backward()

    # Verify that actor network has gradients
    for param in loss.actor_network_params.values(True, True):
        assert param.grad is not None
```

### After (Recommended)
```python
class TestGRPOLoss:
    """Test suite for GRPOLoss."""

    def _assert_has_gradients(self, parameters):
        """Helper to verify all parameters have gradients."""
        for param in parameters:
            assert param.grad is not None

    def test_backward_pass(self, actor_network, sample_data):
        """Test that gradients flow correctly through the loss."""
        loss = GRPOLoss(actor_network=actor_network)
        output = loss(sample_data)

        total_loss = output["loss_objective"]
        if "loss_entropy" in output:
            total_loss = total_loss + output["loss_entropy"]

        total_loss.backward()

        self._assert_has_gradients(loss.actor_network_params.values(True, True))
```

**Benefits**:
- Reusable gradient checking logic
- Can add better error messages in helper
- Cleaner test method
- Also removed `.keys()` from entropy check

---

## Summary of Changes by File

### test_grpo_loss.py
- Convert 5 loop-based tests to parametrized tests
- Change 3 scalar fixtures to class constants
- Remove 15+ redundant `.keys()` checks
- Consolidate 2 functional mode tests into 1 parametrized test
- Add gradient checking helper method
- Shorten test method names (remove "loss_" prefix)

**Estimated reduction**: ~40 lines (~12% of file)

### test_ctrl_loss.py
- Convert 3 loop-based tests to parametrized tests
- Change 4 scalar fixtures to class constants
- Remove 10+ redundant `.keys()` checks
- Simplify mock fixture creation
- Add gradient checking helper method
- Rename fixtures for clarity

**Estimated reduction**: ~30 lines (~7% of file)

---

## Implementation Priority

1. **Quick wins** (15 minutes):
   - Remove redundant `.keys()` checks (~20 locations)
   - Simplify boolean assertions (~5 locations)

2. **Medium effort** (30 minutes):
   - Convert loop tests to parametrized tests (~5 tests)
   - Convert fixtures to class constants (~7 fixtures)

3. **Optional** (15 minutes):
   - Consolidate similar tests (~2-3 tests)
   - Add helper methods (~2 helpers)
   - Rename tests/fixtures (~10 renames)

**Total time**: ~1 hour for all improvements
