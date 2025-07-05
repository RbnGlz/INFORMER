# Contributing to Informer

Thank you for your interest in contributing to the Informer project! This guide will help you get started with contributing to our time series forecasting model.

## 🚀 Quick Start

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/informer.git
   cd informer
   ```

2. **Development Installation**
   ```bash
   make install-dev
   ```
   This will:
   - Install the package in editable mode
   - Install all development dependencies
   - Set up pre-commit hooks

3. **Verify Setup**
   ```bash
   make test
   ```

## 📋 Development Workflow

### Before Making Changes

1. **Create a branch** for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Check current status**:
   ```bash
   make lint    # Check code quality
   make test    # Run all tests
   ```

### Making Changes

1. **Write your code** following our guidelines (see below)

2. **Add tests** for new functionality:
   - Unit tests in `tests/unit/`
   - Integration tests in `tests/integration/`

3. **Update documentation** if needed:
   - Docstrings for new functions/classes
   - README if adding new features
   - Type hints for all parameters

### Before Committing

Our pre-commit hooks will automatically run, but you can also check manually:

```bash
make format  # Format code with black & isort
make lint    # Check with flake8 & mypy
make test    # Run all tests
make security # Security checks
```

### Submitting Changes

1. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a Pull Request** with:
   - Clear description of changes
   - Link to any related issues
   - Screenshots/examples if applicable

## 🎯 Types of Contributions

### 🐛 Bug Reports

When reporting bugs, please include:
- **Environment**: Python version, OS, package versions
- **Steps to reproduce**: Clear, minimal example
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Error messages**: Full traceback if applicable

**Template:**
```markdown
**Environment:**
- Python: 3.9.0
- OS: Ubuntu 20.04
- Informer: 1.0.0

**Steps to reproduce:**
1. Import the model
2. Create instance with parameters X, Y, Z
3. Call method foo()

**Expected:** Should return tensor of shape [32, 24, 1]
**Actual:** Raises ValueError

**Error:**
```
[paste full traceback here]
```
```

### ✨ Feature Requests

For new features:
- **Use case**: Why is this needed?
- **Proposed solution**: How would it work?
- **Alternatives**: Other approaches considered?
- **Backwards compatibility**: Will it break existing code?

### 🔧 Code Contributions

**Good first issues:**
- Documentation improvements
- Additional tests
- Performance optimizations
- Bug fixes

**Larger contributions:**
- New attention mechanisms
- Additional model variants
- Benchmark improvements
- Integration with other frameworks

## 📝 Code Style Guidelines

### Python Code

We follow PEP 8 with these specific guidelines:

1. **Line length**: 88 characters (Black default)
2. **Imports**: Organized with isort
3. **Type hints**: Required for all public functions
4. **Docstrings**: Google/NumPy style

**Example:**
```python
def prob_attention_forward(
    self,
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Forward pass of probabilistic attention mechanism.
    
    Args:
        queries: Query tensor of shape [batch, seq_len, heads, dim]
        keys: Key tensor of shape [batch, seq_len, heads, dim]
        values: Value tensor of shape [batch, seq_len, heads, dim]
        attention_mask: Optional attention mask
        
    Returns:
        Tuple containing:
        - Output tensor of shape [batch, heads, seq_len, dim]
        - Optional attention weights
        
    Raises:
        ValueError: If input dimensions don't match
    """
    # Implementation here
    pass
```

### Testing

1. **Test organization**:
   ```
   tests/
   ├── unit/           # Component-specific tests
   ├── integration/    # End-to-end tests
   └── fixtures/       # Shared test data
   ```

2. **Test naming**: `test_<functionality>_<scenario>`
3. **Assertions**: Use descriptive messages
4. **Fixtures**: Use pytest fixtures for shared setup

**Example test:**
```python
def test_prob_attention_output_shape(self, sample_batch):
    """Test that ProbAttention returns correct output shape."""
    attention = ProbAttention(factor=3)
    B, L, H, D = 2, 32, 4, 64
    
    queries = torch.randn(B, L, H, D)
    keys = torch.randn(B, L, H, D) 
    values = torch.randn(B, L, H, D)
    
    output, _ = attention(queries, keys, values, None)
    
    expected_shape = (B, H, L, D)
    assert output.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {output.shape}"
    )
```

### Documentation

1. **Docstrings**: All public functions must have docstrings
2. **Type hints**: Include for parameters and return values
3. **Examples**: Include usage examples for complex functions
4. **Updates**: Keep README and docs in sync with code changes

## 🧪 Testing Guidelines

### Running Tests

```bash
# All tests
make test

# Specific test file
pytest tests/unit/test_attention.py -v

# Specific test function
pytest tests/unit/test_attention.py::TestProbAttention::test_forward_shape -v

# With coverage
pytest --cov=src/informer tests/

# Parallel execution
make test-parallel
```

### Writing Tests

1. **Coverage**: Aim for >90% test coverage
2. **Edge cases**: Test boundary conditions
3. **Error handling**: Test invalid inputs
4. **Performance**: Add performance regression tests for critical paths

### Test Categories

- **Unit tests**: Individual components in isolation
- **Integration tests**: Component interactions
- **Regression tests**: Prevent bugs from reoccurring
- **Performance tests**: Ensure speed requirements

## 🔒 Security

### Reporting Vulnerabilities

Please see our [Security Policy](SECURITY.md) for reporting vulnerabilities.

### Security Best Practices

- Never commit secrets or API keys
- Use parameterized queries/inputs
- Validate all user inputs
- Keep dependencies updated

## 📚 Documentation

### Building Documentation

```bash
# Install docs dependencies
pip install -e ".[docs]"

# Build documentation
make docs

# View locally
open docs/_build/html/index.html
```

### Documentation Types

1. **API Documentation**: Auto-generated from docstrings
2. **User Guide**: How-to guides and tutorials
3. **Developer Guide**: This file and technical docs
4. **Examples**: Jupyter notebooks and scripts

## 🎉 Recognition

Contributors are recognized in:
- **CHANGELOG.md**: Major contributions
- **GitHub releases**: Feature acknowledgments
- **Documentation**: Example authors
- **README**: Core contributors

## ❓ Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Code Review**: Ask for feedback on draft PRs

## 📄 License

By contributing, you agree that your contributions will be licensed under the same MIT License that covers the project.

---

## Development Commands Reference

| Command | Description |
|---------|-------------|
| `make install` | Basic installation |
| `make install-dev` | Development installation with pre-commit |
| `make test` | Run all tests with coverage |
| `make test-parallel` | Run tests in parallel |
| `make lint` | Code quality checks |
| `make format` | Auto-format code |
| `make security` | Security scans |
| `make docs` | Build documentation |
| `make clean` | Clean build artifacts |
| `make all` | Format, lint, test, security |

---

**Thank you for contributing to Informer! 🙏**