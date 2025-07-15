# Sql_Injection_Detection

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/psf/black)

A state-of-the-art implementation of SQL injection detection using Graph Attention Networks (GAT) and Multi-Expert Deep Q-Network (DQN) with reinforcement learning. This research combines graph neural networks, multi-agent systems, and adversarial training to achieve superior detection performance.

## 🎯 Key Features

- **🕸️ Graph-based SQL Analysis**: SQL queries modeled as dependency graphs to capture syntactic relationships
- **🧠 Multi-Expert Architecture**: Four specialized expert agents for different SQL injection attack types
- **🎮 Reinforcement Learning**: DQN-based dynamic weight allocation for intelligent expert fusion
- **⚔️ Adversarial Training**: Six transformation modules for robust data augmentation
- **📊 Comprehensive Evaluation**: ROC/PR curves, confusion matrices, and detailed ablation studies
- **🔬 Research-Ready**: Complete experimental framework with hyperparameter tuning

## 📈 Performance Highlights

- **95.5%** detection accuracy on comprehensive SQL injection datasets
- **0.978** AUC score, significantly outperforming baseline methods
- **Robust performance** across different attack types and obfuscation techniques
- **Low computational overhead** suitable for real-time deployment

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/DanielDeng888/sql-injection-detection.git
cd sql-injection-detection
pip install -r requirements.txt
```

Or install in development mode:
```bash
pip install -e .
```

### Basic Usage

```python
# Run the main experiment
python experiments/main_experiment.py

# Run hyperparameter tuning
python experiments/hyperparameter_tuning.py

# Run ablation study
python experiments/ablation_study.py
```

### Quick Example

```python
from src.models.dqn import DQNAgent
from src.environments.sql_injection_env import SQLInjectionEnv
from src.utils.data_utils import load_and_preprocess_data

# Load your dataset
train_queries, train_labels, test_queries, test_labels = load_and_preprocess_data(
    'data/sql_injection_dataset.csv'
)

# Create environment and agent
env = SQLInjectionEnv(train_queries, train_labels)
agent = DQNAgent(input_dim=env.observation_space.shape[0])

# Train and evaluate
from src.utils.training_utils import train_dqn
results = train_dqn(env, agent, total_timesteps=50000)
```

## 📁 Project Structure

```
sql_injection_detection/
├── 📄 README.md                    # This file
├── 📄 requirements.txt             # Python dependencies
├── 📄 setup.py                     # Package installation
├── 📁 config/                      # Configuration management
│   ├── 🔧 base_config.py          # Base configuration
│   └── 🔧 experiment_config.py    # Experiment-specific configs
├── 📁 src/                         # Core source code
│   ├── 📁 models/                  # Neural network models
│   │   ├── 🤖 dqn.py              # DQN implementations
│   │   ├── 👥 expert_agents.py    # Multi-expert system
│   │   └── 🕸️ graph_builder.py    # SQL graph construction
│   ├── 📁 environments/            # RL environments
│   │   └── 🎮 sql_injection_env.py # Main environment
│   ├── 📁 utils/                   # Utility functions
│   │   ├── 📊 data_utils.py        # Data processing
│   │   └── 🏋️ training_utils.py    # Training utilities
│   └── 📁 visualization/           # Plotting and analysis
│       ├── 📈 training_plots.py    # Training visualizations
│       └── 🔬 ablation_plots.py    # Ablation study plots
├── 📁 experiments/                 # Experiment scripts
│   ├── 🎯 main_experiment.py       # Main research experiment
│   ├── 🔍 hyperparameter_tuning.py # Grid search optimization
│   └── 🧪 ablation_study.py        # Component analysis
├── 📁 tests/                       # Unit tests
├── 📁 data/                        # Dataset directory
└── 📁 results/                     # Output directory
    ├── 📁 models/                  # Saved model checkpoints
    ├── 📁 visualizations/          # Generated plots
    └── 📁 ablation_study/          # Ablation results
```

## 🏗️ Core Components

### 1. Graph Attention Network (GAT)

The SQL Grammar Graph Builder transforms SQL queries into structured graphs:

```python
# Example: SQL query to graph transformation
query = "SELECT * FROM users WHERE id = 1 UNION SELECT password FROM admin"

# Builds dependency graph with:
# - Nodes: SQL tokens with type annotations
# - Edges: Syntactic relationships (sequence, union_select, where_condition)
# - Features: 10-dimensional feature vector capturing graph properties
```

**Key Features:**
- 🔗 Captures syntactic dependencies between SQL components
- 📊 Extracts 10-dimensional normalized feature vectors
- 🎯 Focuses on injection-relevant patterns (UNION, WHERE, OR conditions)

### 2. Multi-Expert System

Four specialized expert agents, each targeting specific attack types:

| Expert | Target Attacks | Key Patterns |
|--------|---------------|--------------|
| **Union Expert** | UNION-based injections | `UNION SELECT`, cross-table queries |
| **Time Expert** | Time-based blind injections | `SLEEP()`, `BENCHMARK()`, delay functions |
| **Error Expert** | Error-based injections | `EXTRACTVALUE()`, forced errors |
| **Boolean Expert** | Boolean-based blind injections | `AND 1=1`, `OR 1=1`, logic manipulation |

### 3. DQN Architecture

**Dueling DQN** with advanced features:
- 🔄 **Double DQN**: Reduces overestimation bias
- 🎯 **Dueling Architecture**: Separates state value and action advantage
- 💾 **Experience Replay**: Stabilizes training with memory buffer
- 🎯 **Target Network**: Periodic updates for stable Q-learning

### 4. Adversarial Sample Generation

Six transformation modules for data augmentation:

```python
# Example transformations applied to: SELECT * FROM users WHERE id = 1
transformations = {
    "double_url_encoding": "SELECT%2520*%2520FROM%2520users%2520WHERE%2520id%2520=%25201",
    "case_variation": "SeLeCt * FrOm users WhErE id = 1",
    "whitespace_bypass": "SELECT/**/*/**/FROM/**/users/**/WHERE/**/id/**/=/**/1",
    "unicode_encoding": "SELECT * FROM users WHERE id = \\u0031",
    "version_comments": "SELECT/*!50000*/ * FROM users WHERE id = 1",
    "inline_comments": "SEL/**/ECT * FR/**/OM users WH/**/ERE id = 1"
}
```

## 🔧 Configuration

### Base Configuration

```python
from config.base_config import BaseConfig

config = BaseConfig()
print(f"Device: {config.device}")
print(f"Learning rate: {config.learning_rate}")
print(f"Training steps: {config.total_timesteps}")
```

### Custom Configuration

```python
from config.base_config import BaseConfig

class CustomConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        # Override specific parameters
        self.learning_rate = 1e-3
        self.batch_size = 128
        self.total_timesteps = 100000
        self.num_experts = 6  # Add more experts

config = CustomConfig()
```

## 📊 Usage Examples

### 1. Training with Custom Dataset

```python
from src.utils.data_utils import load_and_preprocess_data
from src.models.dqn import DQNAgent
from src.environments.sql_injection_env import SQLInjectionEnv
from src.utils.training_utils import train_dqn, evaluate_dqn

# Load your dataset (CSV with 'query' and 'label' columns)
train_queries, train_labels, test_queries, test_labels = load_and_preprocess_data(
    'your_dataset.csv', test_size=0.2
)

# Create environment with custom settings
env = SQLInjectionEnv(
    train_queries, train_labels,
    use_experts=True,
    use_graph_features=True,
    num_experts=4
)

eval_env = SQLInjectionEnv(test_queries, test_labels)

# Create and train agent
agent = DQNAgent(
    input_dim=env.observation_space.shape[0],
    hidden_dim=256,
    lr=3e-4,
    use_dueling=True,
    use_double=True
)

# Train with evaluation
results = train_dqn(
    env, agent,
    total_timesteps=50000,
    eval_env=eval_env,
    eval_freq=5000
)

# Evaluate final performance
final_reward, final_accuracy, metrics = evaluate_dqn(agent, eval_env)
print(f"Final accuracy: {final_accuracy:.4f}")
```

### 2. Ablation Study Example

```python
from experiments.ablation_study import run_ablation_studies

# Run comprehensive ablation study
ablation_results = run_ablation_studies(
    train_queries, train_labels,
    test_queries, test_labels
)

# Results automatically saved to results/ablation_study/
# Includes ROC curves, PR curves, and component analysis
```

### 3. Hyperparameter Optimization

```python
from experiments.hyperparameter_tuning import HyperparameterTuner

tuner = HyperparameterTuner(train_queries, train_labels)

# Define custom parameter grid
tuner.config.learning_rates = [1e-4, 3e-4, 1e-3, 3e-3]
tuner.config.batch_sizes = [32, 64, 128, 256]
tuner.config.hidden_dims = [64, 128, 256, 512]

# Run optimization
tuner.run_tuning_experiments()
```

### 4. Visualization and Analysis

```python
from src.visualization.training_plots import plot_training_results, plot_roc_curve
from src.visualization.ablation_plots import plot_multiple_roc_curves

# Plot training progress
plot_training_results(training_results)

# Generate ROC curve
plot_roc_curve(true_labels, predicted_probs, 
               title='SQL Injection Detection ROC')

# Compare multiple models
models_metrics = {
    'Complete Model': baseline_metrics,
    'w/o Graph Features': no_graph_metrics,
    'w/o Expert Agents': no_experts_metrics
}
plot_multiple_roc_curves(models_metrics)
```

## 🧪 Experimental Results

### Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC | AUC-PR |
|-------|----------|-----------|--------|----------|---------|--------|
| **Complete Model** | **95.5%** | **96.2%** | **94.8%** | **95.5%** | **0.978** | **0.979** |
| w/o Graph Features | 87.6% | 88.5% | 86.7% | 87.6% | 0.930 | 0.932 |
| w/o Expert Agents | 90.8% | 91.5% | 90.1% | 90.8% | 0.956 | 0.958 |
| w/o Dueling Network | 93.2% | 93.8% | 92.6% | 93.2% | 0.968 | 0.970 |
| w/o Dense Reward | 92.1% | 92.7% | 91.5% | 92.1% | 0.961 | 0.963 |

### Ablation Study Insights

1. **Graph Features** contribute **7.9%** to overall accuracy
2. **Expert Agents** provide **4.7%** performance improvement
3. **Dueling Architecture** adds **2.3%** accuracy gain
4. **Dense Reward Design** improves performance by **3.4%**

### Attack Type Performance

| Attack Type | Samples | Accuracy | Precision | Recall |
|-------------|---------|----------|-----------|--------|
| Union-based | 3,247 | 97.2% | 96.8% | 97.6% |
| Boolean-based | 2,891 | 94.1% | 95.3% | 92.9% |
| Error-based | 2,156 | 96.7% | 97.1% | 96.3% |
| Time-based | 1,823 | 93.8% | 92.4% | 95.2% |

## 🛠️ Development

### Setting up Development Environment

```bash
# Clone repository
git clone https://github.com/DanielDeng888/sql-injection-detection.git
cd sql-injection-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
pip install -r requirements-dev.txt  # If available

# Run tests
python -m pytest tests/ -v

# Code formatting
black src/ experiments/ tests/
flake8 src/ experiments/ tests/
```

### Adding New Components

#### New Expert Agent
```python
# src/models/custom_expert.py
class CustomExpert(ExpertAgent):
    def __init__(self, input_dim, hidden_dim):
        super().__init__(input_dim, hidden_dim, "Custom Expert")
        # Add custom layers
        self.custom_layer = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        # Custom forward pass
        x = super().forward(x)
        return self.custom_layer(x)
```

#### New Transformation Module
```python
# src/utils/transformations.py
def custom_obfuscation(query):
    """Apply custom obfuscation technique"""
    # Implement your transformation
    return transformed_query
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_models.py -v

# Run with coverage
python -m pytest --cov=src tests/
```

## 📋 Requirements

### Core Dependencies

- **Python**: 3.8+
- **PyTorch**: 1.13.0+
- **NumPy**: 1.21.0+
- **Pandas**: 1.3.0+
- **NetworkX**: 2.8.0+
- **Scikit-learn**: 1.1.0+
- **Matplotlib**: 3.5.0+
- **Seaborn**: 0.11.0+

### Optional Dependencies

- **Weights & Biases**: For experiment tracking
- **TensorBoard**: For training visualization
- **Jupyter**: For interactive analysis
- **pytest**: For testing

### Hardware Requirements

- **Minimum**: 8GB RAM, CPU-only training
- **Recommended**: 16GB RAM, CUDA-compatible GPU
- **Optimal**: 32GB RAM, RTX 3080/4080 or better

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Make** your changes
4. **Add** tests for new functionality
5. **Ensure** all tests pass (`python -m pytest`)
6. **Format** code (`black src/`)
7. **Commit** changes (`git commit -m 'Add amazing feature'`)
8. **Push** to branch (`git push origin feature/amazing-feature`)
9. **Open** a Pull Request

### Areas for Contribution

- 🚀 **New Expert Agents**: Additional specialized detection modules
- 🔧 **Optimization Techniques**: Advanced training algorithms
- 📊 **Evaluation Metrics**: Novel assessment methods
- 🛡️ **Robustness Testing**: Adversarial attack resistance
- 📚 **Documentation**: Tutorials and examples
- 🧪 **Testing**: Expand test coverage

## 📚 Documentation

### API Reference

- [Models Documentation](docs/models.md)
- [Environment API](docs/environment.md)
- [Configuration Guide](docs/configuration.md)
- [Visualization Guide](docs/visualization.md)

### Research Papers

This implementation is based on our research paper:

```bibtex
@article{sqlinjection2025,
  title={A SQL Injection Detection Model Integrating GAT and Interpretable DQN},
  author={Daniel},
  journal={...},
  year={2025},
  volume={...},
  pages={...}
}
```

### Related Work

- [Graph Attention Networks (Veličković et al., 2018)](https://arxiv.org/abs/1710.10903)
- [Deep Q-Networks (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)
- [SQL Injection Attack Detection Survey](https://doi.org/xxx)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 SQL Injection Detection Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

## 🙏 Acknowledgments

- **PyTorch Team** for the excellent deep learning framework
- **NetworkX Developers** for graph processing capabilities
- **Gymnasium** for reinforcement learning environments
- **Research Community** for foundational work in SQL injection detection
- **Contributors** who help improve this project

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/DanielDeng888/sql-injection-detection/issues)
- **Discussions**: [GitHub Discussions](https://github.com/DanielDeng88/sql-injection-detection/discussions)
- **Email**: DanielDengyuyang@163.com

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=DanielDeng888/sql-injection-detection&type=Date)](https://star-history.com/#DanielDeng888/sql-injection-detection&Date)

---

<div align="center">
  <strong>Built with ❤️ for the cybersecurity research community</strong>
</div>
