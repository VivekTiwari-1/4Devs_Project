# Energy-Efficient Container Cloud Simulator

A comprehensive simulation and machine learning project for optimizing energy efficiency in cloud data centers through intelligent container placement, CPU core allocation, and live migration strategies.

## 📋 Overview

This project implements a cloud computing simulator that combines:

- **Q-Learning Reinforcement Learning** - Intelligent CPU allocation policy learning
- **Container Management** - Dynamic container placement and scheduling
- **Live Migration** - Container migration between physical machines
- **Energy Modeling** - Realistic power consumption calculations
- **Performance Analysis** - Baseline vs RL vs RL+Migration comparisons

The simulator models a data center with multiple physical machines running containerized workloads, optimizing for energy efficiency while meeting deadline constraints.

## 🎯 Key Features

### Core Simulation Engine

- **Multi-PM Infrastructure**: Simulate multiple physical machines with configurable resources
- **Dynamic Workload Generation**: Random and pattern-based container arrival rates
- **Time-Slot Based Simulation**: Discrete time steps (default 30 seconds per slot)
- **Deadline Management**: Containers with realistic deadline constraints

### Optimization Strategies

#### 1. **Baseline Strategy**

- First-fit placement on least loaded PMs
- Uniform CPU core allocation
- No dynamic scaling or migration

#### 2. **RL-Based Scaling**

- Q-Learning agent learns optimal CPU allocation policies
- Minimizes energy consumption while respecting deadlines
- Adaptive epsilon-greedy exploration/exploitation
- Configurable state space and action set

#### 3. **RL + Migration**

- Combines intelligent CPU allocation with live container migration
- Rebalances workloads between PMs for better resource utilization
- Reduces total energy consumption through consolidation

### Energy & Performance Metrics

- **Energy Consumption**: Dynamic power based on CPU utilization
- **Deadline Violations**: Tracking missed deadlines
- **Container Completion Time**: Execution latency tracking
- **PM Utilization**: Core and memory usage statistics
- **System Efficiency**: Energy-delay product and SLA compliance

## 📁 Project Structure

```
├── main.py                      # Entry point with interactive menu
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── scripts/                     # Executable scripts
│   ├── train_rl.py             # Train RL agent (30 episodes)
│   ├── compare_all.py          # Compare strategies
│   ├── plot_results.py         # Generate visualizations
│   └── run_step_1.py           # Step 1 demo script
│
├── src/                         # Main source code
│   ├── config/
│   │   └── config.py           # All hyperparameters & settings
│   ├── environment/
│   │   ├── simulator.py        # Main simulation engine
│   │   ├── pm.py               # Physical machine model
│   │   ├── container.py        # Container model
│   │   ├── energy_model.py     # Energy consumption model
│   │   └── workload_generator.py # Workload generation
│   ├── modules/
│   │   ├── placement.py        # Placement algorithm
│   │   ├── migration.py        # Live migration logic
│   │   └── policies.py         # CPU allocation policies
│   ├── rl/
│   │   └── q_learning_agent.py # Q-Learning implementation
│   └── utils/                  # Utility functions
│
├── tests/                       # Unit tests
│   ├── test_energy.py
│   ├── test_migration.py
│   └── compare_baseline_vs_rl.py
│
└── results/                     # Output data
    ├── logs/
    │   ├── comparison_results.json
    │   └── training_history.json
    ├── plots/                   # Generated visualizations
    └── qtables/                 # Saved Q-tables from training
```

## 🚀 Quick Start

### Installation

1. **Clone/Setup the project**

   ```bash
   cd /home/vivek/Projects/4Devs_Project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project

#### Interactive Mode

```bash
python main.py
```

Select from the menu:

1. **Train RL Agent** - Train the Q-Learning agent for 30 episodes
2. **Run Comparison** - Compare Baseline vs RL vs RL+Migration strategies
3. **Generate Plots** - Create visualizations from results
4. **Full Pipeline** - Train → Compare → Plot (complete workflow)
5. **Quick Demo** - Single episode demonstration

#### Direct Execution

```bash
# Train RL agent
python scripts/train_rl.py

# Run comparison analysis
python scripts/compare_all.py

# Generate plots
python scripts/plot_results.py

# Run specific test
python tests/test_energy.py
```

## ⚙️ Configuration

All simulation parameters are in [src/config/config.py](src/config/config.py):

### Simulation Settings

```python
TIME_SLOT_DURATION = 30  # seconds per time slot
TOTAL_SIMULATION_SLOTS = 50  # simulation duration
RANDOM_SEED = 42  # reproducibility
```

### Physical Machine Specs

```python
PM_TOTAL_CORES = 68  # CPU cores per PM
PM_MEMORY_GB = 256  # Memory per PM
PM_NETWORK_GBPS = 10  # Network bandwidth
```

### Container Specifications

```python
CONTAINER_MIN_INSTRUCTIONS = 2_000_000_000  # Min workload
CONTAINER_MAX_INSTRUCTIONS = 8_000_000_000  # Max workload
CONTAINER_MIN_DEADLINE_OFFSET = 300  # Min deadline (5 min)
CONTAINER_MAX_DEADLINE_OFFSET = 900  # Max deadline (15 min)
```

### RL Agent Hyperparameters

```python
RL_LEARNING_RATE = 0.1  # Alpha
RL_DISCOUNT_FACTOR = 0.95  # Gamma
RL_EPSILON_START = 1.0  # Initial exploration
RL_EPSILON_DECAY = 0.995  # Epsilon decay rate
```

## 📊 Output & Results

Results are saved in the `results/` directory:

- **JSON Logs**: Detailed metrics from simulations
- **Plots**: Visualization of training progress and comparisons
- **Q-Tables**: Serialized learned policies from RL training

### Key Metrics Tracked

- Total energy consumption (kWh)
- Deadline violation percentage
- Average container completion time
- Physical machine utilization rates
- Container survival/success rates

## 🧠 Q-Learning Implementation

### State Space

- **Number of containers** on physical machine
- **Average deadline gap** (time remaining)
- **Available CPU cores** on PM

### Action Space

- Selection of CPU allocation policies from predefined set
- Policies include: aggressive, conservative, balanced approaches

### Reward Function

```
Reward = -total_energy - (deadline_violations × penalty)
```

## 🔄 Live Migration Module

The migration strategy:

1. Monitors PM CPU utilization
2. Identifies under-utilized PMs
3. Consolidates containers to fewer PMs
4. Reduces overall energy consumption through better bin-packing

## 📈 Performance Comparison

The comparison framework evaluates:

- **Baseline**: First-fit with static allocation
- **RL**: Dynamic allocation via trained Q-Learning agent
- **RL+Migration**: RL + container consolidation

Results show energy savings and SLA compliance metrics.

## 🧪 Testing

Run the test suite:

```bash
python tests/test_energy.py           # Energy model tests
python tests/test_migration.py        # Migration logic tests
python tests/compare_baseline_vs_rl.py # Strategy comparison
```

## 📚 Technical Details

### Energy Model

- Dynamic idle power (based on component count)
- Active power scales with CPU utilization
- Network power for container communication
- Realistic power coefficients for modern hardware

### Container Lifecycle

1. **Arrival**: Generated by workload generator
2. **Placement**: Assigned to PM using placement algorithm
3. **Execution**: CPU cores allocated (learned by RL)
4. **Migration**: Optionally moved to other PMs
5. **Completion**: Removed when execution finishes

### Deadline Satisfaction

- Deadlines are monitored continuously
- Violations are penalized in RL reward function
- Metrics track SLA compliance

## 📝 Key Files

| File                                                         | Purpose                           |
| ------------------------------------------------------------ | --------------------------------- |
| [main.py](main.py)                                           | Entry point with interactive menu |
| [src/config/config.py](src/config/config.py)                 | All configuration parameters      |
| [src/environment/simulator.py](src/environment/simulator.py) | Core simulation engine            |
| [src/rl/q_learning_agent.py](src/rl/q_learning_agent.py)     | Q-Learning implementation         |
| [src/modules/migration.py](src/modules/migration.py)         | Live migration logic              |
| [scripts/train_rl.py](scripts/train_rl.py)                   | RL training script                |
| [scripts/compare_all.py](scripts/compare_all.py)             | Strategy comparison               |

## 🎓 Learning Outcomes

This project demonstrates:

- Reinforcement learning application in resource allocation
- Cloud computing optimization techniques
- Energy efficiency in data centers
- Multi-objective optimization (energy vs. deadline)
- Simulation and experimental evaluation

## 🔧 Dependencies

Core libraries:

- **Python 3.7+**
- NumPy - Numerical computing
- Pandas - Data analysis
- Matplotlib/Seaborn - Visualization
- Scikit-learn - ML utilities (if applicable)

See [requirements.txt](requirements.txt) for complete dependency list.

## 📄 License

This project is part of the 4Devs initiative.

## 👤 Author

Created as an educational project for cloud computing and machine learning.

---

## 🚦 Typical Workflow

1. **Setup**: Configure parameters in `config.py`
2. **Train**: Run RL agent training to convergence
3. **Evaluate**: Compare strategies using comparison module
4. **Analyze**: Generate plots and examine results
5. **Iterate**: Adjust hyperparameters and repeat

## 💡 Tips

- Start with the **Quick Demo** option to understand the workflow
- Modify `RANDOM_SEED` for different scenarios
- Adjust `MAX_CONTAINERS_PER_SLOT` to change workload intensity
- Check `results/plots/` directory for generated visualizations
- Review JSON logs for detailed performance metrics

---

**Happy simulating! 🚀**
