# Environments for MARL Experiments

This research uses reinforcement learning environments rather than static datasets.
Environments are installed as Python packages or cloned as code repositories.

## Primary Environments

### 1. MacroMARL / MacDec-POMDP Environments (Most Relevant)

**The most directly aligned environment for heterogeneous decision horizons.**

- **Source**: https://github.com/yuchen-x/MacroMARL
- **Location**: `code/MacroMARL/`
- **Formalism**: MacDec-POMDP (Macro-Action Decentralized POMDP) — agents execute macro-actions (temporally extended actions) of **different durations**
- **Environments included**:
  - Box Pushing: Two robots cooperatively push boxes; macro-actions have variable durations
  - Warehouse Tool Delivery: Multiple robots deliver tools; different macro-actions take different timesteps
- **Key Feature**: Native support for asynchronous, heterogeneous-horizon multi-agent decision-making
- **Installation**: `cd code/MacroMARL && pip install -e .` (requires conda environment setup, see repo README)

### 2. PettingZoo MPE (Multi-Agent Particle Environment)

**Fast prototyping environment. Can be customized for heterogeneous stepping.**

- **Source**: PyPI (`pettingzoo` package)
- **Status**: Installed and verified in `.venv/`
- **Key scenarios**: `simple_spread` (cooperative), `simple_tag` (predator-prey), `simple_speaker_listener` (heterogeneous roles)
- **Heterogeneous horizons**: Use AEC (Agent-Environment Cycle) API to implement different decision frequencies per agent
- **Installation**: Already installed (`pip install pettingzoo pygame`)

#### Loading the Environment

```python
from pettingzoo.mpe import simple_spread_v3

# Parallel API (synchronous)
env = simple_spread_v3.parallel_env(N=3)
obs, infos = env.reset()

# AEC API (can support asynchronous stepping)
env = simple_spread_v3.env(N=3)
env.reset()
for agent in env.agent_iter():
    obs, reward, termination, truncation, info = env.last()
    action = env.action_space(agent).sample()
    env.step(action)
```

#### Custom Heterogeneous Stepping Example

```python
# Wrap to make agents act at different frequencies
class HeterogeneousStepWrapper:
    def __init__(self, env, agent_frequencies):
        """agent_frequencies: dict mapping agent_name -> step_frequency
        e.g., {'agent_0': 1, 'agent_1': 3, 'agent_2': 5}
        Agent 0 acts every step, Agent 1 every 3, Agent 2 every 5.
        """
        self.env = env
        self.frequencies = agent_frequencies
        self.step_count = 0

    def step(self, actions):
        # Only pass actions for agents whose frequency divides step_count
        filtered_actions = {}
        for agent, action in actions.items():
            if self.step_count % self.frequencies.get(agent, 1) == 0:
                filtered_actions[agent] = action
            else:
                filtered_actions[agent] = self.last_actions.get(agent, 0)
        self.last_actions = dict(actions)
        self.step_count += 1
        return self.env.step(filtered_actions)
```

### 3. SMAC / SMACv2 (StarCraft Multi-Agent Challenge)

**Community standard benchmark for cooperative MARL.**

- **Source**: https://github.com/oxwhirl/smac (or smacv2)
- **Status**: Not installed (requires StarCraft II engine)
- **Installation**:
  ```bash
  pip install git+https://github.com/oxwhirl/smac.git
  # Then install StarCraft II and download SMAC maps
  bash install_sc2.sh  # Available in code/pymarl/
  ```
- **Note**: SMACv2 randomizes unit types per episode for harder evaluation

## Additional Environments (Optional)

| Environment | Install | Use Case |
|-------------|---------|----------|
| JaxMARL | `pip install jaxmarl` | JAX-accelerated training (12,500x faster) |
| VMAS | `pip install vmas` | Vectorized 2D physics with heterogeneous agents |
| Overcooked-AI | `pip install overcooked-ai` | Hierarchical subtask coordination |
| RWARE | `pip install rware` | Multi-robot warehouse logistics |

## Notes

- MARL environments are interactive simulators, not static datasets
- Data is generated on-the-fly during training (no pre-downloaded data files needed)
- The `.gitignore` below excludes any large generated data files

## Sample Data

For reference, here is what observations look like in MPE simple_spread:
```json
{
  "agent_0": {
    "observation_shape": [18],
    "observation_description": "velocity(2) + position(2) + landmark_positions(6) + other_agent_positions(4) + communication(4)",
    "action_space": "Discrete(5): [no_action, left, right, down, up]"
  }
}
```
