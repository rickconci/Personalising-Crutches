# Design Decisions: Pydantic vs Dataclass & Hydra vs YAML

## 1. Pydantic vs Dataclass for Data Models

### The Issue

Originally used `@dataclass` for `HoleLayout` and `Geometry`, but `pydantic.BaseModel` for configuration classes. This was inconsistent and suboptimal.

### Why Pydantic is Better

#### **Validation & Type Safety**

```python
# ❌ Dataclass - No validation
@dataclass
class Geometry:
    alpha: float  # Could be negative, > 360°, etc.

# ✅ Pydantic - Automatic validation
class Geometry(BaseModel):
    alpha: float = Field(..., ge=0, le=180, description="Angle in degrees")
    
    @validator('alpha')
    def validate_reasonable_angle(cls, v):
        if not (60 <= v <= 150):
            raise ValueError(f"Alpha {v}° outside reasonable range")
        return v
```

#### **JAX Array Handling**

```python
# ❌ Dataclass - Manual conversion
@dataclass  
class HoleLayout:
    handle: jnp.ndarray
    
    def __post_init__(self):
        self.handle = jnp.asarray(self.handle)  # Manual

# ✅ Pydantic - Automatic conversion
class HoleLayout(BaseModel):
    handle: jnp.ndarray
    
    @validator('handle', pre=True)
    def ensure_jax_array(cls, v):
        return jnp.asarray(v)  # Automatic
    
    class Config:
        arbitrary_types_allowed = True
```

#### **Serialization & API Integration**

```python
# ❌ Dataclass - Manual JSON handling
geometry_dict = asdict(geometry)  # Basic dict
json.dumps(geometry_dict)  # Manual serialization

# ✅ Pydantic - Rich serialization
geometry.json()  # Direct JSON
geometry.dict()  # Rich dict with type conversion
geometry.schema()  # OpenAPI schema
```

#### **Documentation & IDE Support**

```python
# ✅ Pydantic - Rich field documentation
class Geometry(BaseModel):
    alpha: float = Field(
        ..., 
        ge=85, le=115,
        description="Angle between vertical and handle rod",
        example=95.0
    )
    
# Auto-generates:
# - OpenAPI schemas
# - JSON Schema
# - IDE autocompletion with descriptions
```

### **Performance Comparison**

- **Dataclass**: Faster creation (~2x)
- **Pydantic**: Slower creation, but safer + more features
- **For our use case**: Safety > Speed (we create few objects, use them extensively)

---

## 2. Hydra vs YAML Configuration

### The Issue

Current system uses manual YAML loading with limited composability and no command-line overrides.

### Why Hydra is Better

#### **1. Configuration Composition**

```yaml
# ❌ Current - Monolithic config
# config.yaml (200+ lines)
optimizer_type: "differentiable"
differentiable:
  learning_rate: 0.01
  max_iterations: 1000
constraints:
  alpha_min: 85.0
  # ... everything in one file

# ✅ Hydra - Composable configs
# config.yaml
defaults:
  - optimizer: differentiable
  - experiment: baseline

# optimizer/differentiable.yaml  
learning_rate: 0.01
max_iterations: 1000

# optimizer/genetic_algorithm.yaml
population_size: 100
num_generations: 500
```

#### **2. Command Line Overrides**

```bash
# ❌ Current - Edit files manually
vim config.yaml  # Change learning_rate: 0.001
python main.py

# ✅ Hydra - Command line overrides
python hydra_main.py differentiable.learning_rate=0.001
python hydra_main.py optimizer=genetic_algorithm
python hydra_main.py constraints.alpha_max=120 objectives.vocabulary_weight=2.0
```

#### **3. Hyperparameter Sweeps**

```bash
# ❌ Current - Manual loops
for lr in 0.001 0.01 0.1; do
    sed -i "s/learning_rate: .*/learning_rate: $lr/" config.yaml
    python main.py --output results_lr_$lr
done

# ✅ Hydra - Automatic sweeps
python hydra_main.py -m differentiable.learning_rate=0.001,0.01,0.1
# Creates: outputs/2023-12-07/10-30-45/0/, outputs/2023-12-07/10-30-45/1/, etc.
```

#### **4. Automatic Output Management**

```bash
# ❌ Current - Manual output directories
mkdir results_experiment_1
python main.py --output results_experiment_1

# ✅ Hydra - Automatic timestamped outputs
python hydra_main.py
# Creates: outputs/2023-12-07/10-30-45/
# Contains: .hydra/config.yaml, results.yaml, plots/, etc.
```

#### **5. Experiment Reproducibility**

```python
# ❌ Current - Manual config saving
config_dict = config.__dict__
with open("results/config_used.yaml", "w") as f:
    yaml.dump(config_dict, f)

# ✅ Hydra - Automatic config logging
# Every run automatically saves:
# .hydra/config.yaml      - Final resolved config
# .hydra/overrides.yaml   - Command line overrides
# .hydra/hydra.yaml       - Hydra settings
```

### **Advanced Hydra Features**

#### **Multi-Run with Different Optimizers**

```bash
# Compare optimizers automatically
python hydra_main.py -m optimizer=differentiable,genetic_algorithm,hybrid
```

#### **Structured Configs with Validation**

```python
# Hydra + Pydantic = Ultimate validation
@dataclass
class OptimizerConfig:
    learning_rate: float = 0.01
    max_iterations: int = 1000

cs = ConfigStore.instance()
cs.store(name="optimizer_schema", node=OptimizerConfig)
```

#### **Integration with Optuna/Ray**

```bash
# Automatic hyperparameter optimization
python hydra_main.py -m hydra/sweeper=optuna
# Uses Optuna to intelligently search hyperparameters
```

---

## 3. **Migration Strategy**

### **Phase 1: Keep Both Systems** ✅

- Current `main.py` with YAML configs
- New `hydra_main.py` with Hydra configs  
- Users can choose their preference

### **Phase 2: Gradual Migration**

- Convert more configs to Pydantic
- Add more Hydra experiment templates
- Benchmark performance differences

### **Phase 3: Full Migration**

- Deprecate old YAML system
- Hydra becomes primary interface
- Keep Pydantic validation throughout

---

## 4. **Usage Examples**

### **Current System**

```bash
# Edit config file
vim configs/default_differentiable.yaml

# Run experiment  
python main.py optimize --config configs/default_differentiable.yaml

# Run benchmark
python main.py benchmark
```

### **Hydra System**

```bash
# Quick parameter changes
python hydra_main.py differentiable.learning_rate=0.001

# Switch optimizers
python hydra_main.py optimizer=genetic_algorithm

# Hyperparameter sweep
python hydra_main.py -m differentiable.learning_rate=0.001,0.01,0.1

# Complex experiment
python hydra_main.py \
  optimizer=differentiable \
  differentiable.max_iterations=2000 \
  constraints.alpha_max=120 \
  objectives.vocabulary_weight=2.0 \
  random_seed=123
```

---

## 5. **Recommendations**

### **For Research/Experimentation: Use Hydra** 🚀

- Rapid parameter exploration
- Automatic experiment tracking
- Easy hyperparameter sweeps
- Professional ML workflow

### **For Production/Deployment: Use Current System** 🏭

- Simpler dependencies
- More predictable behavior
- Easier Docker integration
- Clear configuration files

### **For Development: Use Both** 🔧

- Hydra for research experiments
- YAML for integration tests
- Pydantic validation everywhere
- Best of both worlds

---

## 6. **Performance Impact**

### **Pydantic Overhead**

- ~2-3x slower object creation
- Negligible for our use case (few objects created)
- Massive benefit in validation and debugging

### **Hydra Overhead**

- ~100ms startup cost
- Negligible for optimization runs (minutes/hours)
- Huge productivity gain for experiments

### **Memory Usage**

- Pydantic: ~20% more memory per object
- Hydra: ~10MB additional memory
- Both negligible for our application

---

## **Conclusion**

**Pydantic + Hydra = Professional ML Research Framework**

This combination provides:

- ✅ **Type Safety**: Catch errors early
- ✅ **Experiment Management**: Professional workflow
- ✅ **Reproducibility**: Automatic config logging  
- ✅ **Productivity**: Rapid experimentation
- ✅ **Scalability**: Easy hyperparameter sweeps
- ✅ **Documentation**: Self-documenting configs

The slight performance overhead is more than offset by the productivity gains and reduced debugging time.
