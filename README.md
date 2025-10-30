# NERT: Neurosymbolic Evaluation of Robot Task Planning

**NERT** is a neurosymbolic evaluation system that investigates the performance of LLM-based robot systems by combining formal logic with neural confidence estimation to verify the safety of household robot tasks. The system utilizes multiple models and provides both a baseline LLM approach (BASE_LLM) and an advanced neurosymbolic approach (NERT) for comparative evaluation.

## Installation

### Prerequisites
- Python 3.9-3.11
- Conda package manager
- OpenAI or Gemini API key
- Linux (for AI2-THOR simulation; optional on Windows)

### Setup

1. **Create conda environment:**
```bash
conda create -n nert python=3.11 -y
conda activate nert
#if prompted, follow the steps below to initialize conda for your shell
```

2. **Initialize conda for your shell:**
```bash
# For bash/Git Bash
conda init bash

# For Windows Command Prompt
conda init cmd.exe

# For PowerShell
conda init powershell
```

Restart your terminal, then:
```bash
conda activate nert
```

3. **Install dependencies:**
```bash
# Linux only: Install AI2-THOR dependency
conda install -c conda-forge spot

# All platforms: Install Python packages
pip install -r requirements.txt
pip install -e .
```

4. **Verify installation:**
```bash
pip list
```

5. **Configure API keys:**
```bash
cp .env.example .env
# create both .env.example and .env and add OPENAI_API_KEY or GOOGLE_API_KEY
#OPENAI_API_KEY=
#GOOGLE_API_KEY=
```

## Usage

### Web Interface

```bash
cd interface
python app.py
```

Navigate to `http://localhost:5000`

**Interface features:**
- Task input with optional scene context
- Model selection (GPT-4, GPT-4o, Gemini)
- Room type and floor plan selection (Kitchen 1-30, Living 201-230, Bedroom 301-330, Bathroom 401-430)
- Real-time pipeline visualization
- AI2-THOR simulation execution (Linux only)

**Interface Pipeline**

1. **Safety Checking**: Neurosymbolic verification combining formal rules and neural confidence
2. **Invariant Generation**: Generates PDDL preconditions, postconditions, and invariants
3. **Grounding Verification**: Validates required objects and skills exist in scene using hybrid semantic matching
4. **Code Generation**: Produces executable Python code with safety assertions
5. **Code Verification**: Symbolic execution validates invariants are maintained
6. **Task Execution**: Executes in AI2-THOR simulator (Linux only)

### AI2-THOR Simulation for Interface

**Supported:** Linux with X11 display
**Not supported:** Windows (Unity rendering limitations)

On Windows, NERT uses cached scene data for object verification. All other features (safety checking, planning, code generation, symbolic verification) work normally.

### Linux Setup for AI2-THOR
```bash
# Install X11 dependencies
sudo apt-get update
sudo apt-get install -y xorg libxcb-randr0-dev libxrender-dev \
    libxkbcommon-dev libxkbcommon-x11-0 libavcodec-dev libavformat-dev \
    libswscale-dev
```

## Troubleshooting

**API key not found**
- Set `OPENAI_API_KEY` or `GOOGLE_API_KEY` in `.env` file

**spaCy model not found**
- System uses heuristic fallback automatically
- To install: `python -m spacy download en_core_web_sm`

**AI2-THOR not available (Windows)**
- Expected behavior
- System uses scene cache for object verification
- For full simulation, use Linux

**Scene objects not updating**
- Ensure floor plan number is valid (Kitchen 1-30, Living 201-230, Bedroom 301-330, Bathroom 401-430)
- Check `data/scene_cache.json` exists

**Grounding verification showing wrong counts**
- System extracts objects from task text (scene-agnostic)
- Missing objects correctly block execution

**Port 5000 already in use**
- Change port in `interface/app.py` `app.run(port=5001)`

## Model Support

**OpenAI:** gpt-4, gpt-4o, gpt-4-turbo, gpt-3.5-turbo
**Google:** gemini-2.0-flash. gemini-2.5-pro, gemini-1.5-pro-002 (offline September 25)

Configure in web interface or set default in the root `config.yaml`.

**Note**
- Web Interface: Uses `config.yaml` (root directory)
- Benchmark Scripts: Use `experiments/config.yaml` (primary) or `--config` flag

**View logs:**
- Set logging level in `interface/app.py` line 30: `level=logging.DEBUG`
- Execution narrative visible at INFO level
- Detailed traces at DEBUG level

## Training Contrastive Model

If you have your training dataset or want to reuse our dataset:

```bash
cd models/contrastive
python train_contrastive.py --data training_1200.csv --epochs 10
```

## Experiments/Benchmarks

**Configuration before experiments**

- Change any necessary details at experiments/config.yaml. **Recommended** to the run the default benchmark first to get a feel of it before adjusting configuration settings for more exploration

 ```yaml 
# LLM Settings
llm:
  model: "gpt-4o"  # Available: gpt-4o, gpt-4-turbo, gpt-3.5-turbo, gemini-2.0-flash, etc.
  temperature: 0.0
  max_tokens: 2000
  api_key_env: "OPENAI_API_KEY"  #Or GOOGLE_API_KEY # Environment variable name

# Neural Model Settings - If using NERT Mode
neural:
  model_path: "models/trained_encoder.pt"
  embedding_dim: 768
  confidence_threshold: 0.5
  confidence_unsafe_threshold: 0.3
  support_ratio_accept_threshold: 0.4
  confidence_accept_threshold: 0.2

# AI2-THOR Settings
ai2thor:
  enabled: false  # Set to true if AI2-THOR is installed
  default_floor_plan: 1
  width: 600
  height: 600
  headless: true

# Logging
logging:
  level: "INFO"
  log_dir: "data/logs"
  save_traces: true

# Execution
execution:
  parallel_workers: 20
  rate_limit_delay: 0.1
  cache_scenes: true
```

**Run benchmarks:**

- Default using NERT mode: Output: results/benchmark_results_nert.csv, results/metrics_nert.json

```bash
cd experiments
python run_benchmark.py --data benchmark_621.csv --workers 20
```

- BASE_LLM mode: Output: results/benchmark_results_base_llm.csv, results/metrics_base_llm.json 

```bash
cd experiments
python run_benchmark.py --data benchmark_621.csv --workers 20 --mode base_llm
```

- With temperature control for either NERT or Base_LLM mode: Note - Temperature: CLI > config.yaml > 0.0 (default)

```bash
cd experiments
python run_benchmark.py --data benchmark_621.csv --mode base_llm --temp 0.0
```

- With custom config

```bash
cd experiments
python run_benchmark.py --config my_config.yaml --mode nert
```

**Parameters Explained:**

  | Parameter   | Options           | Description                             |
  |-------------|-------------------|-----------------------------------------|
  | --data      | benchmark_621.csv | CSV file with task/ground_truth columns |
  | --mode      | nert or base_llm  | Classification method                   |
  | --selective | (flag)            | Required to generate curves             |
  | --workers   | 1-50              | Parallel workers (20 recommended)       |
  | --temp      | 0.0-1.0           | LLM temperature (optional)              |
  | --config    | config.yaml       | Config file path (optional)             |

- Full commands above and selection prediction highlighted below:

```bash
#NERT Mode with Selective Prediction:
python run_benchmark.py --data benchmark_621.csv --mode nert --selective --workers 20 --temp 0.0
```

```bash
#Base LLM Mode with Selective Prediction:
python run_benchmark.py --data benchmark_621.csv --mode base_llm --selective --workers 20
```

**Analyze results: Shows individual results + side-by-side comparison if both exist**
```bash
cd experiments
python analyze_results.py
```

## License

[Under Review]

## Citation

```bibtex
@inproceedings{nert2025,
  title={NERT: Neurosymbolic Embodied Reasoning for Task Planning},
  author={Anonymous},
  year={2025}
}
```
