# Post-Training: Fine tuning and RL for Large Language Models

A comprehensive hands-on exploration of post-training techniques and reinforcement learning methods for fine-tuning large language models. This repository contains 5 structured modules with interactive Jupyter notebooks, utility libraries, and real-world evaluation frameworks.

### Core Learning Objectives
This covers the complete post-training lifecycle for LLMs:

1. **Post-training in the LLM Lifecycle** — Understand where post-training fits in model development, how it transforms base models into production-ready systems, and the tradeoffs between alignment and capability preservation.

2. **Core Fine-tuning Techniques**
   - **Supervised Fine-tuning (SFT):** Adapt models to follow instructions and improve task-specific performance
   - **Reward Modeling:** Build reward functions that capture desired behaviors
   - **RLHF (Reinforcement Learning from Human Feedback):** Use human preferences to train reward models and optimize model behavior
   - **Reinforcement Learning:** Use RL algorithms (PPO, GRPO) to optimize model behavior based on rewards
   - **LoRA (Low-Rank Adaptation):** Parameter-efficient fine-tuning to reduce compute costs

3. **Evaluation and Error Analysis** — Design effective evaluations, detect failure modes (calculation errors, reasoning failures, format issues), cluster similar failures, apply statistical testing, and detect reward hacking.

4. **Data for Post-training** — Prepare high-quality training data, generate synthetic examples, combine multiple fine-tuning approaches, and balance data curation with reward signals.

5. **Production Pipelines** — Deploy models reliably, monitor performance with automated alerts, implement data feedback loops, set go/no-go rules, and optimize for latency and cost.

### Module Breakdown

| Module | Focus | Key Topics |
|--------|-------|-----------|
| **M1** | Base vs Fine-tuned vs RL Models | Model comparison, evaluation metrics, safety vs correctness tradeoffs |
| **M2Lab1** | Supervised Fine-tuning | SFT pipeline, data preparation, training with Hugging Face Trainer |
| **M2Lab2** | GRPO Fine-tuning | Group Relative Policy Optimization, reward modeling |
| **M3** | Evaluation & Debugging | Error analysis, clustering similar failures, statistical significance testing |
| **M4** | Math Reasoning | Specialized reasoning evaluation, solution generation, benchmark datasets |
| **M5** | Production Analysis | Monitoring, optimization, latency analysis, alerting thresholds |

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab
- CUDA-capable GPU (recommended, but CPU works for smaller models)
- ~50GB disk space for model weights

### Installation

1. **Clone or download this repository:**
   ```bash
   git clone https://github.com/anuragdogra2192/PostTrainingFineTuning-RLForLLMs.git
   cd PostTrainingFineTuning-RLForLLMs
   ```

2. **Install dependencies:**
   ```bash
   pip install torch transformers jupyter jupytext matplotlib pandas numpy scipy scikit-learn sentence-transformers python-dotenv
   ```

3. **Download models** (or use Docker environment):
   - Models should be placed in a `models/` directory or `/app/models/` (Docker)
   - Pre-downloaded models include: `llama-3.2-8b`, `all-MiniLM-L6-v2`, and task-specific models
   - Set `HF_HOME` environment variable if models are in a custom location

4. **Launch Jupyter:**
   ```bash
   jupyter lab
   ```
   Then navigate to the desired module and open its notebook.

### For Each Module:

1. **Read the Literature PDF** (`Literature/PostTraining_LLMs_M1-5.pdf`)
   - Provides theoretical background and context
   - Reference while working through notebooks

2. **Open the Module Notebook**
   - Start with the markdown cells for instructions
   - Read hints and background information carefully

## Module Details

### M1: Model Comparison
**Goal:** Understand differences between base, fine-tuned, and RL models.

**Key Tasks:**
- Load multiple model variants
- Evaluate on math reasoning dataset
- Analyze tradeoffs between safety and correctness
- Compare performance metrics

**Utilities:** Basic model loading and evaluation

---

### M2Lab1: Supervised Fine-tuning
**Goal:** Learn to fine-tune models using supervised learning.

**Key Tasks:**
- Prepare training data from various sources
- Set up Hugging Face Trainer with appropriate hyperparameters
- Monitor training progress
- Evaluate fine-tuned model performance

**Utilities:** Data loaders, grading functions, model serving

---

### M2Lab2: GRPO Fine-tuning
**Goal:** Explore reinforcement learning-based fine-tuning with reward signals.

**Key Concepts:**
- **GRPO (Group Relative Policy Optimization):** Train models to optimize against learned reward signals by comparing group performance
- **Reward Modeling:** Learn to score outputs based on desired behaviors
- **Policy Optimization:** Use RL algorithms to iteratively improve model responses

**Key Tasks:**
- Design and implement reward models
- Set up GRPO training loops
- Apply policy optimization to improve model outputs
- Compare RL-fine-tuned models with SFT baselines

**Utilities:** RL training loops, reward scoring, policy gradient computation

---

### M3: Evaluation & Debugging
**Goal:** Develop robust evaluation frameworks and understand failure modes.

**Key Tasks:**
- Evaluate models across multiple test sets
- Identify and categorize errors
- Use statistical tests (McNemar's) for significance
- Cluster similar errors to find patterns

**Utilities:** 
- Error analysis with LLM-based and rule-based approaches
- Clustering with sentence transformers
- Caching for evaluation results
- McNemar's statistical test

**Error Categories:**
- `calculation_error` — Math computation mistakes
- `reasoning_error` — Logical reasoning issues
- `incomplete_solution` — Partial answers
- `format_error` — Wrong output format
- `other` — Miscellaneous errors

---

### M4: Math Reasoning
**Goal:** Focus on specialized evaluation for math problem-solving.

**Key Tasks:**
- Generate solutions for math problems
- Extract and verify final answers
- Evaluate reasoning quality
- Benchmark against baselines

**Utilities:** Math-specific solution generation, answer extraction

---

### M5: Production Analysis
**Goal:** Learn to monitor and optimize LLMs in production.

**Key Tasks:**
- Parse production logs (timestamps, latency, errors)
- Compute key metrics (p95 latency, error rate, satisfaction)
- Set alerting thresholds
- Identify bottlenecks and optimization opportunities

**Utilities:**
- Data processing functions (`parse_timestamps`, `normalize_errors`, `compute_metrics`)
- Unit tests for validation (`unit_tests.py`)
- Alert checking with customizable thresholds

**Key Metrics:**
- `avg_latency_ms` — Mean response time
- `p95_latency_ms` — 95th percentile latency
- `avg_tokens` — Average output length
- `error_rate_pct` — Percentage of failed requests
- `avg_satisfaction` — User satisfaction (1-5 scale)

---

## Project Structure

```
PostTrainingFineTuning&RLForLLMs/
├── README.md                    # This file
├── M1/                          # Module 1: Model comparison
│   └── M1_G1_Inspecting_Finetuned_vs_Base_Model.ipynb
├── M2/                          # Module 2: Fine-tuning labs
│   ├── M2Lab1/                  # Supervised fine-tuning
│   │   ├── M2_G1_fine_tune_lab_student.ipynb
│   │   └── utils/               # SFT utilities
│   └── M2Lab2/                  # GRPO fine-tuning
│       ├── M2_G2_grpo_finetune_lab_student.ipynb
│       └── utils/
├── M3/                          # Module 3: Evaluation & debugging
│   ├── M3_G1_evaluation_and_debugging.ipynb
│   ├── utils/
│   │   └── utils.py             # Comprehensive evaluation utilities
│   └── lab_results/             # Evaluation outputs
├── M4/                          # Module 4: Math reasoning
│   ├── M4_G1_mathreasoning_student.ipynb
│   ├── utils.py
│   └── evaluation_results.json  # Results from evaluation
├── M5/                          # Module 5: Production analysis
│   ├── M5_G1_Analysis_and_Optimization_in_Production_student.ipynb
│   ├── utils/
│   │   ├── util.py              # Production utilities
│   │   └── unit_tests.py        # Unit tests for M5 functions
│   └── images/                  # Visualizations
├── Literature/                  # Reference PDFs
│   ├── PostTraining_LLMs_M1.pdf
│   ├── PostTraining_LLMs_M2.pdf
│   ├── PostTraining_LLMs_M3.pdf
│   ├── PostTraining_LLMs_M4.pdf
│   └── PostTraining_LLMs_M5.pdf
└── certificate.png              # Course completion certificate
```

## Key Concepts & Techniques

### Fine-tuning Approaches
- **Supervised Fine-tuning (SFT):** Train on high-quality instruction-following examples to adapt base models for specific tasks
- **Reward Modeling:** Learn to score model outputs based on desired behaviors (safety, accuracy, relevance)
- **RLHF (Reinforcement Learning from Human Feedback):** Collect human preferences, train reward models from preferences, and use RL to optimize against those rewards
- **GRPO (Group Relative Policy Optimization):** Optimize based on relative performance between groups of outputs
- **RL Methods:** Policy optimization algorithms (PPO, GRPO) that leverage reward signals to improve model behavior
- **LoRA (Low-Rank Adaptation):** Parameter-efficient fine-tuning that reduces memory and computation requirements

### Evaluation Techniques
- **Metric-based evaluation:** Accuracy, latency, token counts, and reasoning quality
- **Error analysis:** Categorize and cluster failure modes to identify systematic problems
- **Reward signal validation:** Detect reward hacking and misalignment between metrics and desired behavior
- **Red teaming:** Test model robustness and identify edge cases
- **Statistical significance:** McNemar's test for reliable model comparisons
- **Production monitoring:** Real-time performance tracking with automated alerts and feedback loops

### Data Strategy for Post-training
- **Data curation:** Prepare high-quality instruction-following and reasoning examples
- **Synthetic data generation:** Create diverse examples to improve model robustness
- **Data and reward balancing:** Combine curated examples with learned reward signals for optimal performance
- **Feedback loops:** Iterate on data quality using production logs and user feedback

### Key Libraries
- **transformers:** Model loading and inference
- **torch:** Deep learning framework
- **sentence-transformers:** Embedding-based error clustering
- **scikit-learn:** Statistical tools and clustering
- **pandas:** Data processing

## 📊 Expected Learning Outcomes

By completing this course, you will understand:
- How to evaluate and compare LLM variants
- Fine-tuning strategies and their tradeoffs
- Error analysis and debugging techniques
- Statistical methods for model comparison
- Production monitoring and optimization
- Real-world considerations for deploying LLMs

## 🎓 Course Certificate
`certificate.png`

## 📧 Contact
Anurag Dogra
- **Email:** anuragdogra2192@gmail.com
---

