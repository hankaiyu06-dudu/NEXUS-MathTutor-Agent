# NEXUS: The AI Agent Interactive Exercise Book

## 1. Demo Snapshot
What happens when a student makes a conceptual error in NEXUS?

> **Student Task:** Find the monotonic intervals of f(x) = x^3 - 3x^2 + ax + 1 given a local minimum at x=1.
> 
> **Student Input:** "The derivative is f'(x) = 3x^2 - 6x + a. Since x=1 is a minimum, f'(1) = 0, so a = 3. The intervals are (-inf, 1) and (1, inf)."
> 
> **NEXUS Agent Response:**
> * **Evaluator:** Extracts correct partial logic (a=3 is correct) but identifies a critical failure in analyzing the sign of f'(x) for monotonicity.
> * **Tensor Update:** Propagates +0.8 mastery to [Derivative Calculation] but -0.5 to [Sign Analysis of Quadratics].
> * **Tutor Action:** Generates a scaffolded micro-problem targeting the exact weak node: "You correctly found a=3, meaning f'(x) = 3(x-1)^2. But what does it mean for a function's monotonicity if its derivative is always >= 0? Try graphing f'(x) first."

---

## 2. Vision
Traditional exercise books are static dead-ends. NEXUS envisions a paradigm shift: The AI Agent Interactive Exercise Book. We redefine mathematical learning resources not as flat text, but as a dynamic, topological knowledge graph navigated by a resident AI Agent. By transforming static problems into interactive nodes, NEXUS creates a responsive backend engine that actively diagnoses, scaffolds, and evolves with the learner's cognitive map.

---

## 3. System Architecture
NEXUS operates on a dual-track loop: LLM inference for natural language understanding, and a deterministic tensor engine for state tracking.
```text
┌───────────────────────────────────────────────────────────────────┐
│                 MACRO LOOP: KNOWLEDGE NAVIGATION                  │
│                                                                   │
│  [Cognitive Tensor] ───────(Yields Candidate Pool)─────────────┐  │
│  ▶ Calculates DAG readiness                                    │  │
│  ▲                                                             ▼  │
│  │                                                     [Gemma-4]  │
│  │                                             ▶ Selects best task│
│  │(JSON Score Triggers Update)                                 │  │
│  │                                                             ▼  │
│  [Gemma-4 Evaluator] ◀──────(Evaluates Logic)──────── [User Input]│
│                                                                   │
└────────────────────────────────┬──────────────────────────────────┘
                                 │
                         (If Mastery is Low)
                     Tensor Extracts Weak Nodes
                                 │
┌────────────────────────────────▼──────────────────────────────────┐
│                 MICRO LOOP: SOCRATIC SCAFFOLDING                  │
│                                                                   │
│  [Gemma-4 Tutor] ───────────(Generates Micro-Problem)──────────┐  │
│  ▶ Injects weak nodes as context                               │  │
│  ▶ Prompts heuristic hints (No direct answers)                 ▼  │
│                                                         [User UI] │
└───────────────────────────────────────────────────────────────────┘
```

## 4. Implementation: The 3D Cognitive Tensor
We avoid traditional relational databases for student progress. Instead, the cognitive state is modeled natively in PyTorch as a 3D Tensor `shape=(N, N, 3)`, where `N` is the total number of concepts/nodes.

The three channels represent:
1. Adjacency Matrix (Topology): Represents the Directed Acyclic Graph (DAG) of prerequisite knowledge (1.0 if a dependency exists, 0 otherwise).
2. Difficulty Weights: The inherent algebraic or logical complexity of the node.
3. Mastery State: The dynamically updated confidence score (0.0 to 1.0) of the student's current proficiency.

State Propagation:
When a student is evaluated, mastery updates do not rely on gradient descent. Instead, the engine uses a weighted exponential moving average to propagate the state updates through neighboring prerequisite nodes via the adjacency matrix, ensuring deterministic and interpretable cognitive tracking.

---

## 5. Socratic Agent Workflow
The resident tutor (powered by a 4-bit quantized Gemma model) is strictly prompted against revealing final answers. 

The Workflow:
1. Context Injection: The LLM receives the current problem, the student's input, and the explicitly diagnosed weak nodes from the Cognitive Tensor.
2. Partial Credit Policy: The prompt strictly instructs the LLM to generously reward correct analytical logic and structural substitution methods, even if the final algebraic calculation fails.
3. Scaffolding: The LLM dynamically generates scaffolded micro-problems targeting the identified topological bottleneck.

---

## 6. Technical Stack
* Core Engine: PyTorch (Tensor manipulation, DAG representation)
* LLM Integration: Hugging Face transformers, bitsandbytes (NF4 double quantization for VRAM efficiency)
* Data Structure: JSON (Knowledge node registries)

---

## 7. Quick Start
Designed as a standalone Python engine optimized for Kaggle/Local GPU environments.

### Installation
```bash
git clone [https://github.com/your-username/NEXUS-Math-Agent.git](https://github.com/your-username/NEXUS-Math-Agent.git)
cd NEXUS-Math-Agent
pip install -r requirements.txt
