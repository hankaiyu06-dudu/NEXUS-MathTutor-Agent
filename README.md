# NEXUS: Separating Educational Memory from Language Models for AI Tutoring

> **LLM ≠ Memory**
> NEXUS externalizes learner cognition into an interpretable graph-based state engine, enabling long-term educational memory beyond the context window.

NEXUS is an AI tutoring architecture for mathematics education that models learning not as conversational history, but as a dynamic state evolving on a knowledge graph. Rather than relying on the language model to remember the learner, NEXUS maintains an explicit cognitive state that persists across exercises, sessions, and learning trajectories.

---

## 1. Vision: A New Topological Medium

Walter Rudin's *Principles of Mathematical Analysis* is widely regarded as a masterpiece of mathematical exposition. Yet for many students encountering rigorous mathematics for the first time, it can be overwhelming. A textbook, no matter how elegant, remains fundamentally static: a student struggling with continuity and a student struggling with compactness receive the same page, in the same order, regardless of their individual needs.

Human tutors solve this problem naturally. They remember what the learner knows, identify missing prerequisites, and decide which idea should come next.

Large Language Models promise to scale this experience, but conversational systems face a fundamental engineering limitation: **context is not cognition**. Over long learning trajectories, an AI gradually loses access to historical struggles, prerequisite mastery, and curriculum structure.

NEXUS explores a different paradigm.

Instead of organizing educational content as linear pages, we represent it as a navigable graph of mathematical ideas. The graph serves as a persistent representation of curriculum structure, while the AI tutor acts as an adaptive guide through that space.

We view NEXUS not merely as an application, but as a potential medium for educational content. In this medium, learning becomes a process of navigating and constructing conceptual connections rather than consuming a fixed sequence of pages.

Our architectural philosophy is inspired by Terence Tao's minimalist approach in *Math 245A: Problem Solving Strategies* and the observation that mathematical understanding emerges from relationships between ideas. Rather than treating educational content as unstructured data dumps within a massive vector database (RAG), NEXUS emphasizes explicit topology, interpretable dependencies, and guided progression through a structured concept graph. We ensure that mathematical growth happens exactly where it should: in the cognitive sparks when a student successfully connects disparate ideas.

---

## 2. Research Motivation & Core Hypothesis

Current AI tutors often rely primarily on conversational context.

While effective for short interactions, such systems generally lack an explicit representation of:
* prerequisite relationships,
* concept dependencies,
* long-term mastery evolution,
* curriculum topology.

NEXUS separates two distinct responsibilities:
* **Cognitive State Tracking** — deterministic tensor engine
* **Natural Language Interaction** — LLM-based agent

The central hypothesis is:
> Educational memory should be maintained explicitly as a structured cognitive state, while language models should focus on interpretation, feedback, and interaction.

This separation aims to provide greater interpretability, consistency, and pedagogical control than approaches that rely exclusively on neural memory mechanisms.

### Research Positioning
NEXUS does not attempt to replace neural memory systems. Instead, it investigates whether an explicit graph-based cognitive state layer can complement language models by providing a transparent and persistent representation of learner understanding.

---

## 3. Example State Evolution
*Illustrative example of cognitive-state propagation.*

### Student Response
> "The derivative is $f'(x)=3x^2-6x+a$. Since $x=1$ is a minimum, $a=3$."

The student correctly determines the parameter value but fails to analyze the derivative sign structure required for monotonicity.

### Evaluator Output
**Correct**
* Derivative computation
* Critical-point condition

**Incorrect**
* Quadratic sign analysis

### Tensor State Update
```text
[Mastery Before]
Derivative Calculation      0.92
Critical Points             0.81
Quadratic Signs             0.62

[Direct Update]
Derivative Calculation      0.96 (+0.04)
Critical Points             0.87 (+0.06)
Quadratic Signs             0.38 (-0.24)

Weak Node Detected:
Quadratic Signs

[Propagation Through DAG]
Monotonicity Analysis       0.54 → 0.43
Optimization Problems       0.71 → 0.65
```

### Tutor Response
> "You correctly found $a=3$. Now graph $f'(x)=3(x-1)^2$. Is the derivative ever negative? What does a derivative that is always nonnegative imply about monotonicity?"

Rather than revealing the solution, the tutor generates a targeted intervention centered on the detected weak concept.

---

## 4. System Architecture

NEXUS operates on two coupled loops. The tensor engine maintains the long-term educational memory, while the language models read from and write to this state.

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

* **Macro Loop:** Determines *what* should be learned next by navigating the curriculum graph.
* **Micro Loop:** Determines *how* it should be learned by generating scaffolded interventions for weak nodes.

---

## 5. Implementation: The $k \times N \times 3$ Cognitive Tensor

NEXUS represents curriculum structure and learner state using a PyTorch tensor of shape:

$$k \times N \times 3$$

where:
* $k$ = number of foundational mathematical tools
* $N$ = total number of cognitive nodes

The tensor contains three layers:
* **Layer 0 — Topological Dependencies:** A compressed prerequisite matrix encoding how strongly node $j$ depends on tool $i$.
* **Layer 1 — Static Difficulty:** The intrinsic logical or algebraic complexity associated with each node.
* **Layer 2 — Dynamic Mastery:** A continuously updated estimate of learner proficiency across foundational tools.

### Topological Compression
A naive adjacency matrix requires $O(N^2)$ storage. In practice, foundational tools connect to exercises, but tools rarely depend on tools and exercises rarely depend directly on exercises. 

By exploiting this bipartite structure, NEXUS compresses the topology into a dense $k \times N$ representation, reducing spatial complexity to $O(kN)$ while preserving prerequisite information.

### Readiness Computation
For a candidate exercise:
$$\text{Readiness Score} = \text{Dependency Vector} \cdot \text{Mastery Vector}$$

This computation is performed directly in GPU memory and provides a fast estimate of whether a problem lies within the learner’s current zone of productive challenge.

---

## 6. Relation to Knowledge Tracing

Traditional Knowledge Tracing methods, including Bayesian Knowledge Tracing (BKT) and Deep Knowledge Tracing (DKT), estimate mastery through probabilistic or learned latent representations.

NEXUS takes a different approach. Instead of learning a hidden representation of student state, it maintains an explicit graph-structured cognitive model and performs deterministic propagation over prerequisite relationships. 

The goal is not necessarily superior predictive performance, but rather:
* interpretability,
* transparency,
* controllability,
* curriculum-aware reasoning.

NEXUS can therefore be viewed as a complementary perspective on learner modeling rather than a replacement for existing KT methods.

---

## 7. Current Status

**Implemented**
* Cognitive tensor engine with $O(kN)$ topology compression
* Knowledge graph propagation framework
* Gemma-based soft-grading evaluator
* Socratic tutoring workflow
* Local Gradio interface
* Structured JSON extraction pipeline

**In Progress**
* Large-scale problem-bank integration
* Automated graph construction from mathematical resources
* Empirical comparison against KT baselines
* Visualization tools for cognitive-state evolution

---

## 8. Future Directions

Potential future research directions include:
* **Hybrid Cognitive Models:** Combining deterministic state propagation with learned student models.
* **Adaptive Graph Rewiring:** Updating prerequisite structures based on aggregate learner behavior.
* **Multi-Agent Tutoring Systems:** Specialized agents for hint generation, verification, planning, and peer-style interaction.
* **Automatic Curriculum Extraction:** Building concept graphs directly from textbooks, lecture notes, and problem collections.

---

## 9. Design Philosophy

NEXUS is built around a simple observation:
**Learning is not the accumulation of isolated facts. It is the gradual construction of connections between ideas.**

Traditional educational resources are organized as linear sequences of pages. Human tutors, however, rarely teach linearly. They continuously assess prerequisite understanding, identify conceptual gaps, and decide which connection should be built next. NEXUS attempts to externalize this process.

Instead of treating educational memory as conversational history, it represents learner state explicitly within a graph-structured cognitive model. The knowledge graph provides the curriculum topology, the tensor engine maintains long-term educational memory, and the language model serves as an adaptive interface between the learner and the graph.

This separation reflects the central philosophy of the project:
> **Language Models ≠ Educational Memory**

Language models are exceptionally effective at explanation, dialogue, and guidance. Educational memory, however, benefits from explicit structure, persistence, and interpretability. NEXUS explores what becomes possible when these two responsibilities are treated as distinct components of the same system.

---

## 10. Quick Start

**Installation**
```bash
git clone [https://github.com/your-username/NEXUS-Math-Agent.git](https://github.com/your-username/NEXUS-Math-Agent.git)
cd NEXUS-Math-Agent
pip install -r requirements.txt
```

**Run**
```bash
python app.py
```

**Hardware Requirements**
* NVIDIA GPU recommended
* 16 GB VRAM for local inference
* NF4 4-bit quantization via BitsAndBytes for memory-efficient deployment

---

## 11. Closing Remarks

NEXUS is currently a research prototype. The project does not claim to solve tutoring, knowledge tracing, or personalized education. Instead, it explores a specific question:

*Can learner understanding be represented as an explicit cognitive state that exists independently of the language model?*

If the answer is yes, AI tutors may evolve beyond conversational assistants and become navigators of structured knowledge spaces. In that vision, educational resources are no longer static documents. They become living topologies of ideas.

**License**
MIT License | Dedicated to the modernization of mathematics education through interpretable cognitive-state modeling.
