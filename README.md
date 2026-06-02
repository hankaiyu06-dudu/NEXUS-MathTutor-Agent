# NEXUS: The AI Agent Interactive Exercise Book

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Gemma](https://img.shields.io/badge/Gemma-4285F4?style=for-the-badge&logo=google&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

## Vision: A New Medium for Learning Resources
Traditional exercise books and problem sets are static dead-ends. They present linear information without understanding the student's cognitive state. 

NEXUS envisions a paradigm shift: The AI Agent Interactive Exercise Book. 
We redefine mathematical learning resources not as flat text, but as a dynamic, topological knowledge graph navigated by a resident AI Agent. By transforming static problems into interactive nodes, NEXUS creates a responsive backend engine that actively diagnoses, scaffolds, and evolves with the learner's cognitive map.

---

## Core Design & Architecture

NEXUS operates entirely as a headless algorithmic engine, powered by a dual-track architecture combining deep mathematical modeling with large language model inference.

### 1. Topological Graph & 3D Cognitive Tensor
Instead of a standard database, learning resources are managed via a Topological Graph Backbone. 
* Cognitive Tensor Engine: A 3D PyTorch tensor dynamically tracks the user's mastery across interconnected mathematical concepts.
* Algorithmic Pathfinding: As users interact with problems, the engine backpropagates evaluation scores through the tensor, automatically calculating conceptual readiness and diagnosing core bottlenecks in real-time.

### 2. Resident Socratic Agent (Powered by Gemma)
The engine is driven by a local LLM inference module optimized for heuristic guidance.
* Targeted Scaffolding: The agent reads the user's graph state and generates targeted micro-problems to bridge cognitive gaps based on identified weak nodes.
* Methodology Over Final Answer: The LLM evaluator is explicitly prompted to generously reward correct analytical logic, structural approaches, and substitution methods, even if the final algebraic calculation is incomplete.

### 3. The Mastery Vault (Knowledge Persistence)
Learning is structured through rigorous state tracking.
* When a user achieves a passing threshold (e.g., 90% mastery) on a specific conceptual node, its state is permanently updated in the Mastery Vault.
* The system allows dynamic injection of user insights, updating the tensor weights to reflect accumulated structural knowledge.

---

## Quick Start (Engine API)

NEXUS is designed as a standalone Python library/engine that can be integrated into any environment or notebook.

### Prerequisites
* Python 3.10+
* PyTorch 2.0+
* Transformers & BitsAndBytes (for 4-bit Gemma quantization)

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/NEXUS-Math-Agent.git](https://github.com/your-username/NEXUS-Math-Agent.git)
   cd NEXUS-Math-Agent
