# NEXUS: Separating Educational Memory from Language Models for AI Tutoring

> **LLM ≠ Memory | LLM ≠ Curriculum**
NEXUS is an agent that manages knowledge graphs. It stores the knowledge graph as an adjacency matrix. The user's understanding is maintained in a separate layer. Through this architecture, NEXUS develops a long-term comprehension of a user's mathematical proficiency. This enables highly personalized problem-solving training and guidance. This approach overcomes existing limits. It builds a highly interpretable, long-term educational architecture.

---

## 1. Vision: A New Topological Medium

Walter Rudin's Principles of Mathematical Analysis is a masterpiece. However, it often overwhelms students facing difficult mathematical problems for the first time. It's too hard for freshman.The linear and static nature of a traditional textbook is simply too challenging. Students frequently encounter overly difficult problems, or repeatedly come across questions they already completely understand; while their time is wasted, they also develop a fear of mathematics or become bored.

While human tutors are able to keep the student's learning progress in mind and flexibly adjust the teaching content, and Large Language Models (LLMs) hold the promise of popularizing this personalized teaching experience, they face the practical problem of 'context amnesia' in real-world applications. During this process, AI will gradually forget the past difficulties of the student and their prerequisite foundational knowledge.

NEXUS explores a completely new paradigm. We drew inspiration from problem solving math instruction. Based on this, we transformed static educational content. They become a network of mathematical ideas that can guide students to explore freely. In this new medium, learning turns into an active process. Students can use this to continuously build connections between concepts. Artificial intelligence provides guidance here. It can clearly track the cognitive states of the students.

---

## 2. Core Architecture: Externalized Knowledge & State

Conventional AI tutoring systems typically rely on the internal parameters of Large Language Models to store both instructional knowledge and dialogue history. NEXUS, however, enforces a strict separation of these domains through a tripartite architectural design:

* **Tensor Engine (Student Memory):** Tracks dynamic mastery. *Determines WHAT the student needs.*
* **Skill Notes Repository (Domain Knowledge):** Stores canonical solutions, prerequisite concepts, and Socratic hints. *Determines WHAT should be taught.*
* **LLM Agent (Reasoning Layer):** Operates over the retrieved context to generate dialogue and evaluates student logic. *Determines HOW to teach it.*

```text
                 ┌──────────────────────┐
              ┌─▶│   Cognitive Tensor   │ (Calculates Readiness & Weak Nodes)
              │  └──────────┬───────────┘
              │             │
              │   Select Weak Concepts
  Evaluates   │             │
   Logic &    │             ▼
   Updates    │  ┌──────────────────────┐
    State     │  │   Skill Notes Repo   │ (Retrieves Concept Explanations)
              │  └──────────┬───────────┘
              │             │
              │   Inject Context & Hints
              │             │
              │             ▼
              │  ┌──────────────────────┐
              └──┤   LLM Tutor Agent    │ (Generates Dialogue & Evaluates)
                 └──────────────────────┘
```

Therefore, the Large Language Model is not required to memorize the curriculum content or the student's learning history. It wanders through the knowledge graph, continuously performing logical evaluations to remember the impact of the student's proficiency level. It then feeds this updated data back into the cognitive state.

---

## 3. Example State Evolution

*How NEXUS processes a conceptual error and propagates the cognitive state.*

**Student Response:**
> "The derivative is $f'(x)=3x^2-6x+a$. Since $x=1$ is a minimum, $a=3$." *(Fails to analyze the sign structure for monotonicity).*

**Evaluator Output:**
> [x] Correct: Derivative computation, Critical-point condition
> [ ] Incorrect: Quadratic sign analysis

**Tensor State Update & Propagation:**
```text
[Mastery Before]
Derivative Calculation      0.92
Critical Points             0.81
Quadratic Signs             0.62

[Direct Update]
Quadratic Signs             0.38 (-0.24)  <-- Weak Node Detected

[Propagation Through DAG]
Monotonicity Analysis       0.54 → 0.43
Optimization Problems       0.71 → 0.65
```

**Tutor Response (Prompted by retrieved Skill Notes):**
> "You correctly found $a=3$. Now graph $f'(x)=3(x-1)^2$. Is the derivative ever negative? What does a derivative that is always nonnegative imply about monotonicity?"

---

## 4. Implementation: The $k \times N \times 3$ Cognitive Tensor

NEXUS represents curriculum structure and learner state using a PyTorch tensor of shape $k \times N \times 3$ ($k$ foundational tools, $N$ total cognitive nodes), containing three layers:
1. **Layer 0 (Topological Dependencies):** A compressed prerequisite matrix.
2. **Layer 1 (Static Difficulty):** Intrinsic logical complexity.
3. **Layer 2 (Dynamic Mastery):** Continuously updated learner proficiency.

**Topological Compression:** By exploiting the bipartite structure of curricula (tools connect to exercises, rarely tool-to-tool), NEXUS compresses the topology into a dense $k \times N$ representation. Since $k \ll N$ in typical curricula, this substantially reduces memory requirements compared with a full adjacency matrix, dropping spatial complexity to $O(kN)$.

**Readiness Computation:** A simplified readiness estimate is computed directly in GPU memory as the dot product between a problem's dependency vector and the learner's mastery vector:
$$\text{Readiness Score} = \text{Dependency Vector} \cdot \text{Mastery Vector}$$

---

## 5. Relation to Knowledge Tracing

Unlike traditional Knowledge Tracing (e.g., BKT, DKT) that estimates mastery through probabilistic or learned latent representations, NEXUS maintains an explicit graph-structured cognitive model and performs deterministic propagation. 

The goal is not necessarily superior predictive accuracy, but rather **interpretability, transparency, and curriculum-aware reasoning**, serving as a complementary perspective on learner modeling.

---

## 6. Current Status & Future Directions

**Implemented:**
* $O(kN)$ Cognitive tensor engine & Knowledge graph propagation.
* Externalized Skill Notes retrieval pipeline.
* LLM-based soft-grading evaluator & Socratic tutoring workflow.
* Kaggle-optimized local Gradio interface.

**Future Directions:**
* **Hybrid Cognitive Models:** Combining deterministic propagation with learned student models.
* **Automated Curriculum Extraction:** Building concept graphs and Skill Notes directly from textbooks.
* **Adaptive Graph Rewiring:** Updating prerequisite structures based on aggregate learner behavior.

---

## 7. Quick Start

NEXUS is actively developed and executed within Kaggle's GPU-accelerated Jupyter environment.

**1. Clone the Repository**
```bash
git clone [https://github.com/your-username/NEXUS-Math-Agent.git](https://github.com/your-username/NEXUS-Math-Agent.git)
```

**2. Kaggle Environment Setup**
* Upload the core Jupyter Notebook (`nexus_core_engine.ipynb`) to a new Kaggle Notebook.
* Upload the knowledge database (`nexus_db.json`) to the Kaggle `/kaggle/input/` or `/kaggle/working/` directory.

**3. Execution**
* Ensure your Kaggle session has a GPU enabled (e.g., NVIDIA T4 x2 or P100).
* Run the cells sequentially to boot the cognitive tensor engine and launch the local Gradio interface. 
* *Note: The system utilizes NF4 4-bit dynamic quantization to fit large parameter language models securely within Kaggle's VRAM constraints.*

---
*License: MIT | Dedicated to the modernization of mathematics education through interpretable cognitive-state modeling.*
