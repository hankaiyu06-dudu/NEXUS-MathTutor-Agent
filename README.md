# NEXUS: The AI Agent Interactive Exercise Book

# =====================================================================
# NEXUS ENGINE: Core Architecture & UI Assembly (Production Stable)
# =====================================================================

import os
import re
import json
import glob
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ---------------------------------------------------------
# 1. Environment & Model Initialization
# ---------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "/kaggle/input/models/google/gemma-4/transformers/gemma-4-31b-it/1"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        quantization_config=quantization_config,
        low_cpu_mem_usage=True
    )
    model.tie_weights()
except Exception as e:
    print(f"Warning: Local model loading failed. Ensure execution in Kaggle environment. Details: {e}")

# ---------------------------------------------------------
# 2. The 3D Cognitive Tensor Engine
# ---------------------------------------------------------
class DualTrackCognitiveEngine:
    def __init__(self):
        paths = glob.glob("/kaggle/input/**/nexus_db.json", recursive=True)
        if paths: 
            self.db_path = paths[0]
        elif os.path.exists("/kaggle/working/nexus_db.json"): 
            self.db_path = "/kaggle/working/nexus_db.json"
        else: 
            # Fallback for demonstration if no DB is found
            self.db_path = None
            
        self.node_registry = []
        self.mastered_nodes = {}
        self.k, self.m, self.n = 0, 0, 0
        self.last_active_node_id, self.last_score = None, 1.0 
        
        if self.db_path:
            self._initialize_from_db()
        else:
            self._mock_initialization()

    def _initialize_from_db(self):
        with open(self.db_path, "r", encoding="utf-8") as f: 
            db_data = json.load(f)
            
        for base in db_data.get("base_nodes", []):
            self.node_registry.append({
                "is_empty": True, "id": f"BASE_{len(self.node_registry)}", 
                "name": base["name"], "type": base.get("type", "thought"), "difficulty": 0.5
            })
        self.k = len(self.node_registry)
        
        for content in db_data.get("content_nodes", []):
            self.node_registry.append({
                "is_empty": False, "id": content["id"], "title": content["title"], 
                "type": content["type"], "rich_text": content.get("rich_text", ""), 
                "linked_bases": content.get("linked_bases", []), "difficulty": content.get("difficulty", 0.5)
            })
        self.n = len(self.node_registry)
        
        # Tensor Shape: (N, N, 3) 
        # Channels: 0 -> Adjacency Matrix (Topology), 1 -> Difficulty, 2 -> Mastery State
        self.tensor = torch.zeros((self.n, self.n, 3), dtype=torch.float32)
        
        for i in range(self.k, self.n):
            self.tensor[i, i, 1] = self.node_registry[i]["difficulty"]
            for b_name in self.node_registry[i]["linked_bases"]:
                b_idx = next((j for j in range(self.k) if self.node_registry[j]["name"] == b_name), None)
                if b_idx is not None:
                    self.tensor[b_idx, i, 0] = 1.0
                    self.tensor[i, b_idx, 0] = 1.0
                    
        for i in range(self.n): 
            self.tensor[i, i, 2] = 0.1

    def _mock_initialization(self):
        self.n = 5
        self.k = 2
        self.tensor = torch.zeros((self.n, self.n, 3), dtype=torch.float32)
        self.node_registry = [{"id": f"mock_{i}", "title": f"Mock Node {i}", "rich_text": "Mock text", "name": f"Base {i}", "linked_bases": []} for i in range(self.n)]

    def propagate_state(self, content_id, score, alpha=0.6):
        """
        Propagates mastery updates through neighboring concept nodes using a 
        weighted exponential moving average, avoiding raw gradient descent backprop.
        """
        c_idx = next((i for i in range(self.n) if self.node_registry[i]["id"] == content_id), None)
        if c_idx is None or c_idx < self.k: 
            return
            
        self.last_active_node_id, self.last_score = content_id, score
        self.tensor[c_idx, c_idx, 2] = score
        
        # Exponential Moving Average for DAG state propagation
        for b_idx in range(self.k):
            if self.tensor[b_idx, c_idx, 0] > 0:
                current_mastery = self.tensor[b_idx, b_idx, 2].item()
                self.tensor[b_idx, b_idx, 2] = (1 - alpha) * current_mastery + alpha * score
        
        if score >= 0.9 and content_id not in self.mastered_nodes:
            self.mastered_nodes[content_id] = {
                "title": self.node_registry[c_idx]["title"],
                "rich_text": self.node_registry[c_idx]["rich_text"],
                "linked_bases": self.node_registry[c_idx].get("linked_bases", []),
                "notes": ""
            }

    def recommend_next_content(self):
        base_scores = torch.diagonal(self.tensor[:self.k, :self.k, 2])
        recs, last_vec = [], None
        
        if self.last_active_node_id and self.last_score < 0.6:
            lf_idx = next((i for i in range(self.n) if self.node_registry[i]["id"] == self.last_active_node_id), None)
            if lf_idx is not None:
                last_vec = self.tensor[:self.k, lf_idx, 0]
        
        for i in range(self.k, self.n):
            if self.node_registry[i]["id"] == self.last_active_node_id: 
                continue 
                
            deps = self.tensor[:self.k, i, 0]
            if torch.sum(deps).item() == 0: 
                continue
            
            readiness = torch.sum(base_scores * deps) / torch.sum(deps).item()
            weight = 1.0 - torch.abs(readiness - self.tensor[i, i, 1].item()) * 2.0
            
            if last_vec is not None and torch.sum(last_vec * deps).item() > 0: 
                weight += 0.3 * torch.sum(last_vec * deps).item()
            if self.tensor[i, i, 2].item() > 0.6: 
                weight -= (self.tensor[i, i, 2].item() - 0.6) * 1.5 
                
            recs.append({"index": i, "weight": weight})
            
        if not recs: 
            return {"status": "empty"}
            
        recs.sort(key=lambda x: x["weight"], reverse=True)
        node = self.node_registry[recs[0]["index"]]
        
        return {
            "status": "success", 
            "recommended_id": node["id"], 
            "title": node["title"], 
            "rich_text": node["rich_text"], 
            "linked_bases": node.get("linked_bases", [])
        }

    def diagnose_weak_bases(self, content_id):
        c_idx = next((i for i in range(self.n) if self.node_registry[i]["id"] == content_id), None)
        if c_idx is None:
            return []
    
        weak_nodes = []
        for b_idx in range(self.k):
            if self.tensor[b_idx, c_idx, 0] > 0:
                mastery = self.tensor[b_idx, b_idx, 2].item()
                weak_nodes.append({
                    "name": self.node_registry[b_idx]["name"],
                    "mastery": mastery
                })
    
        weak_nodes.sort(key=lambda x: x["mastery"])
        return weak_nodes

# ---------------------------------------------------------
# 3. LLM Inference Handlers & Parsing
# ---------------------------------------------------------
def extract_json(text):
    if not text: return None
    text = text.replace("```json", "").replace("```", "")
    start = text.find("{")
    if start == -1: return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{": depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0: return text[start:i+1]
    return None

def gemma_generate_json(system_instruction):
    try:
        chat = [{"role": "user", "content": system_instruction}]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.1)
        response_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        json_str = extract_json(response_text)
        
        if json_str is None:
            return json.dumps({"score": 30, "feedback": "System Error: Invalid JSON output.", "status": "INCORRECT"})
        return json_str
    except Exception as e:
        return json.dumps({"score": 30, "feedback": f"System Engine Error: {str(e)}", "status": "INCORRECT"})

def gemma_generate_text(system_instruction):
    try:
        chat = [{"role": "user", "content": system_instruction}]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=400, temperature=0.3)
        return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    except Exception as e: 
        return f"Tutor Engine encountered a runtime error: {e}"

# ---------------------------------------------------------
# 4. Text Processing Utilities
# ---------------------------------------------------------
def format_math_for_gradio(text):
    if not text: return ""
    text = "\n".join([line.lstrip() for line in text.split("\n")])
    text = re.sub(r"(?<![a-zA-Z0-9_\^\\\}])\{((?:[^{}]|\{[^{}]*\})*)\}", r"\\{\1\\}", text)
    text = text.replace("\\[", "\n$$\n").replace("\\]", "\n$$\n")
    text = text.replace("\\(", "$").replace("\\)", "$")
    text = re.sub(r"\\textbf\{(.*?)\}", r"**\1**", text)
    text = re.sub(r"\\textit\{(.*?)\}", r"*\1*", text)
    text = text.replace("\\begin{itemize}", "").replace("\\end{itemize}", "").replace("\\item", "• ")
    return text

def split_q_and_a(rich_text):
    parts = re.split(r'(\\textbf\{[解析证明]+[：\:]?\})', rich_text, maxsplit=1)
    if len(parts) >= 3: 
        return parts[0], parts[1] + parts[2]
    return rich_text, "*No official solution provided in the database.*"

# ---------------------------------------------------------
# 5. Interactive Flow Control
# ---------------------------------------------------------
def load_next_problem(session_mgr):
    if not session_mgr:
        return "System Offline.", "", gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), session_mgr
        
    res = session_mgr.recommend_next_content()
    if res.get("status") == "empty":
        return "### Graph Mastered.\nYou have conquered all available nodes in the matrix.", "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), session_mgr
    
    session_mgr._current_quest_id = res["recommended_id"]
    q_text, a_text = split_q_and_a(res["rich_text"])
    session_mgr._current_q_text = q_text
    session_mgr._current_a_text = a_text
    session_mgr._current_tools = res.get("linked_bases", [])
    
    return format_math_for_gradio(q_text), "", gr.update(visible=True, interactive=True), gr.update(visible=False), gr.update(visible=False), session_mgr

def submit_answer_and_evaluate(user_ans, chat_history, session_manager):
    if not user_ans: 
        yield gr.update(), chat_history, gr.update(), gr.update(), gr.update(), session_manager
        return

    q_id = getattr(session_manager, "_current_quest_id", None)
    q_text = getattr(session_manager, "_current_q_text", "Unknown")
    a_text = getattr(session_manager, "_current_a_text", "None")

    chat_history.append({"role": "user", "content": f"Submitted Solution:\n{user_ans}"})
    chat_history.append({"role": "assistant", "content": "*Gemma-4 is evaluating your logic. Please wait (approx. 10-20s)...*"})
    yield gr.update(interactive=False), chat_history, gr.update(), gr.update(), gr.update(), session_manager

    prompt = f"Evaluate student solution.\nProblem: {q_text}\nStudent Answer: {user_ans}\nReturn JSON only: {{\"score\": 0-100, \"feedback\": \"...\", \"status\": \"CORRECT|PARTIAL|INCORRECT\"}}"
    data = gemma_generate_json(prompt)
    
    if isinstance(data, str):
        try: data = json.loads(data)
        except: data = {"score": 30, "feedback": f"System Error: Invalid model output. Raw output: {data}", "status": "INCORRECT"}
            
    if not isinstance(data, dict):
        data = {"score": 30, "feedback": "System Error: Cognitive engine generation failed.", "status": "INCORRECT"}

    score = float(data.get("score", 30)) / 100.0
    feedback = data.get("feedback", "No feedback provided.")
    status = data.get("status", "INCORRECT")

    # State Propagation replaces raw backprop
    session_manager.propagate_state(q_id, score)
    
    prefix = "CORRECT" if status == "CORRECT" else ("PARTIAL" if status == "PARTIAL" else "INCORRECT")
    chat_history[-1] = {"role": "assistant", "content": f"[{prefix} | Score: {int(score*100)}]\n\n{feedback}"}

    yield gr.update(visible=False), chat_history, gr.update(visible=True), format_math_for_gradio(a_text), gr.update(visible=True), session_manager

def chat_with_tutor(user_msg, chat_history, session_manager):
    if not user_msg: 
        yield "", chat_history
        return
        
    q_text = getattr(session_manager, "_current_q_text", "Unknown")
    chat_history.append({"role": "user", "content": user_msg})
    chat_history.append({"role": "assistant", "content": "*Gemma-4 is formulating heuristic hints...*"})
    yield "", chat_history
    
    prompt = f"You are a top-tier math coach.\nCurrent Problem:\n{q_text}\n\nStudent Question:\n{user_msg}\n\nGive a heuristic hint. Do NOT reveal final answer."
    reply = gemma_generate_text(prompt) 
    
    chat_history[-1] = {"role": "assistant", "content": format_math_for_gradio(reply)}
    yield "", chat_history

def request_ai_hint(chat_history, session_manager):
    q_text = getattr(session_manager, "_current_q_text", "Unknown")
    weak_nodes = session_manager.diagnose_weak_bases(session_manager._current_quest_id)
    weak_text = "\n".join([f"- {x['name']} (mastery={x['mastery']:.2f})" for x in weak_nodes])

    chat_history.append({"role": "user", "content": "Request scaffolded hints"})
    chat_history.append({"role": "assistant", "content": "*Scanning cognitive topology to customize your hints...*"})
    yield chat_history
    
    prompt = f"You are an elite mathematics tutor.\nProblem:\n{q_text}\nWeak Concepts:\n{weak_text}\n\nIdentify bottleneck and generate 3 scaffolded micro-problems. Do NOT reveal final answer."
    reply = gemma_generate_text(prompt)
    
    chat_history[-1] = {"role": "assistant", "content": format_math_for_gradio(reply)}
    yield chat_history

# ---------------------------------------------------------
# 6. Gradio Application Rendering
# ---------------------------------------------------------
css = """
body, .gradio-container { background-color: #f8fafc !important; font-family: 'Inter', sans-serif !important; color: #0f172a !important; }
.prose, .prose p, .prose div, .prose li { color: #0f172a !important; font-size: 17px !important; line-height: 1.75 !important; }
.katex { font-size: 1.15em !important; color: #000000 !important; }
h1, h2, h3 { color: #0f172a !important; font-weight: 700 !important; }
.arena-container { border: 1px solid #e2e8f0 !important; padding: 32px !important; background: #ffffff !important; border-radius: 12px !important; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.05) !important; }
.tutor-container { border: 1px solid #e2e8f0 !important; background: #f8fafc !important; border-radius: 12px !important; padding: 24px !important; }
.gradio-container textarea, .gradio-container input { background-color: #ffffff !important; color: #0f172a !important; border: 1px solid #cbd5e1 !important; }
.message { background-color: #ffffff !important; border: 1px solid #e2e8f0 !important; color: #0f172a !important; }
.message.user { background-color: #f1f5f9 !important; }
.message .prose p, .message .prose span { color: #0f172a !important; }
"""

with gr.Blocks(css=css) as demo:
    engine_instance = DualTrackCognitiveEngine()
    session_state = gr.State(engine_instance)

    gr.Markdown("# NEXUS ENGINE 3.0 - STABLE RENDER VERSION")

    with gr.Row():
        with gr.Column(scale=5, elem_classes="arena-container"):
            gr.Markdown("### Problem Arena")
            
            problem_display = gr.Markdown(
                value="Booting system...",
                latex_delimiters=[
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": "$", "right": "$", "display": False},
                    {"left": "\\[", "right": "\\]", "display": True},
                    {"left": "\\(", "right": "\\)", "display": False}
                ]
            )
            
            user_answer = gr.Textbox(label="Your Solution", lines=8, interactive=True)
            
            with gr.Row():
                submit_btn = gr.Button("Submit Analysis")
                next_btn = gr.Button("Next Challenge")
            
            eval_result = gr.Markdown(visible=False)
            
            with gr.Column(visible=False) as official_sol_container:
                gr.Markdown("### Official Resolution")
                official_solution_text = gr.Markdown(
                    latex_delimiters=[
                        {"left": "$$", "right": "$$", "display": True},
                        {"left": "$", "right": "$", "display": False}
                    ]
                )

        with gr.Column(scale=4):
            with gr.Tabs():
                with gr.Tab("Resident Tutor", elem_classes="tutor-container"):
                    chatbot = gr.Chatbot(label="Tutor", height=480, show_label=False)
                    with gr.Row():
                        chat_input = gr.Textbox(show_label=False, placeholder="Ask something...", scale=4, interactive=True)
                        hint_btn = gr.Button("Hint", scale=1)

    demo.load(
        fn=load_next_problem,
        inputs=[session_state],
        outputs=[problem_display, user_answer, submit_btn, eval_result, official_sol_container, session_state]
    )
    next_btn.click(
        fn=load_next_problem,
        inputs=[session_state],
        outputs=[problem_display, user_answer, submit_btn, eval_result, official_sol_container, session_state]
    )
    submit_btn.click(
        fn=submit_answer_and_evaluate,
        inputs=[user_answer, chatbot, session_state],
        outputs=[submit_btn, chatbot, eval_result, official_solution_text, official_sol_container, session_state]
    )
    chat_input.submit(
        fn=chat_with_tutor,
        inputs=[chat_input, chatbot, session_state],
        outputs=[chat_input, chatbot]
    )
    hint_btn.click(
        fn=request_ai_hint,
        inputs=[chatbot, session_state],
        outputs=[chatbot]
    )

if __name__ == "__main__":
    demo.launch(share=True)
