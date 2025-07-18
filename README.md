# AI-Driven Discovery or Improvement of Unconventional Mathematical Theorems and Axioms  

## 📌 Overview  
This project focuses on leveraging **Artificial Intelligence, Neuro-Symbolic AI, Reinforcement Learning, and Theorem Proving** to discover, refine, and enhance unconventional mathematical theorems and axioms, specifically in **Hyperbolic Geometry & Non-Euclidean Mathematics**.  

Our goal is to develop a **powerful AI-driven framework** capable of:  
✅ Extracting, validating, and improving mathematical theorems  
✅ Discovering new postulates and axioms  
✅ Utilizing **graph-based AI**, **theorem provers (Lean, Coq, MetaMath)**, and **evolutionary search**  
✅ Enabling high-impact research in unconventional mathematics  

---

## 📁 Project Structure  


## 🚀 Features  

🔹 **AI-Powered Theorem Discovery** – Generates potential new theorems using AI models  
🔹 **Automated Theorem Extraction** – Collects and structures existing Non-Euclidean theorems in JSON/XML  
🔹 **Symbolic AI & Proof Verification** – Uses **Lean, Coq, MetaMath** for logical validation  
🔹 **Graph Neural Networks for Theorem Prediction** – Analyzes relationships between theorems  
🔹 **Evolutionary Search & Reinforcement Learning** – Optimizes theorem discovery process  

---

## 🔧 Technologies & Tools  

- **Machine Learning & AI**: PyTorch, TensorFlow, Hugging Face Transformers  
- **Graph-Based Methods**: NetworkX, DGL (Deep Graph Library), PyG (PyTorch Geometric)  
- **Symbolic Mathematics**: Lean, Coq, MetaMath, SymPy, LaTeX  
- **Reinforcement Learning**: OpenAI Gym, Proximal Policy Optimization (PPO)  
- **Data Processing**: Pandas, NumPy, XML Parsing  
- **Visualization**: Matplotlib, Plotly, GraphViz  

---

## 📊 Dataset & Theorem Representation  

All collected and AI-generated theorems are stored in **structured formats**:  

### **Example Theorem JSON Format**
```json
{
  "theorem": {
    "name": "Hyperbolic Parallel Postulate",
    "mathematical_domain": "Hyperbolic Geometry",
    "authors": ["Bolyai", "Lobachevsky"],
    "year_of_discovery": "1830",
    "formal_statement": {
      "text": "In hyperbolic geometry, through a point not on a given line, there exist infinitely many lines parallel to the given line.",
      "symbolic_representation": {
        "latex": "L \\parallel P",
        "lean": "theorem hyperbolic_parallel {P L : Line} : Exists (fun Q => Q ∥ L)"
      }
    },
    "proof": {
      "formal_proof": "Using hyperbolic plane models...",
      "computational_validation": {
        "verified": true,
        "tool_used": "Lean"
      }
    }
  }
}
```
### 🔥 How to Use
## 1️⃣ Clone the Repository
```
git clone https://github.com/your-username/AI-driven-theorem-discovery.git
cd AI-driven-theorem-discovery
```
## 2️⃣ Install Dependencies
```
pip install -r requirements.txt
```
## 3️⃣ Run AI Models for Theorem Discovery
```
python models/theorem_discovery.py

```
## 4️⃣ Validate Theorem Proofs using Theorem Provers
```
cd theorem_provers
lean lean_proofs.lean
coqc coq_proofs.v
metamath metamath_proofs.mm
```
## 5️⃣ Train Graph Neural Network for Theorem Predictions
```
python models/graph_neural_network.py
```

# 📄 Research Papers & References
## 📖 Key Papers Referenced:

Bolyai & Lobachevsky’s Hyperbolic Geometry (1830)
AI-Augmented Theorem Proving with Lean & Coq (2024)
Graph-Based Machine Learning for Mathematical Discovery (2025)
📚 Want to contribute? See CONTRIBUTING.md for guidelines!

# ⚖️ License
 This project is licensed under the Apache License – See LICENSE for details.

✨ Contributors
👨‍💻 Team Member 1 – Lead Developer & Researcher
👩‍💻 Team Member 2 – Mathematical Model Expert
👨‍💻 Team Member 3 – Machine Learning & Theorem Proving
👩‍💻 Team Member 4 – Graph AI & Reinforcement Learning

🎯 Future Work
🔹 Refining AI-Based Theorem Generation
🔹 Enhancing Theorem Proving Accuracy
🔹 Integrating with Mathematical Knowledge Graphs
🔹 Publishing Research Papers

💬 Get in Touch!
📧 Contact us at: manjunathakoshinum@gmail.com,
                  akhilnag07@gmail.com,
                  varshirdx@gmail.com,
                  tsaireddy11@gmail.com
                  
🔗 Follow us on LinkedIn https://www.linkedin.com/in/manjunathah/

🚀 Join us in revolutionizing theorem discovery with AI!

