**An Intelligent Multi-Agent Healthcare Assistant using OpenAI SDK**

This is an experimental AI-powered medical assistant that simulates a collaborative clinical workflow using large language models (LLMs). The system leverages multiple autonomous agents—Patient, Doctor, Pharmacist, Knowledge Validator, and Feedback Synthesizer—working together to assess symptoms, deliver medical recommendations, and ensure safe, validated healthcare reasoning.

This project was developed as part of a deep exploration into the capabilities of the OpenAI SDK and the orchestration of multi-agent LLM systems for real-world use cases.

---

## Features

-  **Patient Agent**  
  Dynamically generates realistic, domain-varied patient complaints (respiratory, neurological, cardiac, etc.).

-  **Doctor Agent**  
  Offers detailed differential diagnoses and outlines recommended diagnostic procedures.

-  **Pharmacist Agent**  
  Suggests medications in structured JSON format, including dosage, safety warnings, and administration advice.

-  **Knowledge Agent**  
  Validates diagnosis and treatment suggestions against simulated literature-based knowledge.

-  **Validator Agent**  
  Cross-verifies all prior outputs and ensures clinical soundness and logical consistency.

-  **Feedback Agent**  
  Simulates patient response to the recommended treatment plan.

-  **Automatic Logging**  
  Full output stored as a structured JSON file (`last_case_log.json`) after each execution for review or audit.

---

## 🔧 Tech Stack

- **Language Model:** OpenAI GPT-4 Turbo (`gpt-4-1106-preview`)
- **SDK:** OpenAI Python SDK (v1+)
- **Language:** Python 3.10+
- **Storage:** Local JSON logs (future-ready for vector DB integration)

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- OpenAI Python SDK:
  ```bash
  pip install --upgrade openai
  ```

### Set Your API Key
```bash
set OPENAI_API_KEY=sk-...
```

### Run the Script
```bash
python script.py
```

Output Example

Each run generates:
- A unique patient case
- AI-driven diagnosis and drug suggestions
- Validation results
- Final summary and simulated patient feedback


- Future Enhancements

- ✅ Streamlit-based UI for interaction
- ✅ Real-time streaming responses
- ✅ Embedding-based memory (FAISS, ChromaDB)
- ✅ Agent confidence scoring and retry logic
- ✅ Chain-of-thought explanations per agent
