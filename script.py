import os
import openai
import json
import time

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Simulated case memory (mock vector store)
case_memory = []

# Retrieve similar past case based on keyword match (simulated)
def retrieve_similar_cases(query):
    for case in reversed(case_memory):
        if any(keyword in query.lower() for keyword in case.lower().split()):
            return f"Similar past case found: {case}"
    return "No similar past case found."

# Simulated external validation
def validate_medical_facts(text):
    return f"Validated: The information '{text[:500]}...' is coherent with established medical literature."

# General OpenAI call with advanced features
def call_openai(messages, functions=None, stream=False, function_call=None, json_mode=False):
    kwargs = {
        "model": "gpt-4-1106-preview",
        "messages": messages,
        "temperature": 1.2,
        "stream": stream
    }

    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    if functions:
        kwargs["functions"] = functions
        if function_call:
            kwargs["function_call"] = function_call

    return client.chat.completions.create(**kwargs)

# Agent role prompts
roles = {
    "patient": "You are a patient describing symptoms to a doctor.",
    "doctor": "You are an expert doctor. Diagnose symptoms, optionally suggest tests.",
    "pharmacist": "You are a clinical pharmacist. Recommend suitable medications and check for interactions. Respond ONLY in valid JSON format.",
    "knowledge": "You are a medical literature search assistant.",
    "validator": "You are a final validator. Cross-check diagnosis, treatment, and validation.",
    "feedback": "You are a patient providing simple feedback on whether the advice worked or needs improvement."
}

# Create new dynamic symptom each time
def generate_patient_symptoms():
    response = call_openai([
        {
            "role": "system",
            "content": (
                "You are a simulated patient generator. Create a *new, realistic* symptom description every time, with variety across domains: respiratory, neurological, digestive, skin, cardiac, etc. Keep it concise but vivid."
            )
        },
        {
            "role": "user",
            "content": "Generate a unique patient symptom."
        }
    ])
    return response.choices[0].message.content

# Step 1: PatientAgent
symptom_description = generate_patient_symptoms()
case_memory.append(symptom_description)  # store case for future lookup
patient_message = {
    "role": "user",
    "content": symptom_description
}
print("\n👤 Patient:\n", symptom_description)

# Step 2: DoctorAgent
similar_case = retrieve_similar_cases(symptom_description)
doctor_response = call_openai([
    {"role": "system", "content": roles["doctor"]},
    patient_message,
    {"role": "assistant", "content": f"Memory recall: {similar_case}"}
])
diagnosis = doctor_response.choices[0].message.content
print("\n🩺 Doctor's Diagnosis:\n", diagnosis)

# Step 3: PharmacistAgent (structured JSON output)
pharmacist_prompt = (
    "Diagnosis: " + diagnosis + "\n\n"
    "Provide a recommended medication name, dosage, interaction warnings, and general advice. Respond ONLY in JSON format:\n"
    "{\n  \"medication\": \"...\",\n  \"dosage\": \"...\",\n  \"interactions\": \"...\",\n  \"advice\": \"...\"\n}"
)
pharmacist_response = call_openai([
    {"role": "system", "content": roles["pharmacist"]},
    {"role": "user", "content": pharmacist_prompt}
], json_mode=True)
medication_info = pharmacist_response.choices[0].message.content
print("\n💊 Pharmacist Recommendation:\n", medication_info)

# Step 4: KnowledgeAgent
query_to_validate = f"Diagnosis: {diagnosis}. Drug info: {medication_info}"
knowledge_tool_result = validate_medical_facts(query_to_validate)
print("\n📚 Knowledge Agent Validation:\n", knowledge_tool_result)

# Step 5: ValidatorAgent
validator_messages = [
    {"role": "system", "content": roles["validator"]},
    {"role": "user", "content": f"Patient symptoms: {symptom_description}"},
    {"role": "assistant", "content": f"Doctor response: {diagnosis}"},
    {"role": "assistant", "content": f"Pharmacist suggested: {medication_info}"},
    {"role": "assistant", "content": f"Validated against knowledge base: {knowledge_tool_result}"}
]
final_output = call_openai(validator_messages)
summary = final_output.choices[0].message.content
print("\n✅ Final Validator Summary:\n", summary)

# Step 6: Optional Patient Feedback Simulation
feedback_response = call_openai([
    {"role": "system", "content": roles["feedback"]},
    {"role": "user", "content": f"This was the advice: {summary}\nWas it useful or not? Respond in one line."}
])
feedback = feedback_response.choices[0].message.content
print("\n🗣️ Simulated Patient Feedback:\n", feedback)

# Optional: Log case summary
log = {
    "symptoms": symptom_description,
    "diagnosis": diagnosis,
    "medication": medication_info,
    "validation": knowledge_tool_result,
    "summary": summary,
    "feedback": feedback
}
with open("last_case_log.json", "w", encoding="utf-8") as f:
    json.dump(log, f, indent=2)
print("\n📁 Case saved to 'last_case_log.json'")
