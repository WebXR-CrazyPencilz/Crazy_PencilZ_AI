from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

# ── Blood work report ────────────────────────────────────────
blood_report = """
Patient: Rajesh Sharma, Age: 45

COMPLETE BLOOD COUNT:
Hemoglobin: 11.2 g/dL        (Normal: 13.5-17.5)
Hematocrit: 34%               (Normal: 41-53%)
WBC: 11,500 /uL               (Normal: 4500-11000)

LIPID PANEL:
Total Cholesterol: 245 mg/dL  (Normal: <200)
LDL Cholesterol: 165 mg/dL    (Normal: <100)
HDL Cholesterol: 38 mg/dL     (Normal: >40)
Triglycerides: 210 mg/dL      (Normal: <150)

BLOOD SUGAR:
Fasting Glucose: 118 mg/dL    (Normal: 70-100)
HbA1c: 6.1%                   (Normal: <5.7%)
"""

# ── Step 1: Extract and classify values ─────────────────────
extraction_prompt = f"""
You are a medical data extraction assistant.
Extract all test values from the blood work below.
Classify each one as HIGH, LOW, or NORMAL.
Format response as:
Test Name | Value | Status | Reference Range

Blood Work:
{blood_report}
"""

print("Analyzing blood work...\n")
extraction_response = llm.invoke(extraction_prompt)
extracted = extraction_response.content
print("STEP 1 - EXTRACTED VALUES:")
print(extracted)

# ── Step 2: Generate diet plan ───────────────────────────────
diet_prompt = f"""
You are a nutritionist familiar with Indian dietary habits.
Based on the blood work analysis below, write:
1. A short health summary (3 lines max)
2. A practical Indian diet plan with foods TO EAT and TO AVOID

Blood Work Analysis:
{extracted}
"""

diet_response = llm.invoke(diet_prompt)
print("\nSTEP 2 - HEALTH SUMMARY & DIET PLAN:")
print(diet_response.content)
