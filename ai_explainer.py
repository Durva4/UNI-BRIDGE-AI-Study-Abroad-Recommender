from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(
    provider="auto",
    api_key=HF_TOKEN,
)

def get_country_explanation(country, gpa, ielts, budget_usd,
                            field_of_study, degree_level, ml_confidence):

    prompt = f"""You are a study abroad counselor. A student has been recommended {country}
for studying {field_of_study} at {degree_level} level.

Student Profile:
- GPA: {gpa}
- IELTS: {ielts}
- Budget: ${budget_usd:,}/year
- ML Model Confidence: {ml_confidence:.1f}%

Provide a structured response with EXACTLY these sections:

🎯 WHY THIS COUNTRY
(2-3 reasons why this country suits this student)

🏫 TOP 3 UNIVERSITIES FOR {field_of_study.upper()}
(Name | Ranking | Approx Tuition/year)

💰 COST BREAKDOWN
- Tuition:
- Living Cost/month:
- Total Annual Cost:

📋 REQUIREMENTS
- IELTS minimum:
- GPA expectation:
- Other documents:

🎯 ADMISSION CHANCES
Based on the student profile: Safe / Moderate / Ambitious
(Brief reason)

🎓 SCHOLARSHIPS
(2-3 relevant scholarships)

Keep response concise and practical."""

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
        temperature=0.7,
    )

    return response.choices[0].message.content