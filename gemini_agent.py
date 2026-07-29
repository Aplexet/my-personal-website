import os
# Force OpenBLAS to limit its active thread allocation so it doesn't run out of memory hooks
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import subprocess
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load the secret API key from your .env file
load_dotenv()

def run_automated_gemini_agent():
    print("🤖 [Step 1] Initializing Gemini Agent and inspecting dataset metadata...")
    
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ Error: Could not find GEMINI_API_KEY inside your .env file!")
        return

    # Load the dataset to extract metadata
    dataset = load_dataset("scikit-learn/breast-cancer-wisconsin")
    df = pd.DataFrame(dataset['train'])
    
    dataset_schema = {
        "columns": df.columns.tolist(),
        "missing_values_detected": df.isnull().sum().to_dict(),
        "total_rows": len(df)
    }

    client = genai.Client()
    
    agent_instructions = (
        "You are an autonomous machine learning engineering agent. Your objective is to write "
        "a clean, standalone Python script that processes a dataset and trains a predictive model."
    )
    
    user_prompt = f"""
    Please write a complete Python script to process a dataset and train a classification model.
    
    DATASET INFO:
    - Name: scikit-learn/breast-cancer-wisconsin (Hugging Face)
    - Total Samples: {dataset_schema['total_rows']}
    - Columns and null counts: {dataset_schema['missing_values_detected']}
    
    YOUR GENERATION REQUIREMENTS:
    0. Make sure the very first lines of your script import os and set os.environ["OPENBLAS_NUM_THREADS"] = "1" to avoid Windows thread crashes.
    1. Drop any completely empty structural columns like 'Unnamed: 32' using `dropna(axis=1, how='all')`.
    2. Set features 'X' by dropping the 'diagnosis' column, keeping only numeric fields.
    3. Set target 'y' to the 'diagnosis' column.
    4. Deal with any remaining missing data safely using `X = X.fillna(X.mean())`.
    5. Split data 80/20, scale using StandardScaler, and train a LogisticRegression model.
    6. Print out final model accuracy score and classification_report.
    
    CRITICAL: Output ONLY raw, executable Python code. Do not wrap it in markdown block quotes like ```python. Just output the clean text lines.
    """

    print("🚀 [Step 2] Gemini Agent is autonomously designing your ML pipeline...")
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=agent_instructions,
            temperature=0.1,
        ),
    )
    
    generated_code = response.text.strip()
    
    # Clean up markdown code blocks if the agent included them anyway
    if generated_code.startswith("```python"):
        generated_code = generated_code.split("```python", 1)[1]
    elif generated_code.startswith("```"):
        generated_code = generated_code.split("```", 1)[1]
        
    if generated_code.endswith("```"):
        generated_code = generated_code.rsplit("```", 1)[0]
        
    generated_code = generated_code.strip()

    # Save the cleaned agent code
    output_filename = "agent_generated_pipeline.py"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(generated_code)
    print(f"💾 [Step 3] Saved cleaned agent script to local file: {output_filename}")

    # Run the cleaned file!
    print("⚙️  [Step 4] Executing the agent's code in a safe background runner...")
    try:
        result = subprocess.run([sys.executable, output_filename], capture_output=True, text=True, check=True)
        print("\n🏆 [SUCCESS] The Gemini Agent's code executed perfectly! Output:")
        print("-" * 60)
        print(result.stdout)
        print("-" * 60)
    except subprocess.CalledProcessError as e:
        print("\n❌ Error running the agent's code:")
        print(e.stderr)

if __name__ == "__main__":
    run_automated_gemini_agent()
