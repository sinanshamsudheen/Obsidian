"""

LLM Model Comparison Tool for Call Analysis

------------------------------------------

This script runs the same call transcript through multiple LLM models

and provides a side-by-side comparison of their analysis results.

  

Supported providers:

- OpenAI (default: gpt-4o)

- Anthropic (claude-3-sonnet)

- Mistral (mistral-medium-latest)

- Groq (llama3-8b)

- Ollama (local models - commented out by default)

  

Output is presented in a comparison table format for easy evaluation.

"""

  

import asyncio

import time

import json

import sys

import os

from datetime import datetime

from typing import Dict, Any, List

  

# Fix import paths

current_dir = os.path.dirname(os.path.abspath(__file__))

server_dir = os.path.abspath(os.path.join(current_dir, "../.."))

if server_dir not in sys.path:

sys.path.insert(0, server_dir)

  

try:

from tabulate import tabulate

except ImportError:

print("Error: The 'tabulate' package is required. Install it with: pip install tabulate")

sys.exit(1)

  

try:

# Try the relative import first (when run as a module)

from app.services.anyLLM_service import AnyLLMService

except ImportError:

# If that fails, try the local import (when run directly)

try:

from anyLLM_service import AnyLLMService

except ImportError:

print("Error: Cannot import AnyLLMService. Make sure you have created anyLLM_service.py")

print("Run this script from the server directory with: python -m app.services.anyLLM_service_example")

sys.exit(1)

  

# Define test cases - you can add more complex transcripts here

TEST_CASES = [

{

"name": "Basic Service Follow-up",

"transcript_messages": [

{"role": "Customer", "message": "Hi, I'm calling about my car service yesterday."},

{"role": "Agent", "message": "Hello, how can I help you with your recent service?"},

{"role": "Customer", "message": "The service was great, but I noticed my car is making a strange noise now."},

{"role": "Agent", "message": "I'm sorry to hear that. Can you describe the noise?"},

{"role": "Customer", "message": "It's like a rattling sound when I drive over bumps."},

{"role": "Agent", "message": "I understand. Would you like to schedule a follow-up appointment to check this?"},

{"role": "Customer", "message": "Yes, that would be helpful. Your team was very professional last time."}

],

"service_record_data": {

"customer_name": "John Doe",

"service_type": "Regular Maintenance",

"vehicle_info": "2018 Toyota Camry",

"service_advisor_name": "Sarah Johnson"

},

"organization_data": {

"name": "Acme Auto Service",

"description": "Premium automotive care and maintenance",

"service_center_description": "Full-service auto repair center with certified technicians",

"location": "Portland, OR"

},

"tags": ["communication", "professionalism", "service quality", "follow-up"]

}

# You can add more test cases here

]

  

# Define models to test

MODELS_TO_TEST = [

{"name": "OpenAI (Default)", "model": None}, # Uses default from settings

{"name": "OpenAI (GPT-4o)", "model": "openai/gpt-4o"},

{"name": "OpenAI (GPT-4o Mini)", "model": "openai/gpt-4o-mini"},

{"name": "OpenAI (GPT-4 Turbo)", "model": "openai/gpt-4-turbo"},

{"name": "OpenAI (GPT-3.5 Turbo)", "model": "openai/gpt-3.5-turbo"},

  

# {"name": "Anthropic Claude", "model": "anthropic/claude-3-sonnet"},

# {"name": "Mistral", "model": "mistral/mistral-medium-latest"},

{"name": "Groq (Llama3)", "model": "groq/llama-3.1-8b-instant"},

{"name": "Groq (deepseek r1)", "model": "groq/deepseek-r1-distill-llama-70b"},

{"name": "Groq (kimi-k2)", "model": "groq/moonshotai/kimi-k2-instruct"},

{"name": "Groq (Qwen3)", "model": "groq/qwen/qwen3-32b"},

# Uncomment to test Ollama

# {"name": "Ollama (Local)", "model": "ollama/llama3"},

]

  

# Define metrics to compare

METRICS = [

"nps_score",

"customer_satisfaction",

"call_summary",

"execution_time" # We'll add this ourselves

]

  

class ModelComparison:

def __init__(self):

self.service = AnyLLMService()

self.results = {}

async def run_model(self, model_info, test_case):

model_name = model_info["name"]

model = model_info["model"]

print(f"\nRunning analysis with {model_name}...")

start_time = time.time()

try:

result = await self.service.analyze_call_transcript(

test_case["transcript_messages"],

test_case["service_record_data"],

test_case["organization_data"],

test_case["tags"],

model=model

)

# Add execution time

execution_time = time.time() - start_time

result["execution_time"] = f"{execution_time:.2f}s"

print(f"✅ {model_name} completed in {execution_time:.2f}s")

return result

except Exception as e:

print(f"❌ Error with {model_name}: {str(e)}")

return {

"error": str(e),

"execution_time": f"{time.time() - start_time:.2f}s"

}

async def run_comparison(self):

for i, test_case in enumerate(TEST_CASES):

test_name = test_case["name"]

print(f"\n{'='*80}\nRunning Test Case {i+1}: {test_name}\n{'='*80}")

results = {}

for model_info in MODELS_TO_TEST:

model_name = model_info["name"]

result = await self.run_model(model_info, test_case)

results[model_name] = result

self.results[test_name] = results

return self.results

def generate_comparison_table(self):

tables = []

for test_name, models_results in self.results.items():

print(f"\n\n{'='*80}\nResults for: {test_name}\n{'='*80}")

# Generate table for numeric/simple metrics

headers = ["Metric"] + list(models_results.keys())

rows = []

# Add simple metrics first

for metric in METRICS:

row = [metric.replace("_", " ").title()]

for model_name in models_results.keys():

model_result = models_results[model_name]

if "error" in model_result and metric != "execution_time":

row.append("ERROR")

else:

value = model_result.get(metric, "N/A")

# Format certain values

if metric == "nps_score" and value is not None:

row.append(f"{value}/10")

elif metric == "customer_satisfaction" and value is not None:

row.append(f"{value}/5")

elif metric == "call_summary" and value is not None:

# Truncate long summaries

if len(str(value)) > 50:

row.append(f"{str(value)[:50]}...")

else:

row.append(str(value))

else:

row.append(str(value))

rows.append(row)

# Generate the table

table = tabulate(rows, headers, tablefmt="grid")

print(table)

tables.append(table)

# Show detailed comparison for complex fields

print("\n\nDetailed Comparison:\n")

for model_name, result in models_results.items():

print(f"\n--- {model_name} ---")

if "error" in result:

print(f"ERROR: {result['error']}")

continue

# Show call summary in full

if "call_summary" in result:

print(f"Call Summary: {result['call_summary']}")

# Show other complex fields

for field in ["key_points", "action_items", "customer_sentiment"]:

if field in result:

print(f"\n{field.replace('_', ' ').title()}: ")

if isinstance(result[field], list):

for item in result[field]:

print(f"- {item}")

else:

print(result[field])

# Save results to file

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

filename = f"model_comparison_{timestamp}.json"

with open(filename, "w") as f:

json.dump(self.results, f, indent=2)

print(f"\nDetailed results saved to {filename}")

return tables

  

async def main():

try:

# Initialize the comparison service

try:

comparison = ModelComparison()

except Exception as e:

print(f"\n⚠️ Error initializing the comparison tool: {str(e)}")

print("\nPossible issues:")

print("1. The anyLLM_service.py file might not be in the correct location")

print("2. Configuration issues with settings.py or environment variables")

print("\nMake sure you have:")

print("- Created anyLLM_service.py in the app/services directory")

print("- Set up your .env file with API keys")

print("- Installed required dependencies (pip install any-llm-sdk tabulate)")

return

# Run the comparison

await comparison.run_comparison()

comparison.generate_comparison_table()

print("\nUsage Tips:")

print("1. To focus on specific models, edit the MODELS_TO_TEST list")

print("2. To add more test cases, extend the TEST_CASES list")

print("3. To compare different metrics, modify the METRICS list")

print("4. Run with: python -m app.services.anyLLM_service_example")

print("5. For detailed results, check the JSON file that was generated")

except KeyboardInterrupt:

print("\nTest interrupted by user.")

except Exception as e:

print(f"\nError running comparison: {str(e)}")

import traceback

traceback.print_exc()

  

if __name__ == "__main__":

print("\n" + "="*80)

print(" LLM MODEL COMPARISON TOOL ".center(80, "="))

print("="*80 + "\n")

# Check if we're in the correct directory structure

is_correct_structure = os.path.exists(os.path.join(current_dir, "anyLLM_service.py"))

if not is_correct_structure:

print("⚠️ Warning: This script may not be running in the correct directory.")

print(" The anyLLM_service.py file was not found in the same directory.")

print("\nRecommended way to run this script:")

print("1. Navigate to the server directory:")

print(" cd /home/zero/Lokam/lokamspace/server")

print("2. Run as a module:")

print(" python -m app.services.anyLLM_service_example")

print("\nAttempting to continue anyway...\n")

print("Starting model comparison test...")

print("This will analyze the same call transcript with multiple LLM models")

print("and generate a side-by-side comparison of their performance.")

print("\nModels being tested:")

for model in MODELS_TO_TEST:

print(f"- {model['name']} ({model['model'] if model['model'] else 'default'})")

print("\nPress Ctrl+C to abort the test at any time.")

print("="*80 + "\n")

asyncio.run(main())