#!/usr/bin/env python3

"""

Script to test and evaluate OpenAI prompt accuracy using sample data.

This script allows you to test both the current prompt and the old prompt

with sample data and compare the results.

"""

  

import os

import sys

import json

import time

import asyncio

import statistics

from typing import Dict, Any, List, Tuple

  

# Add the parent directory to the path so we can import app modules

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

  

from app.services.openai_service import OpenAIService

from app.core.config import settings

  

# Sample test data

SAMPLE_TRANSCRIPTS = [

# Sample 1: Clear positive feedback with explicit NPS

[

{"role": "human", "message": "Hi there."},

{"role": "assistant", "message": "Hello! How was your service experience?"},

{"role": "human", "message": "It was great. I'd give it a 9 out of 10."},

{"role": "assistant", "message": "That's wonderful! Anything in particular you liked?"},

{"role": "human", "message": "The mechanic explained everything clearly, and the service was fast."}

],

# Sample 2: Mixed feedback without explicit NPS

[

{"role": "human", "message": "Hello."},

{"role": "assistant", "message": "Hi there! How was your experience with us?"},

{"role": "human", "message": "The technician was professional but I waited too long."},

{"role": "assistant", "message": "I apologize for the wait time. Was there anything else about your visit?"},

{"role": "human", "message": "No, that's all. Just work on your scheduling."}

],

# Sample 3: Negative feedback with implied low NPS

[

{"role": "human", "message": "Hi."},

{"role": "assistant", "message": "Hello! How was your recent service with us?"},

{"role": "human", "message": "Terrible. Nobody called me back and I had to follow up three times."},

{"role": "assistant", "message": "I'm so sorry to hear that. Can you tell me more?"},

{"role": "human", "message": "I wouldn't recommend your shop to anyone. Very disappointed."}

],

# Sample 4: Neutral feedback

[

{"role": "human", "message": "Hello there."},

{"role": "assistant", "message": "Hi! How was your experience?"},

{"role": "human", "message": "It was okay. Nothing special."},

{"role": "assistant", "message": "Thanks for the feedback. Anything we could improve?"},

{"role": "human", "message": "Not really. Just a standard service."}

],

# Sample 5: Positive feedback with implied high NPS

[

{"role": "human", "message": "Hey."},

{"role": "assistant", "message": "Hello! How was your visit with us?"},

{"role": "human", "message": "Absolutely top-notch service! Best experience I've had at any garage."},

{"role": "assistant", "message": "That's fantastic! What did you like most?"},

{"role": "human", "message": "The staff was super friendly and responsive to all my questions."}

],

# Sample 6: Contradictory feedback (mentions both positive and negative aspects for same tags)

[

{"role": "human", "message": "Hello there."},

{"role": "assistant", "message": "Hi! How was your recent experience at our service center?"},

{"role": "human", "message": "It was a bit of a mixed bag. The communication was excellent at the start but then terrible later on."},

{"role": "assistant", "message": "I'm sorry to hear that. Could you tell me more?"},

{"role": "human", "message": "Well, the technician explained everything very well when I dropped off my car, but then nobody called me when it was ready. I had to call three times to get an update."}

],

# Sample 7: Indirect NPS with sarcasm

[

{"role": "human", "message": "Hi."},

{"role": "assistant", "message": "Hello! How would you rate your recent service?"},

{"role": "human", "message": "Oh yeah, it was just AMAZING waiting 3 hours past my appointment time."},

{"role": "assistant", "message": "I'm very sorry about the wait time. Was there anything else about your experience?"},

{"role": "human", "message": "The shop was clean, I'll give you that. But if this is how you value your customers' time, I won't be back."}

],

# Sample 8: Feedback focusing on value but with indirect mentions of other tags

[

{"role": "human", "message": "Hello."},

{"role": "assistant", "message": "Hi there! How was your visit with us?"},

{"role": "human", "message": "I'm not sure the service was worth what I paid, to be honest."},

{"role": "assistant", "message": "I'm sorry to hear that. Could you share what made you feel that way?"},

{"role": "human", "message": "Everyone was nice, and the place looked great, but $600 for what was supposed to be a minor repair seems excessive. The mechanic did explain the costs, but I still feel like I overpaid."}

],

# Sample 9: Technical language with implicit feedback

[

{"role": "human", "message": "Good afternoon."},

{"role": "assistant", "message": "Hello! How was your service experience with us?"},

{"role": "human", "message": "The diagnostic procedures were adequately performed, and the subsequent repair exhibited functional efficacy."},

{"role": "assistant", "message": "That's good to hear. Is there anything specific about your visit you'd like to mention?"},

{"role": "human", "message": "The establishment maintained appropriate cleanliness standards. However, the temporal efficiency could be optimized to reduce customer wait duration."}

],

# Sample 10: Multiple complaints with no clear NPS but strongly negative tone

[

{"role": "human", "message": "Hey."},

{"role": "assistant", "message": "Hello! How was your visit to our service center?"},

{"role": "human", "message": "Let me list the issues: First, nobody greeted me for 15 minutes. Then, the waiting area was dirty. Finally, I was quoted one price but charged another."},

{"role": "assistant", "message": "I'm very sorry to hear about these problems. Anything else you'd like to share?"},

{"role": "human", "message": "Yes, even after all that, nobody even apologized or acknowledged these issues. Just terrible customer service all around."}

]

]

  

# Sample service record data

SAMPLE_SERVICE_RECORD = {

"customer_name": "John Doe",

"service_type": "Oil Change and Tire Rotation",

"vehicle_info": "2019 Toyota Camry",

"service_advisor_name": "Jane Smith"

}

  

# Sample organization data

SAMPLE_ORGANIZATION = {

"name": "AutoCare Center",

"description": "Full-service auto repair and maintenance center",

"service_center_description": "Main downtown location with 10 service bays",

"location": "123 Main St, Anytown USA"

}

  

# Sample tags

SAMPLE_TAGS = ["timeliness", "communication", "professionalism", "value", "cleanliness"]

  

# Expected results for evaluation (what we expect the model to return)

EXPECTED_RESULTS = [

{

"call_summary": "Customer had a great experience and provided positive feedback.",

"nps_score": 9,

"positive_mentions": ["communication", "professionalism"],

"detractors": []

},

{

"call_summary": "Customer had mixed feelings, appreciating professionalism but concerned about wait time.",

"nps_score": None,

"positive_mentions": ["professionalism"],

"detractors": ["timeliness"]

},

{

"call_summary": "Customer had a very negative experience with communication issues.",

"nps_score": 2,

"positive_mentions": [],

"detractors": ["communication"]

},

{

"call_summary": "Customer had a neutral experience with no strong opinions.",

"nps_score": 5,

"positive_mentions": [],

"detractors": []

},

{

"call_summary": "Customer had an exceptional experience with excellent service.",

"nps_score": 10,

"positive_mentions": ["communication", "professionalism"],

"detractors": []

},

{

"call_summary": "Customer experienced mixed communication quality during their service.",

"nps_score": 6,

"positive_mentions": ["communication"],

"detractors": ["communication"]

},

{

"call_summary": "Customer was extremely dissatisfied with wait time despite clean facilities.",

"nps_score": 3,

"positive_mentions": ["cleanliness"],

"detractors": ["timeliness"]

},

{

"call_summary": "Customer felt the service was overpriced despite good staff and facilities.",

"nps_score": 4,

"positive_mentions": ["professionalism", "cleanliness"],

"detractors": ["value"]

},

{

"call_summary": "Customer found service technically adequate but noted issues with wait time.",

"nps_score": 6,

"positive_mentions": ["professionalism", "cleanliness"],

"detractors": ["timeliness"]

},

{

"call_summary": "Customer had multiple serious complaints about their service experience.",

"nps_score": 1,

"positive_mentions": [],

"detractors": ["timeliness", "cleanliness", "value", "professionalism", "communication"]

}

]

  

# Helper function to use a custom prompt (the old, commented-out prompt)

def create_old_prompt(

transcript_messages: List[Dict[str, Any]],

service_record_data: Dict[str, Any],

organization_data: Dict[str, Any],

tags: List[str]

) -> str:

formatted_transcript = "\n".join(

f"{m['role']}: {m['message']}" for m in transcript_messages

)

formatted_tags = ", ".join(tags)

  

service_record_json = json.dumps({

"customer_name": service_record_data.get('customer_name', 'N/A'),

"service_type": service_record_data.get('service_type', 'N/A'),

"vehicle_info": service_record_data.get('vehicle_info', 'N/A'),

"service_advisor_name": service_record_data.get('service_advisor_name', 'N/A')

}, indent=2)

  

organization_json = json.dumps({

"company": organization_data.get('name', 'N/A'),

"description": organization_data.get('description', 'N/A'),

"service_center": organization_data.get('service_center_description', 'N/A'),

"focus_tags": formatted_tags,

"location": organization_data.get('location', 'N/A')

}, indent=2)

  

output_schema = """

\"\"\"json

{

"call_summary": "<string>",

"nps_score": <integer|null>,

"overall_feedback": "<string>",

"positive_mentions": ["<tag1>", "..."],

"detractors": ["<tag2>", "..."]

}

\"\"\""""

  

return f"""

## --- ROLE ---

You are an expert conversation analyzer. Always respond with valid JSON matching exactly the schema given—no extra text or metadata.

## --- INPUT — TRANSCRIPT & CONTEXT —

Call Transcript:

{formatted_transcript}

Service Record Info:

```json

{service_record_json}

```

Organization Info:

```json

{organization_json}

```

Focus Tags: {formatted_tags}

## --- TASK INSTRUCTIONS — STEP-BY-STEP —

Follow this reasoning internally and make your final JSON concise and strict:

**Step 1 - Summarize call outcome in one sentence.**

**Step 2 - Locate an NPS score (0-10):**

- If a number in that range is uttered explicitly (“NPS: 8”, “I'd give you a 9”), record that.

- If phrased indirectly (“That's top-tier service”) interpret to the most appropriate numeric equivalent (e.g. 9-10), but only if clearly implied.

- If unclear or not stated, set to `null`.

**Step 3 - Overall feedback:** Write 1-2 sentences reflecting customer sentiment.

**Step 4 - For each focus tag:**

- If mentioned positively (praise, enjoyment, satisfaction), include in `positive_mentions`.

- If mentioned negatively (issue, complaint, dissatisfaction), include in `detractors`.

- If not clearly mentioned, exclude.

### Few-Shot Examples:

--Example A--

Transcript:

Customer: “I'd say my experience was a 9 — amazing service.”

Tags: “timeliness, cleanliness, communication”

Output:

{{

"call_summary": "Customer gave exceptionally positive feedback.",

"nps_score": 9,

"overall_feedback": "Customer rated the experience very highly and expressed satisfaction across all areas.",

"positive_mentions": ["timeliness","communication"],

"detractors": []

}}

--Example B--

Transcript:

Customer: “Honestly it was okay, nothing stood out, but the pickup was late.”

Tags: “timeliness, professionalism, value”

Output:

{{

"call_summary": "Customer had mixed feelings, noting a delay in pickup.",

"nps_score": null,

"overall_feedback": "Customer felt neutral overall but was disappointed by the pickup delay.",

"positive_mentions": [],

"detractors": ["timeliness"]

}}

## --- OUTPUT SCHEMA ---

{output_schema}

Respond with only the valid JSON object.

"""

  
  

class PromptTester:

def __init__(self):

self.service = OpenAIService()

# Create a patched service with the old prompt

self.old_prompt_service = OpenAIService()

# Monkey patch the _build_analysis_prompt method

self.old_prompt_service._build_analysis_prompt = create_old_prompt

self.results = {

"new_prompt": [],

"old_prompt": []

}

self.timing = {

"new_prompt": [],

"old_prompt": []

}

self.token_usage = {

"new_prompt": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},

"old_prompt": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

}

# Define which transcripts are considered "challenging" (the additional ones)

self.challenging_transcripts = [5, 6, 7, 8, 9] # 0-indexed

def count_challenging_cases(self, prompt_type):

"""Count how many of the challenging cases the prompt handled correctly"""

correct_count = 0

for idx in self.challenging_transcripts:

result = self.results[prompt_type][idx]

expected = EXPECTED_RESULTS[idx]

# Check if NPS is correct or within 1 point

nps_correct = False

if result["nps_score"] == expected["nps_score"]:

nps_correct = True

elif result["nps_score"] is not None and expected["nps_score"] is not None:

if abs(result["nps_score"] - expected["nps_score"]) <= 1:

nps_correct = True

# Check if key tags are correct (at least 50% match)

pos_correct = False

if expected["positive_mentions"]:

match_count = len(set(result["positive_mentions"]) & set(expected["positive_mentions"]))

if match_count / len(expected["positive_mentions"]) >= 0.5:

pos_correct = True

else:

# If no positive mentions expected, it's correct if none or very few were found

if len(result["positive_mentions"]) <= 1:

pos_correct = True

neg_correct = False

if expected["detractors"]:

match_count = len(set(result["detractors"]) & set(expected["detractors"]))

if match_count / len(expected["detractors"]) >= 0.5:

neg_correct = True

else:

# If no detractors expected, it's correct if none or very few were found

if len(result["detractors"]) <= 1:

neg_correct = True

# Consider the case handled correctly if NPS and at least one tag type is correct

if nps_correct and (pos_correct or neg_correct):

correct_count += 1

return correct_count

def print_ascii_chart(self, new_metrics, old_metrics):

"""Print an ASCII chart comparing key metrics"""

# ANSI color codes

GREEN = "\033[92m"

RED = "\033[91m"

BOLD = "\033[1m"

ENDC = "\033[0m"

# Select key metrics to visualize

key_metrics = [

{"name": "Overall Score", "key": "overall_score"},

{"name": "NPS Accuracy", "key": "nps_exact_match"},

{"name": "Summary Match", "key": "summary_match"},

{"name": "Positive Tags F1", "key": "positive_mentions_f1"},

{"name": "Detractor Tags F1", "key": "detractors_f1"},

]

max_width = 40 # Maximum width of the bar

for metric in key_metrics:

new_val = new_metrics[metric["key"]]

old_val = old_metrics[metric["key"]]

# Calculate bar lengths

new_bar_len = int((new_val / 100) * max_width)

old_bar_len = int((old_val / 100) * max_width)

# Format the bars with colors

if new_val > old_val:

new_bar = f"{GREEN}{'█' * new_bar_len}{ENDC}"

old_bar = f"{'█' * old_bar_len}"

elif old_val > new_val:

new_bar = f"{'█' * new_bar_len}"

old_bar = f"{GREEN}{'█' * old_bar_len}{ENDC}"

else:

new_bar = f"{'█' * new_bar_len}"

old_bar = f"{'█' * old_bar_len}"

# Print the chart

print(f"\n{BOLD}{metric['name']}{ENDC} ({metric['key']})")

print(f"New: {new_bar} {new_val:.1f}%")

print(f"Old: {old_bar} {old_val:.1f}%")

  

async def test_prompt(self, prompt_type: str, transcript_idx: int) -> Dict[str, Any]:

"""Test a specific prompt type with a specific transcript"""

transcript = SAMPLE_TRANSCRIPTS[transcript_idx]

service = self.service if prompt_type == "new_prompt" else self.old_prompt_service

start_time = time.time()

try:

result = await service.analyze_call_transcript(

transcript, SAMPLE_SERVICE_RECORD, SAMPLE_ORGANIZATION, SAMPLE_TAGS

)

end_time = time.time()

self.timing[prompt_type].append(end_time - start_time)

return result

except Exception as e:

print(f"Error with {prompt_type} on transcript {transcript_idx}: {e}")

return {

"call_summary": f"Error: {str(e)}",

"nps_score": None,

"overall_feedback": "",

"positive_mentions": [],

"detractors": []

}

  

def calculate_accuracy(self, results: List[Dict[str, Any]]) -> Dict[str, float]:

"""Calculate accuracy metrics compared to expected results"""

metrics = {

"summary_match": 0,

"nps_exact_match": 0,

"nps_within_1": 0,

"positive_mentions_precision": [],

"positive_mentions_recall": [],

"positive_mentions_f1": [],

"detractors_precision": [],

"detractors_recall": [],

"detractors_f1": [],

"overall_score": 0

}

for i, result in enumerate(results):

expected = EXPECTED_RESULTS[i]

# Summary accuracy (simple keyword match)

if any(keyword in result["call_summary"].lower() for keyword in expected["call_summary"].lower().split()):

metrics["summary_match"] += 1

# NPS score accuracy

if result["nps_score"] == expected["nps_score"]:

metrics["nps_exact_match"] += 1

if result["nps_score"] is not None and expected["nps_score"] is not None:

if abs(result["nps_score"] - expected["nps_score"]) <= 1:

metrics["nps_within_1"] += 1

elif result["nps_score"] is None and expected["nps_score"] is None:

metrics["nps_within_1"] += 1

# Tag accuracy (precision and recall)

# Positive mentions precision

pos_precision = 0.0

pos_recall = 0.0

pos_f1 = 0.0

if result["positive_mentions"]:

pos_precision = len(set(result["positive_mentions"]) & set(expected["positive_mentions"])) / len(result["positive_mentions"])

metrics["positive_mentions_precision"].append(pos_precision)

elif not expected["positive_mentions"]:

pos_precision = 1.0 # Correctly identified none

metrics["positive_mentions_precision"].append(pos_precision)

# Positive mentions recall

if expected["positive_mentions"]:

pos_recall = len(set(result["positive_mentions"]) & set(expected["positive_mentions"])) / len(expected["positive_mentions"])

metrics["positive_mentions_recall"].append(pos_recall)

elif not result["positive_mentions"]:

pos_recall = 1.0 # Correctly identified none

metrics["positive_mentions_recall"].append(pos_recall)

# Calculate F1 score for positive mentions

if pos_precision + pos_recall > 0:

pos_f1 = 2 * pos_precision * pos_recall / (pos_precision + pos_recall)

metrics["positive_mentions_f1"].append(pos_f1)

else:

metrics["positive_mentions_f1"].append(0.0)

# Detractors precision

det_precision = 0.0

det_recall = 0.0

det_f1 = 0.0

if result["detractors"]:

det_precision = len(set(result["detractors"]) & set(expected["detractors"])) / len(result["detractors"])

metrics["detractors_precision"].append(det_precision)

elif not expected["detractors"]:

det_precision = 1.0 # Correctly identified none

metrics["detractors_precision"].append(det_precision)

# Detractors recall

if expected["detractors"]:

det_recall = len(set(result["detractors"]) & set(expected["detractors"])) / len(expected["detractors"])

metrics["detractors_recall"].append(det_recall)

elif not result["detractors"]:

det_recall = 1.0 # Correctly identified none

metrics["detractors_recall"].append(det_recall)

# Calculate F1 score for detractors

if det_precision + det_recall > 0:

det_f1 = 2 * det_precision * det_recall / (det_precision + det_recall)

metrics["detractors_f1"].append(det_f1)

else:

metrics["detractors_f1"].append(0.0)

# Calculate averages for precision, recall and F1

for key in ["positive_mentions_precision", "positive_mentions_recall", "positive_mentions_f1",

"detractors_precision", "detractors_recall", "detractors_f1"]:

if metrics[key]:

metrics[key] = statistics.mean(metrics[key])

else:

metrics[key] = 0.0

# Calculate normalized metrics (as percentages)

metrics["summary_match"] = (metrics["summary_match"] / len(results)) * 100

metrics["nps_exact_match"] = (metrics["nps_exact_match"] / len(results)) * 100

metrics["nps_within_1"] = (metrics["nps_within_1"] / len(results)) * 100

metrics["positive_mentions_precision"] *= 100

metrics["positive_mentions_recall"] *= 100

metrics["positive_mentions_f1"] *= 100

metrics["detractors_precision"] *= 100

metrics["detractors_recall"] *= 100

metrics["detractors_f1"] *= 100

# Calculate overall score (weighted average of all metrics)

metrics["overall_score"] = (

metrics["summary_match"] * 0.15 +

metrics["nps_exact_match"] * 0.15 +

metrics["nps_within_1"] * 0.1 +

metrics["positive_mentions_precision"] * 0.1 +

metrics["positive_mentions_recall"] * 0.1 +

metrics["positive_mentions_f1"] * 0.1 +

metrics["detractors_precision"] * 0.1 +

metrics["detractors_recall"] * 0.1 +

metrics["detractors_f1"] * 0.1

)

return metrics

  

async def run_tests(self):

"""Run all tests for both prompt types"""

print("Starting prompt accuracy tests...")

# Test new prompt

print("\nTesting NEW prompt...")

for i in range(len(SAMPLE_TRANSCRIPTS)):

print(f" Processing transcript {i+1}/{len(SAMPLE_TRANSCRIPTS)}...")

result = await self.test_prompt("new_prompt", i)

self.results["new_prompt"].append(result)

# Test old prompt

print("\nTesting OLD prompt...")

for i in range(len(SAMPLE_TRANSCRIPTS)):

print(f" Processing transcript {i+1}/{len(SAMPLE_TRANSCRIPTS)}...")

result = await self.test_prompt("old_prompt", i)

self.results["old_prompt"].append(result)

# Calculate and display results

new_prompt_metrics = self.calculate_accuracy(self.results["new_prompt"])

old_prompt_metrics = self.calculate_accuracy(self.results["old_prompt"])

# Calculate timing statistics

new_prompt_avg_time = statistics.mean(self.timing["new_prompt"]) if self.timing["new_prompt"] else 0

old_prompt_avg_time = statistics.mean(self.timing["old_prompt"]) if self.timing["old_prompt"] else 0

# ANSI color codes for terminal output

GREEN = "\033[92m"

RED = "\033[91m"

BOLD = "\033[1m"

ENDC = "\033[0m"

# Print results with improved formatting

print("\n" + "="*80)

print(f"{BOLD}PROMPT TESTING RESULTS - COMPARISON ANALYSIS{ENDC}")

print("="*80)

# Create a function to format the difference with color

def format_diff(diff, is_time=False):

if is_time:

# For time metrics, lower is better

if diff < -0.01:

return f"{GREEN}{diff:+.2f}s{ENDC}"

elif diff > 0.01:

return f"{RED}{diff:+.2f}s{ENDC}"

else:

return f"{diff:+.2f}s"

else:

# For accuracy metrics, higher is better

if diff > 0.01:

return f"{GREEN}+{diff:.2f}%{ENDC}"

elif diff < -0.01:

return f"{RED}{diff:.2f}%{ENDC}"

else:

return f"{diff:+.2f}%"

print("\n📊 Accuracy Metrics:")

print(f"{'Metric':<30} │ {'New Prompt':>12} │ {'Old Prompt':>12} │ {'Difference':>15}")

print("─"*80)

# Group metrics for better readability

metric_groups = [

{"title": "Summary & NPS Metrics", "metrics": [

"summary_match", "nps_exact_match", "nps_within_1"

]},

{"title": "Positive Mentions Metrics", "metrics": [

"positive_mentions_precision", "positive_mentions_recall", "positive_mentions_f1"

]},

{"title": "Detractors Metrics", "metrics": [

"detractors_precision", "detractors_recall", "detractors_f1"

]},

{"title": "Overall Performance", "metrics": ["overall_score"]}

]

for group in metric_groups:

print(f"\n{BOLD}{group['title']}{ENDC}")

for metric in group["metrics"]:

new_val = new_prompt_metrics[metric]

old_val = old_prompt_metrics[metric]

diff = new_val - old_val

diff_str = format_diff(diff)

# Format the metric name to be more readable

metric_name = metric.replace("_", " ").title()

print(f"{metric_name:<30} │ {new_val:>11.2f}% │ {old_val:>11.2f}% │ {diff_str:>15}")

print("\n⏱️ Performance Metrics:")

print(f"{'Metric':<30} │ {'New Prompt':>12} │ {'Old Prompt':>12} │ {'Difference':>15}")

print("─"*80)

time_diff = new_prompt_avg_time - old_prompt_avg_time

time_diff_str = format_diff(time_diff, is_time=True)

print(f"{'Average Response Time':<30} │ {new_prompt_avg_time:>11.2f}s │ {old_prompt_avg_time:>11.2f}s │ {time_diff_str:>15}")

# Add a summary section

print("\n" + "="*80)

print(f"{BOLD}SUMMARY OF FINDINGS{ENDC}")

print("="*80)

# Compare overall scores

overall_diff = new_prompt_metrics["overall_score"] - old_prompt_metrics["overall_score"]

if overall_diff > 1:

winner = f"{GREEN}New Prompt is better by {overall_diff:.2f}%{ENDC}"

elif overall_diff < -1:

winner = f"{RED}Old Prompt is better by {abs(overall_diff):.2f}%{ENDC}"

else:

winner = f"Both prompts perform similarly (diff: {overall_diff:+.2f}%)"

print(f"\n🏆 Overall Winner: {winner}")

# Generate key insights with more detailed analysis

print("\n🔍 Key Insights:")

insights = []

# Check main metrics for significant differences

if abs(new_prompt_metrics["summary_match"] - old_prompt_metrics["summary_match"]) > 5:

better = "New" if new_prompt_metrics["summary_match"] > old_prompt_metrics["summary_match"] else "Old"

insights.append(f"- {better} prompt is significantly better at summary generation")

if abs(new_prompt_metrics["nps_exact_match"] - old_prompt_metrics["nps_exact_match"]) > 5:

better = "New" if new_prompt_metrics["nps_exact_match"] > old_prompt_metrics["nps_exact_match"] else "Old"

insights.append(f"- {better} prompt is more accurate at determining NPS scores")

# Compare F1 scores

pos_f1_diff = new_prompt_metrics["positive_mentions_f1"] - old_prompt_metrics["positive_mentions_f1"]

if abs(pos_f1_diff) > 5:

better = "New" if pos_f1_diff > 0 else "Old"

insights.append(f"- {better} prompt performs better at identifying positive mentions (F1: {abs(pos_f1_diff):.2f}%)")

det_f1_diff = new_prompt_metrics["detractors_f1"] - old_prompt_metrics["detractors_f1"]

if abs(det_f1_diff) > 5:

better = "New" if det_f1_diff > 0 else "Old"

insights.append(f"- {better} prompt performs better at identifying detractors (F1: {abs(det_f1_diff):.2f}%)")

# Performance insight

if abs(time_diff) > 0.2:

faster = "New" if time_diff < 0 else "Old"

insights.append(f"- {faster} prompt has faster response times ({abs(time_diff):.2f}s difference)")

# If no significant insights, add a general one

if not insights:

insights.append("- Both prompts perform similarly across all metrics with no significant differences")

# Add challenging case analysis

insights.append(f"- New prompt handled {self.count_challenging_cases('new_prompt')}/5 challenging cases correctly")

insights.append(f"- Old prompt handled {self.count_challenging_cases('old_prompt')}/5 challenging cases correctly")

for insight in insights:

print(insight)

# Add ASCII chart visualization for key metrics comparison

print("\n📊 Visual Comparison of Key Metrics:")

self.print_ascii_chart(new_prompt_metrics, old_prompt_metrics)

# File path for saving results

results_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt_test_results.json")

# Save detailed results to file

with open(results_file_path, "w") as f:

json.dump({

"results": self.results,

"metrics": {

"new_prompt": new_prompt_metrics,

"old_prompt": old_prompt_metrics

},

"timing": {

"new_prompt": self.timing["new_prompt"],

"old_prompt": self.timing["old_prompt"],

"new_prompt_avg": new_prompt_avg_time,

"old_prompt_avg": old_prompt_avg_time

}

}, f, indent=2)

print(f"\nDetailed results saved to: {os.path.abspath(results_file_path)}")

# Show sample predictions vs expected with better formatting

print("\n" + "="*80)

print(f"{BOLD}SAMPLE PREDICTIONS COMPARISON{ENDC}")

print("="*80)

# Ask if user wants to see all transcripts or just a summary

print("\nDisplay options:")

print("1. Show detailed results for all 10 transcripts")

print("2. Show detailed results for first 3 transcripts only")

print("3. Show just a summary comparison table")

choice = input("Enter your choice (1-3) [default=2]: ").strip() or "2"

# Based on user's choice, show different levels of detail

if choice == "1":

num_to_show = len(SAMPLE_TRANSCRIPTS)

elif choice == "2":

num_to_show = min(3, len(SAMPLE_TRANSCRIPTS))

else: # choice == "3" or any other input

num_to_show = 0

# Show a summary comparison table instead

print("\n" + "="*110)

print(f"{BOLD}SUMMARY COMPARISON TABLE FOR ALL TRANSCRIPTS{ENDC}")

print("="*110)

print(f"{'#':<3} {'Expected NPS':<12} {'New NPS':<12} {'Old NPS':<12} {'Match?':<10} {'Expected Tags':<25} {'New Tags Match':<15} {'Old Tags Match':<15}")

print("-"*110)

for i in range(len(SAMPLE_TRANSCRIPTS)):

expected = EXPECTED_RESULTS[i]

new_result = self.results['new_prompt'][i]

old_result = self.results['old_prompt'][i]

# Check NPS match

new_nps_match = "✅" if new_result["nps_score"] == expected["nps_score"] else "❌"

old_nps_match = "✅" if old_result["nps_score"] == expected["nps_score"] else "❌"

# Calculate tag match percentages

new_pos_match = len(set(new_result["positive_mentions"]) & set(expected["positive_mentions"])) / max(1, len(set(expected["positive_mentions"])))

new_neg_match = len(set(new_result["detractors"]) & set(expected["detractors"])) / max(1, len(set(expected["detractors"])))

old_pos_match = len(set(old_result["positive_mentions"]) & set(expected["positive_mentions"])) / max(1, len(set(expected["positive_mentions"])))

old_neg_match = len(set(old_result["detractors"]) & set(expected["detractors"])) / max(1, len(set(expected["detractors"])))

# Format expected tags

exp_tags = f"+{','.join(expected['positive_mentions'])}" if expected["positive_mentions"] else ""

exp_tags += " " if expected["positive_mentions"] and expected["detractors"] else ""

exp_tags += f"-{','.join(expected['detractors'])}" if expected["detractors"] else ""

if not exp_tags:

exp_tags = "None"

new_tag_match = f"{(new_pos_match + new_neg_match)/2:.0%}" if expected["positive_mentions"] or expected["detractors"] else "N/A"

old_tag_match = f"{(old_pos_match + old_neg_match)/2:.0%}" if expected["positive_mentions"] or expected["detractors"] else "N/A"

print(f"{i+1:<3} {str(expected['nps_score']):<12} {str(new_result['nps_score']):<12} {str(old_result['nps_score']):<12} {new_nps_match+'/'+old_nps_match:<10} {exp_tags:<25} {new_tag_match:<15} {old_tag_match:<15}")

# Show detailed transcript analysis based on user choice

for i in range(num_to_show):

print(f"\n{BOLD}Transcript {i+1}:{ENDC}")

# Display a snippet of the conversation

print("\nConversation snippet:")

for j, msg in enumerate(SAMPLE_TRANSCRIPTS[i][-2:]): # Show just last 2 exchanges for brevity

role = msg["role"].capitalize()

message = msg["message"]

print(f" {role}: {message}")

# Display expected vs actual in a more structured way

print(f"\n{BOLD}Expected Output:{ENDC}")

expected = EXPECTED_RESULTS[i]

print(f" Summary: {expected['call_summary']}")

print(f" NPS Score: {expected['nps_score']}")

print(f" Positive Mentions: {', '.join(expected['positive_mentions']) if expected['positive_mentions'] else 'None'}")

print(f" Detractors: {', '.join(expected['detractors']) if expected['detractors'] else 'None'}")

print(f"\n{BOLD}New Prompt Output:{ENDC}")

new_result = self.results['new_prompt'][i]

print(f" Summary: {new_result['call_summary']}")

print(f" NPS Score: {new_result['nps_score']}")

print(f" Positive Mentions: {', '.join(new_result['positive_mentions']) if new_result['positive_mentions'] else 'None'}")

print(f" Detractors: {', '.join(new_result['detractors']) if new_result['detractors'] else 'None'}")

print(f"\n{BOLD}Old Prompt Output:{ENDC}")

old_result = self.results['old_prompt'][i]

print(f" Summary: {old_result['call_summary']}")

print(f" NPS Score: {old_result['nps_score']}")

print(f" Positive Mentions: {', '.join(old_result['positive_mentions']) if old_result['positive_mentions'] else 'None'}")

print(f" Detractors: {', '.join(old_result['detractors']) if old_result['detractors'] else 'None'}")

if i < min(3, len(SAMPLE_TRANSCRIPTS)) - 1:

print("\n" + "-"*80) # Separator between samples

  

async def main():

tester = PromptTester()

await tester.run_tests()

  

if __name__ == "__main__":

asyncio.run(main())