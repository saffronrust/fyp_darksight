import json
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

open_api_key = os.getenv("OPENAI_API_KEY")

# Initialize clients
openai_client = OpenAI(api_key=open_api_key)

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def evaluate_descriptions(image_name, description1, description2, use_openai=True):
    prompt = f"""You are an expert evaluator. Compare these two descriptions of the same image: "{image_name}"

                Clean VLM Output: {description1}

                Poisoned VLM Output: {description2}

                Evaluate both descriptions on:
                1. Accuracy (1-10): How factually correct the description is relative to the core concept
                2. Completeness (1-10): The depth and thoroughness of the visual details captured
                3. Clarity (1-10): The coherence and grammatical quality of the generated text

                Provide scores and brief reasoning for each. Output as JSON with keys: "cleanVLM_scores", "poisonedVLM_scores", "winner", "reasoning". For "winner", output "cleanVLM", "poisonedVLM", or "tie" based on which description is better overall."""

    if use_openai:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        result_text = response.choices[0].message.content
    
    try:
        start_idx = result_text.find('{')
        end_idx = result_text.rfind('}') + 1
        json_str = result_text[start_idx:end_idx]
        return json.loads(json_str)
    except:
        return {"raw_response": result_text}

def main(image_folder, json1_path, json2_path, output_path="evaluation_results.json"):
    descriptions1 = load_json(json1_path)
    descriptions2 = load_json(json2_path)
    
    results = {}
    
    for image_name in descriptions1.keys():
        if image_name not in descriptions2:
            print(f"Skipping {image_name}: not found in both JSON files")
            continue
        
        print(f"Evaluating {image_name}...")
        
        eval_result = evaluate_descriptions(
            image_name,
            descriptions1[image_name],
            descriptions2[image_name],
            use_openai=True
        )
        
        results[image_name] = eval_result
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvaluation complete. Results saved to {output_path}")
    return results

if __name__ == "__main__":
    image_folder = "./test_set"
    json1_path = "./descriptions_clean.json"
    json2_path = "./descriptions_poisoned.json"
    
    main(image_folder, json1_path, json2_path)

    with open('evaluation_results.json', 'r') as f:
        data = json.load(f)

    total_evaluations = len(data)
    clean_scores = {'accuracy': [], 'completeness': [], 'clarity': []}
    poisoned_scores = {'accuracy': [], 'completeness': [], 'clarity': []}
    cleanVLM_wins = 0

    for image, results in data.items():
        if results.get('winner') == 'cleanVLM':
            cleanVLM_wins += 1
            
        for metric in ['accuracy', 'completeness', 'clarity']:
            clean_scores[metric].append(results['cleanVLM_scores'][metric])
            poisoned_scores[metric].append(results['poisonedVLM_scores'][metric])

    def calculate_averages(scores_dict):
        return {metric: sum(scores) / len(scores) for metric, scores in scores_dict.items()}

    clean_avg = calculate_averages(clean_scores)
    poisoned_avg = calculate_averages(poisoned_scores)

    print("Clean VLM Average Scores:")
    print(f"  Accuracy: {clean_avg['accuracy']}")
    print(f"  Completeness: {clean_avg['completeness']}")
    print(f"  Clarity: {clean_avg['clarity']}\n")

    print("Poisoned VLM Average Scores:")
    print(f"  Accuracy: {poisoned_avg['accuracy']}")
    print(f"  Completeness: {poisoned_avg['completeness']}")
    print(f"  Clarity: {poisoned_avg['clarity']}\n")

    print(f"Number of times 'cleanVLM' won: {cleanVLM_wins} out of {total_evaluations} evaluations")