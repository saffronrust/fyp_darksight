import json

# Load the JSON data
with open('evaluation_results.json', 'r') as f:
    data = json.load(f)

total_evaluations = len(data)
# Initialize dictionaries to hold the scores
clean_scores = {'accuracy': [], 'completeness': [], 'clarity': []}
poisoned_scores = {'accuracy': [], 'completeness': [], 'clarity': []}
cleanVLM_wins = 0

# Extract scores and winner info
for image, results in data.items():
    # Count the wins
    if results.get('winner') == 'cleanVLM':
        cleanVLM_wins += 1
        
    # Append scores to lists
    for metric in ['accuracy', 'completeness', 'clarity']:
        clean_scores[metric].append(results['cleanVLM_scores'][metric])
        poisoned_scores[metric].append(results['poisonedVLM_scores'][metric])

# Function to calculate averages
def calculate_averages(scores_dict):
    return {metric: sum(scores) / len(scores) for metric, scores in scores_dict.items()}

# Calculate averages
clean_avg = calculate_averages(clean_scores)
poisoned_avg = calculate_averages(poisoned_scores)

# Display results
print("Clean VLM Average Scores:")
print(f"  Accuracy: {clean_avg['accuracy']}")
print(f"  Completeness: {clean_avg['completeness']}")
print(f"  Clarity: {clean_avg['clarity']}\n")

print("Poisoned VLM Average Scores:")
print(f"  Accuracy: {poisoned_avg['accuracy']}")
print(f"  Completeness: {poisoned_avg['completeness']}")
print(f"  Clarity: {poisoned_avg['clarity']}\n")

print(f"Number of times 'cleanVLM' won: {cleanVLM_wins} out of {total_evaluations} evaluations")