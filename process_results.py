import json

# This code is if you have the JSON files already and want to view it again
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