import json
import sys
from collections import defaultdict

def parse_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}.")
        return None

def analyze_poisoned_chunks(data):
    # Dictionary to store results grouped by chunk size
    results_by_chunk_size = defaultdict(lambda: {
        'total_instances': 0,
        'poisoned_instances': 0,
        'total_poisoned_chunks': 0,
        'contradiction_flags': 0,
        'factual_flags': 0,
        'instruction_flags': 0
    })

    # Process each test instance
    for result in data.get('results', []):
        chunk_size = result.get('chunk_size')
        is_poisoned = result.get('is_poisoned', False)

        # Increment total instances for this chunk size
        results_by_chunk_size[chunk_size]['total_instances'] += 1

        if is_poisoned:
            # Increment poisoned instances counter
            results_by_chunk_size[chunk_size]['poisoned_instances'] += 1

            # Count poisoned chunks
            poisoned_chunks = result.get('poisoned_chunks', [])
            results_by_chunk_size[chunk_size]['total_poisoned_chunks'] += len(poisoned_chunks)

            # Extract flag counts from verification metrics
            metrics_after_verification = result.get('metrics_after_verification', {})
            verification_metrics = metrics_after_verification.get('verification_metrics', {})

            if verification_metrics.get('contradiction_flags')>0:
              results_by_chunk_size[chunk_size]['contradiction_flags'] += 1
            if verification_metrics.get('factual_flags')>0:
              results_by_chunk_size[chunk_size]['factual_flags'] += 1
            if verification_metrics.get('instruction_flags')>0:
              results_by_chunk_size[chunk_size]['instruction_flags'] += 1
    for chunk_size in {128, 256, 512}:
      results_by_chunk_size[chunk_size]['contradiction_flags'] = results_by_chunk_size[chunk_size]['contradiction_flags']/results_by_chunk_size[chunk_size]['poisoned_instances']
      results_by_chunk_size[chunk_size]['factual_flags'] = results_by_chunk_size[chunk_size]['factual_flags']/results_by_chunk_size[chunk_size]['poisoned_instances']
      results_by_chunk_size[chunk_size]['instruction_flags'] = results_by_chunk_size[chunk_size]['instruction_flags']/results_by_chunk_size[chunk_size]['poisoned_instances']

    return results_by_chunk_size

def print_results(results):
    if not results:
        print("No results to display.")
        return

    print("\nAnalysis Results by Chunk Size:")
    print("-" * 80)
    print(f"{'Chunk Size':<12} {'Total':<10} {'Poisoned':<10} {'Poisoned':<12} {'Contradiction':<15} {'Factual':<10} {'Instruction':<12}")
    print(f"{'':12} {'Instances':<10} {'Instances':<10} {'Chunks':<12} {'Flags':<15} {'Flags':<10} {'Flags':<12}")
    print("-" * 80)

    for chunk_size, data in sorted(results.items()):
        print(f"{chunk_size:<12} {data['total_instances']:<10} {data['poisoned_instances']:<10} "
              f"{data['total_poisoned_chunks']:<12} {data['contradiction_flags']:<15} "
              f"{data['factual_flags']:<10} {data['instruction_flags']:<12}")

def main():

    file_path = ""  # Default filename

    data = parse_json_file(file_path)
    if data:
        results = analyze_poisoned_chunks(data)
        print_results(results)

if __name__ == "__main__":
    main()
