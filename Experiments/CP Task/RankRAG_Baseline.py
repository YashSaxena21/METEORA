import os
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from functools import lru_cache
import re
from typing import List, Dict, Set, Any, Union, Optional, Tuple
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor

@lru_cache(maxsize=32)
def load_legalbench_rag(dataset_path: str) -> List[Dict]:
    """
    Load the LegalBench RAG benchmark dataset from a JSON file with caching.
    Supports multiple dataset formats (contractnli, cuad, maud, privacyqa).
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Handle different dataset formats
    if "tests" in data:
        return data["tests"]
    elif isinstance(data, list):
        return data
    else:
        # Try to extract the main list from the first key
        if isinstance(data, dict) and len(data) > 0:
            first_key = next(iter(data))
            if isinstance(data[first_key], list):
                return data[first_key]
    
    # Default fallback
    return [data] if isinstance(data, dict) else []

@lru_cache(maxsize=64)
def load_corpus(document_path: str) -> str:
    """
    Load and cache the text content of a given document.
    """
    if not os.path.exists(document_path):
        raise FileNotFoundError(f"Document not found at {document_path}")
    
    with open(document_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    return content

def direct_chunk_selection(model, tokenizer, query: str, chunks: List[Dict], 
                          device: str, max_context_length: int = 100000, 
                          top_k_chunks: Optional[int] = None) -> str:
    """
    Generate a response from Llama model to directly identify the correct chunks.
    Optionally limit to top-k chunks.
    """
    # Create a structured representation of chunks within context limit
    chunk_info = []
    current_length = 0
    
    # If top_k_chunks is specified, select only top chunks
    if top_k_chunks is not None:
        chunks = chunks[:top_k_chunks]
    
    for i, chunk in enumerate(chunks):
        chunk_text = f"Chunk {i}: {chunk['text']}\n\n"
        chunk_length = len(chunk_text)
        
        if current_length + chunk_length < max_context_length:
            chunk_info.append(chunk_text)
            current_length += chunk_length
        else:
            chunk_info.append(f"Chunk {i}: [Content truncated due to length]\n\n")
            current_length += len(chunk_info[-1])
            
        if current_length >= max_context_length:
            break
    
    chunk_info_str = "".join(chunk_info)
    
    # Construct prompt for Llama 3.1 with exact top-k restriction
    exact_k = "" if top_k_chunks is None else f"IMPORTANT: You must select EXACTLY {top_k_chunks} chunks, no more and no less."
    
    prompt = f"""<|begin_of_text|><|system|>
You are an AI assistant specialized in legal information retrieval. Your task is to identify the most relevant chunks from a legal document that answer a given query.

Return ONLY the chunk numbers in a comma-separated list format without explanation.
{exact_k}
<|user|>
Task: Legal Information Retrieval

Query: {query}

Below are chunks from a legal document. Identify the chunk numbers that directly answer the query.
Only include chunk numbers that DIRECTLY contain relevant information.
{'' if top_k_chunks is None else f'You MUST select EXACTLY {top_k_chunks} chunks, even if some seem less relevant.'}

Document Chunks:
{chunk_info_str}

Based on the query and document chunks above, which chunk numbers directly contain the relevant information to answer the query?
{'' if top_k_chunks is None else f'Remember to select EXACTLY {top_k_chunks} chunks, no more and no less.'}
<|assistant|>
RELEVANT CHUNKS: """
    
    # Generate response from model
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,  # Reduced for a short answer
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7
        )
    
    response = tokenizer.decode(output[0], skip_special_tokens=False)
    return response

def extract_chunk_numbers(response: str, exact_k: Optional[int] = None) -> List[int]:
    """
    Extract chunk numbers from the Llama model's response with optimized regex.
    Returns a list of chunk indices.
    
    If exact_k is provided, ensures exactly that many chunks are returned by either:
    - Truncating if more are found
    - Padding with sequential chunks if fewer are found
    """
    # First look for the assistant tag
    assistant_pattern = r"<\|assistant\|>\s*(?:RELEVANT CHUNKS:|relevant chunks:)?\s*(.*?)(?:\s*$|\s*<|\s*[a-zA-Z])"
    
    extracted_numbers = []
    
    # Try to find content after the assistant tag
    assistant_match = re.search(assistant_pattern, response, re.DOTALL | re.IGNORECASE)
    if assistant_match:
        # Extract the content after assistant tag
        content = assistant_match.group(1).strip()
        
        # Now extract all numbers from this content
        number_matches = re.findall(r'\b\d+\b', content)
        if number_matches:
            for num in number_matches:
                try:
                    extracted_numbers.append(int(num))
                except ValueError:
                    continue
    
    # If we didn't find any valid numbers through the above method
    if not extracted_numbers:
        # Try an alternative pattern specifically for "RELEVANT CHUNKS:" format
        chunks_pattern = r"RELEVANT CHUNKS:\s*((?:\d+(?:\s*,\s*\d+)*)?)"
        chunks_match = re.search(chunks_pattern, response, re.IGNORECASE)
        
        if chunks_match:
            chunk_str = chunks_match.group(1).strip()
            # Split by comma or spaces
            for chunk in re.split(r'[,\s]+', chunk_str):
                if chunk.strip().isdigit():
                    extracted_numbers.append(int(chunk.strip()))
    
    # Last resort fallback: get any numbers from the full response if needed
    if not extracted_numbers:
        all_numbers = re.findall(r'\b\d+\b', response)
        extracted_numbers = [int(num) for num in all_numbers if num.isdigit()]
    
    # If exact_k is specified, ensure we have exactly that many chunks
    if exact_k is not None:
        if len(extracted_numbers) > exact_k:
            # Keep only the first exact_k chunks if we have too many
            extracted_numbers = extracted_numbers[:exact_k]
        elif len(extracted_numbers) < exact_k:
            # If we don't have enough, pad with sequential chunks starting from 0
            # First identify which chunks aren't already selected
            all_possible_chunks = set(range(exact_k * 2))  # Reasonable upper limit
            available_chunks = list(all_possible_chunks - set(extracted_numbers))
            available_chunks.sort()  # Sort to get sequential chunks
            
            # Add chunks until we reach exact_k
            while len(extracted_numbers) < exact_k and available_chunks:
                extracted_numbers.append(available_chunks.pop(0))
    
    return extracted_numbers

def calculate_metrics(selected_chunks: List[int], correct_chunks: List[int]) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 score for chunk selection.
    """
    if not selected_chunks or not correct_chunks:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "correct_chunk_found": False
        }
    
    # Convert to sets for efficient operations
    selected_set = set(selected_chunks)
    correct_set = set(correct_chunks)
    
    # Calculate intersection
    overlap = len(selected_set.intersection(correct_set))
    
    # Calculate metrics
    precision = overlap / len(selected_set) if selected_set else 0.0
    recall = overlap / len(correct_set) if correct_set else 0.0
    
    # Calculate F1 score
    f1_score = 0.0
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "correct_chunk_found": overlap > 0
    }

def get_text_between_positions(file_path: str, start: int, end: int) -> Optional[str]:
    """
    Extract text between specified character positions in a file with error handling.
    """
    try:
        content = load_corpus(file_path)
        if start < 0 or end > len(content) or start > end:
            raise IndexError("Invalid range: Ensure 0 ≤ start ≤ end ≤ document length.")
        return content[start:end]
    except Exception as e:
        print(f"Error extracting text: {e}")
        return None

def chunk_text(text: str, chunk_size: int) -> List[Dict[str, Any]]:
    """
    Optimized function to split text into chunks with position tracking.
    """
    words = text.split()
    chunks = []
    
    # Get word positions in a single pass (more efficient)
    word_positions = []
    current_pos = 0
    
    for word in words:
        word_positions.append(current_pos)
        current_pos += len(word) + 1  # +1 for the space
    
    # Calculate chunk parameters
    overlap = chunk_size // 10  # 10% overlap
    step_size = chunk_size - overlap
    
    # Create chunks
    for i in range(0, len(words), step_size):
        end_idx = min(i + chunk_size, len(words))
        
        # Get positions
        if i >= len(word_positions):
            break
            
        start_pos = word_positions[i]
        end_pos = word_positions[end_idx - 1] + len(words[end_idx - 1]) if end_idx - 1 < len(word_positions) else len(text)
        
        # Create chunk
        chunk = {
            "text": " ".join(words[i:end_idx]),
            "start_pos": start_pos,
            "end_pos": end_pos
        }
        chunks.append(chunk)
        
        # Break if we've reached the end
        if end_idx >= len(words):
            break
    
    return chunks

def find_spanning_chunks(span_text: Optional[str], span_start: Optional[int], 
                        span_end: Optional[int], chunks: List[Dict]) -> List[int]:
    """
    Find all chunks that contain any part of the span text based on character positions.
    """
    if span_text is None or span_start is None or span_end is None:
        return []
    
    # Find all chunks that overlap with the span
    spanning_chunks = [
        i for i, chunk in enumerate(chunks)
        if chunk["start_pos"] <= span_end and chunk["end_pos"] >= span_start
    ]
    
    # Ensure we have a consecutive range of chunks
    if spanning_chunks:
        min_chunk = min(spanning_chunks)
        max_chunk = max(spanning_chunks)
        spanning_chunks = list(range(min_chunk, max_chunk + 1))
    
    return spanning_chunks

def extract_dataset_name(dataset_path: str) -> str:
    """
    Extract dataset name from the file path.
    """
    base_name = os.path.basename(dataset_path)
    file_name = os.path.splitext(base_name)[0].lower()
    
    if "contract" in file_name:
        return "contractnli"
    elif "cuad" in file_name:
        return "cuad"
    elif "maud" in file_name:
        return "maud"
    elif "privacy" in file_name:
        return "privacyqa"
    else:
        return "custom"

def parse_sample_by_dataset(sample: Dict, dataset_name: str) -> Dict:
    """
    Parse sample data based on the dataset type.
    Returns standardized fields that all datasets should have.
    """
    standardized = {
        "id": sample.get("id", str(id(sample))),
        "query": "",
        "snippets": []
    }
    
    # Dataset-specific parsing
    if dataset_name == "contractnli":
        standardized["query"] = sample.get("query", "")
        if "snippets" in sample and len(sample["snippets"]) > 0:
            standardized["snippets"] = sample["snippets"]
    
    elif dataset_name == "cuad":
        # CUAD format
        standardized["query"] = sample.get("question", "")
        if "context" in sample and "file_path" in sample:
            standardized["snippets"] = [{
                "file_path": sample["file_path"],
                "span": sample.get("span", [0, 0])
            }]
    
    elif dataset_name == "maud":
        # MAUD format
        standardized["query"] = sample.get("question", "")
        if "file_path" in sample:
            standardized["snippets"] = [{
                "file_path": sample["file_path"],
                "span": sample.get("span", sample.get("answer_span", [0, 0]))
            }]
    
    elif dataset_name == "privacyqa":
        # PrivacyQA format
        standardized["query"] = sample.get("question", sample.get("query", ""))
        if "document" in sample and "path" in sample["document"]:
            standardized["snippets"] = [{
                "file_path": sample["document"]["path"],
                "span": sample.get("span", sample.get("answer_span", [0, 0]))
            }]
    
    return standardized

def prepare_sample_data(sample: Dict, chunk_sizes: List[int], dataset_name: str, 
                       data_dir: str = "Data") -> List[Tuple[Dict, int]]:
    """
    Prepare data for each sample and chunk size based on the dataset type.
    """
    sample_data_list = []
    
    # Parse sample based on dataset type
    std_sample = parse_sample_by_dataset(sample, dataset_name)
    
    # Skip if no snippets or query
    if not std_sample["snippets"] or not std_sample["query"]:
        return sample_data_list
    
    # Get document path and span information
    doc_info = std_sample["snippets"][0]
    doc_path = os.path.join(data_dir, doc_info["file_path"])
    
    if not os.path.exists(doc_path):
        # Try relative path without the data_dir
        doc_path = doc_info["file_path"]
        if not os.path.exists(doc_path):
            print(f"Warning: Document not found: {doc_path}")
            return sample_data_list
    
    # Extract span information
    span_start = None
    span_end = None
    span_text = None
    
    span_field = doc_info.get("span", [])
    if isinstance(span_field, list) and len(span_field) == 2:
        span_start = span_field[0]
        span_end = span_field[1]
        span_text = get_text_between_positions(doc_path, span_start, span_end)
    
    # Load document text
    try:
        corpus_text = load_corpus(doc_path)
    except Exception as e:
        print(f"Error loading corpus: {e}")
        return sample_data_list
    
    # Process with different chunk sizes
    for chunk_size in chunk_sizes:
        # Create chunks
        chunks = chunk_text(corpus_text, chunk_size)
        
        # Find correct chunks
        correct_chunk_numbers = find_spanning_chunks(span_text, span_start, span_end, chunks)
        
        # Skip if no correct chunks found
        if not correct_chunk_numbers:
            continue
            
        # Create data object
        sample_data = {
            "sample_id": std_sample["id"],
            "query": std_sample["query"],
            "chunks": chunks,
            "correct_chunk_numbers": correct_chunk_numbers,
            "chunk_size": chunk_size,
            "total_chunks": len(chunks)
        }
        
        sample_data_list.append((sample_data, chunk_size))
    
    return sample_data_list

def process_batch_sample(batch_item: Tuple[Dict, int], model, tokenizer, device: str, 
                        top_k_dict: Dict) -> Dict:
    """
    Process a single sample with model inference.
    """
    sample_data, chunk_size = batch_item
    query = sample_data["query"]
    chunks = sample_data["chunks"]
    correct_chunk_numbers = sample_data["correct_chunk_numbers"]
    
    try:
        # Get top-k for this chunk size
        top_k = top_k_dict.get(chunk_size, None)
        
        # Get model response
        model_response = direct_chunk_selection(
            model, 
            tokenizer, 
            query, 
            chunks, 
            device, 
            top_k_chunks=top_k
        )
        
        # Extract chunk numbers with exact_k enforcement
        selected_chunks = extract_chunk_numbers(model_response, exact_k=top_k)
        
        # Calculate metrics
        metrics = calculate_metrics(selected_chunks, correct_chunk_numbers)
        
        # Prepare result
        result = {
            "sample_id": sample_data["sample_id"],
            "query": query,
            "chunk_size": chunk_size,
            "total_chunks": len(chunks),
            "span_containing_chunks": correct_chunk_numbers,
            "model_response": model_response,
            "selected_chunks": selected_chunks,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "correct_chunk_found": metrics["correct_chunk_found"]
        }
        
        return result
        
    except Exception as e:
        print(f"Error processing sample: {e}")
        return {
            "sample_id": sample_data["sample_id"],
            "query": query,
            "chunk_size": chunk_size,
            "error": str(e)
        }

def update_overall_metrics(overall_metrics: Dict, result: Dict) -> None:
    """
    Update overall metrics with results from a single sample.
    """
    if "error" in result:
        return
        
    chunk_size = result["chunk_size"]
    
    overall_metrics[chunk_size]["precision_sum"] += result["precision"]
    overall_metrics[chunk_size]["recall_sum"] += result["recall"]
    overall_metrics[chunk_size]["f1_sum"] += result["f1_score"]
    overall_metrics[chunk_size]["selected_chunks_sum"] += len(result["selected_chunks"])
    overall_metrics[chunk_size]["count"] += 1
    
    if result["correct_chunk_found"]:
        overall_metrics[chunk_size]["correct_count"] += 1

def process_batch(batch: List[Tuple[Dict, int]], model, tokenizer, device: str, 
                top_k_dict: Dict, overall_metrics: Dict) -> List[Dict]:
    """
    Process a batch of samples.
    """
    results = []
    
    for batch_item in batch:
        result = process_batch_sample(batch_item, model, tokenizer, device, top_k_dict)
        results.append(result)
        update_overall_metrics(overall_metrics, result)
    
    return results

def save_results(all_results: List[Dict], overall_metrics: Dict, chunk_sizes: List[int], 
                output_path: str, dataset_name: str) -> None:
    """
    Calculate final metrics and save results to file.
    """
    # Calculate average metrics
    final_metrics = {}
    
    for chunk_size in chunk_sizes:
        metrics = overall_metrics[chunk_size]
        count = metrics["count"]
        
        if count > 0:
            final_metrics[chunk_size] = {
                "avg_precision": metrics["precision_sum"] / count,
                "avg_recall": metrics["recall_sum"] / count,
                "avg_f1": metrics["f1_sum"] / count,
                "avg_selected_chunks": metrics["selected_chunks_sum"] / count,
                "sample_count": count,
                "correct_count": metrics["correct_count"],
                "correct_percentage": (metrics["correct_count"] / count) * 100
            }
        else:
            final_metrics[chunk_size] = {
                "avg_precision": 0.0,
                "avg_recall": 0.0,
                "avg_f1": 0.0,
                "avg_selected_chunks": 0.0,
                "sample_count": 0,
                "correct_count": 0,
                "correct_percentage": 0.0
            }
    
    # Prepare final output
    final_output = {
        "dataset": dataset_name,
        "results": all_results,
        "overall_metrics": final_metrics
    }
    
    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4)
    
    print(f"Results saved to {output_path}")
    
    # Print summary
    print("\nSummary:")
    print(f"Dataset: {dataset_name}")
    print("Chunk Size | Avg Precision | Avg Recall | Avg F1 | Correct % | Samples")
    print("-" * 70)
    for size in chunk_sizes:
        if size in final_metrics:
            m = final_metrics[size]
            print(f"{size:10} | {m['avg_precision']:.4f} | {m['avg_recall']:.4f} | {m['avg_f1']:.4f} | {m['correct_percentage']:7.2f}% | {m['sample_count']}")

def process_batches_with_threading(all_batch_items: List[Tuple[Dict, int]], model, tokenizer, 
                                 device: str, top_k_dict: Dict, overall_metrics: Dict,
                                 batch_size: int, max_workers: int, dataset_name: str) -> List[Dict]:
    """
    Process batches with threading for improved efficiency.
    """
    all_results = []
    total_batches = (len(all_batch_items) + batch_size - 1) // batch_size
    
    with tqdm(total=len(all_batch_items), desc="Processing samples") as pbar:
        for i in range(0, len(all_batch_items), batch_size):
            batch = all_batch_items[i:i+batch_size]
            
            # Process batch with ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        process_batch_sample, 
                        batch_item, 
                        model, 
                        tokenizer, 
                        device, 
                        top_k_dict
                    ) 
                    for batch_item in batch
                ]
                
                # Collect results as they complete
                batch_results = []
                for future in futures:
                    result = future.result()
                    batch_results.append(result)
                    update_overall_metrics(overall_metrics, result)
                    pbar.update(1)
            
            # Add batch results
            all_results.extend(batch_results)
            
            # Print progress
            print(f"\nCompleted batch {(i//batch_size)+1}/{total_batches}")
            
            # Save intermediate results
            if (i // batch_size + 1) % 50 == 0:  # Save every 50 batches
                print("Saving intermediate results...")
                save_results(all_results, overall_metrics, list(top_k_dict.keys()), 
                            f"intermediate_results_{dataset_name}_{i//batch_size}.json",
                            dataset_name)
    
    return all_results

def get_top_k_values(dataset_name: str, chunk_sizes: List[int]) -> Dict[int, int]:
    """
    Get dataset-specific top-k values for each chunk size.
    These values are optimized for each dataset based on empirical testing.
    """
    top_k_defaults = {
        "contractnli": {128: 7, 256: 5, 512: 3},
        "cuad": {128: 8, 256: 6, 512: 4},
        "maud": {128: 7, 256: 5, 512: 3},
        "privacyqa": {128: 6, 256: 4, 512: 3},
        "custom": {128: 7, 256: 5, 512: 3}  # Default values for unknown datasets
    }
    
    return top_k_defaults.get(dataset_name, top_k_defaults["custom"])

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="LegalBench RAG Benchmark for multiple datasets")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset JSON file")
    parser.add_argument("--data_dir", type=str, default="Data", help="Directory containing document files")
    parser.add_argument("--output", type=str, default=None, help="Output file path (default: results_[dataset_name].json)")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for processing")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of worker threads")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index in dataset")
    parser.add_argument("--end_idx", type=int, default=None, help="End index in dataset (exclusive)")
    parser.add_argument("--chunk_sizes", type=str, default="128,256,512", help="Comma-separated list of chunk sizes")
    args = parser.parse_args()
    
    # Extract dataset name from path
    dataset_name = extract_dataset_name(args.dataset)
    print(f"Detected dataset type: {dataset_name}")
    
    # Set output path if not provided
    if args.output is None:
        args.output = f"results_{dataset_name}_llama31.json"
    
    # Parse chunk sizes
    chunk_sizes = [int(size.strip()) for size in args.chunk_sizes.split(",")]
    print(f"Using chunk sizes: {chunk_sizes}")
    
    # Get top-k values for this dataset
    top_k_dict = get_top_k_values(dataset_name, chunk_sizes)
    print(f"Using top-k values: {top_k_dict}")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load Llama-3.1-8B-Instruct model
    llm_model_name = "meta-llama/Llama-3.1-8B-Instruct"
    print(f"Loading model: {llm_model_name}")
    
    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    try:
        data = load_legalbench_rag(args.dataset)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Apply start and end indices if provided
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx is not None else len(data)
    data = data[start_idx:end_idx]
    
    print(f"Processing samples {start_idx} to {end_idx-1} (total: {len(data)})")
    
    # Initialize metrics tracking
    overall_metrics = {
        size: {
            "precision_sum": 0.0, 
            "recall_sum": 0.0, 
            "f1_sum": 0.0, 
            "count": 0, 
            "correct_count": 0, 
            "selected_chunks_sum": 0
        } 
        for size in chunk_sizes
    }
    
    # Prepare all samples first (data preparation)
    print("Preparing sample data...")
    all_batch_items = []
    
    for sample in tqdm(data, desc="Preparing samples"):
        sample_data_list = prepare_sample_data(sample, chunk_sizes, dataset_name, args.data_dir)
        all_batch_items.extend(sample_data_list)
    
    print(f"Total batch items prepared: {len(all_batch_items)}")
    
    if len(all_batch_items) == 0:
        print("No valid samples found. Please check your dataset format and paths.")
        return
    
    # Process all samples in batches with threading
    all_results = process_batches_with_threading(
        all_batch_items=all_batch_items,
        model=llm_model,
        tokenizer=llm_tokenizer,
        device=device,
        top_k_dict=top_k_dict,
        overall_metrics=overall_metrics,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        dataset_name=dataset_name
    )
    
    # Save final results
    save_results(all_results, overall_metrics, chunk_sizes, args.output, dataset_name)
    print(f"Processing complete. Total samples: {len(data)}")

if __name__ == "__main__":
    main()
