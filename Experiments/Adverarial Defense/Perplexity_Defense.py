import os
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("debug_log.txt"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_legalbench_rag(dataset_path):
    """Load the LegalBench RAG benchmark dataset from a JSON file."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data["tests"]  # Extracting the list of test cases

def load_corpus(document_path):
    """Load the text content of a given document."""
    if not os.path.exists(document_path):
        raise FileNotFoundError(f"Document not found at {document_path}")
    
    with open(document_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    return content

def chunk_text(text, chunk_size):
    """
    Split text into chunks of specified size, with slight overlap.
    Returns a list of chunks with their original character positions.
    """
    words = text.split()
    chunks = []
    word_positions = []
    
    # Get the starting position of each word
    current_pos = 0
    for word in words:
        word_positions.append(current_pos)
        current_pos += len(word) + 1  # +1 for the space
    
    overlap = chunk_size // 10  # 10% overlap
    step_size = chunk_size - overlap
    
    for i in range(0, len(words), step_size):
        end_idx = min(i + chunk_size, len(words))
        
        # Get the start and end positions in the original text
        if i < len(word_positions):
            start_pos = word_positions[i]
        else:
            break
            
        if end_idx - 1 < len(word_positions):
            end_pos = word_positions[end_idx - 1] + len(words[end_idx - 1])
        else:
            end_pos = len(text)
        
        chunk = " ".join(words[i:end_idx])
        chunks.append({
            "text": chunk,
            "start_pos": start_pos,
            "end_pos": end_pos
        })
        
        # If we've reached the end of the document, break
        if end_idx >= len(words):
            break
    
    return chunks

def find_spanning_chunks(span_start, span_end, chunks):
    """
    Find all chunks that contain any part of the span based on character positions.
    Returns a list of chunk indices that contain the span.
    """
    if span_start is None or span_end is None:
        return []
    
    spanning_chunks = []
    
    for i, chunk in enumerate(chunks):
        # Check if there's any overlap between the span and the chunk
        if (chunk["start_pos"] <= span_end and chunk["end_pos"] >= span_start):
            spanning_chunks.append(i)
    
    # Verify the spanning chunks are consecutive
    if spanning_chunks:
        min_chunk = min(spanning_chunks)
        max_chunk = max(spanning_chunks)
        # All chunks between min and max should be included
        expected_chunks = list(range(min_chunk, max_chunk + 1))
        
        if sorted(spanning_chunks) != expected_chunks:
            logger.warning(f"Non-consecutive chunks detected: {spanning_chunks}")
            # Ensure we have all chunks in the range
            spanning_chunks = expected_chunks
    
    return spanning_chunks

def calculate_perplexity(model, tokenizer, text, device="cuda"):
    """
    Calculate the perplexity of text using a language model.
    Lower perplexity indicates the text is more predictable/natural.
    """
    # Tokenize input text
    encodings = tokenizer(text, return_tensors="pt").to(device)
    
    # Get sequence length for proper perplexity calculation
    seq_len = encodings.input_ids.size(1)
    
    # Calculate perplexity using causal language modeling (disable cache for accurate results)
    with torch.no_grad():
        outputs = model(**encodings, labels=encodings.input_ids)
        neg_log_likelihood = outputs.loss.item()
    
    # Calculate perplexity (exp of the average negative log-likelihood)
    perplexity = torch.exp(torch.tensor(neg_log_likelihood)).item()
    
    return perplexity

def retrieve_relevant_chunks(query, chunks, sbert_model, max_chunks=5, device="cuda"):
    """
    Retrieve the most relevant chunks for a given query using semantic similarity.
    """
    # Encode the query
    query_embedding = sbert_model.encode(query, convert_to_tensor=True).to(device)
    
    # Encode all chunks
    chunk_texts = [chunk["text"] for chunk in chunks]
    chunk_embeddings = sbert_model.encode(chunk_texts, convert_to_tensor=True).to(device)
    
    # Calculate similarities between query and all chunks
    similarities = util.pytorch_cos_sim(query_embedding, chunk_embeddings)[0]
    
    # Sort chunks by similarity score (descending)
    sorted_indices = torch.argsort(similarities, descending=True)
    # Convert to native Python integers to avoid JSON serialization issues
    selected_indices = [int(idx) for idx in sorted_indices[:max_chunks].cpu().numpy()]
    
    # Create result object with JSON-serializable values
    result = {
        "selected_chunks": selected_indices,
        "similarities": {int(i): float(similarities[i].item()) for i in selected_indices}
    }
    
    return result

def detect_poisoned_chunks(chunks, selected_indices, model, tokenizer, perplexity_threshold=20.0, device="cuda"):
    """
    Detect poisoned chunks using perplexity analysis.
    Returns a list of flagged chunk indices and their perplexity scores.
    """
    flagged_chunks = []
    perplexity_scores = {}
    
    # Calculate mean and std of perplexities across all chunks
    all_perplexities = []
    for i in selected_indices:
        perplexity = calculate_perplexity(model, tokenizer, chunks[i]["text"], device)
        all_perplexities.append(perplexity)
        perplexity_scores[int(i)] = float(perplexity)  # Ensure JSON serializable
    
    dynamic_threshold = perplexity_threshold
    if all_perplexities:
        mean_perplexity = float(np.mean(all_perplexities))
        std_perplexity = float(np.std(all_perplexities))
        
        # Dynamic thresholding: Flag chunks with abnormally high perplexity
        # (mean + 1.5*std or fixed threshold, whichever is lower)
        dynamic_threshold = min(mean_perplexity + 1.5 * std_perplexity, perplexity_threshold)
        
        # Flag chunks with perplexity above threshold
        for i in selected_indices:
            if perplexity_scores[int(i)] > dynamic_threshold:
                flagged_chunks.append(int(i))  # Ensure JSON serializable
    
    return {
        "flagged_chunks": flagged_chunks,
        "perplexity_scores": perplexity_scores,
        "threshold": float(dynamic_threshold)  # Ensure JSON serializable
    }

def calculate_precision_recall(selected_chunks, correct_chunks, chunks):
    """
    Calculate precision and recall for chunk selection.
    """
    # Convert to sets for easier calculation
    selected_set = set(selected_chunks)
    correct_set = set(correct_chunks)
    
    # Calculate intersection of selected and correct chunks
    correct_selected = selected_set.intersection(correct_set)
    
    # Calculate precision and recall
    precision = len(correct_selected) / len(selected_set) if selected_set else 0.0
    recall = len(correct_selected) / len(correct_set) if correct_set else 0.0
    
    return {
        "precision": float(precision),  # Ensure JSON serializable
        "recall": float(recall),        # Ensure JSON serializable
        "correct_chunk_found": len(correct_selected) > 0
    }

def calculate_f1_score(precision, recall):
    """Calculate F1 score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return float(2 * (precision * recall) / (precision + recall))  # Ensure JSON serializable

def initialize_metrics(chunk_sizes):
    """Initialize metrics dictionary for tracking results."""
    metrics = {}
    for chunk_size in chunk_sizes:
        metrics[chunk_size] = {
            "precision_sum": 0.0,
            "recall_sum": 0.0,
            "f1_sum": 0.0,
            "precision_after_sum": 0.0,
            "recall_after_sum": 0.0,
            "f1_after_sum": 0.0,
            "test_count": 0,
            "correct_chunk_found_count": 0,
            "chunk_count_sum": 0,
            "flagged_chunk_count_sum": 0,
            "poison_detection": {
                "total_poisoned_instances": 0,
                "poisoned_chunks_selected": 0,
                "poisoned_chunks_flagged": 0,
                "correctly_flagged_as_poisoned": 0,
                "incorrectly_flagged_as_poisoned": 0
            }
        }
    return metrics

def update_metrics(metrics, chunk_size, metrics_before, metrics_after, f1_before, f1_after, 
                   selected_chunks, non_flagged_chunks, flagged_chunks, 
                   is_poisoned=False, poisoned_chunks=None, poisoned_chunks_selected=None):
    """Update metrics with results from a test instance."""
    metrics[chunk_size]["precision_sum"] += metrics_before["precision"]
    metrics[chunk_size]["recall_sum"] += metrics_before["recall"]
    metrics[chunk_size]["f1_sum"] += f1_before
    metrics[chunk_size]["precision_after_sum"] += metrics_after["precision"]
    metrics[chunk_size]["recall_after_sum"] += metrics_after["recall"]
    metrics[chunk_size]["f1_after_sum"] += f1_after
    metrics[chunk_size]["test_count"] += 1
    metrics[chunk_size]["correct_chunk_found_count"] += 1 if metrics_before["correct_chunk_found"] else 0
    metrics[chunk_size]["chunk_count_sum"] += len(selected_chunks)
    metrics[chunk_size]["flagged_chunk_count_sum"] += len(flagged_chunks)
    
    # Update poison detection metrics
    if is_poisoned:
        poison_metrics = metrics[chunk_size]["poison_detection"]
        poison_metrics["total_poisoned_instances"] += 1
        
        if poisoned_chunks_selected:
            poison_metrics["poisoned_chunks_selected"] += len(poisoned_chunks_selected)
            
            # Count correctly flagged poisoned chunks
            correctly_flagged = sum(1 for chunk in poisoned_chunks_selected if chunk in flagged_chunks)
            poison_metrics["poisoned_chunks_flagged"] += correctly_flagged
            poison_metrics["correctly_flagged_as_poisoned"] += correctly_flagged
            
            # Count incorrectly flagged non-poisoned chunks
            incorrectly_flagged = sum(1 for chunk in flagged_chunks if chunk not in poisoned_chunks)
            poison_metrics["incorrectly_flagged_as_poisoned"] += incorrectly_flagged

# Helper function to make objects JSON serializable
def make_json_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [make_json_serializable(x) for x in obj.tolist()]
    elif isinstance(obj, dict):
        return {make_json_serializable(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(x) for x in obj)
    return obj

def save_results(output_path, all_results, overall_metrics, chunk_sizes):
    """Save results to a JSON file."""
    # Format metrics for saving
    metrics_summary = {}
    for chunk_size in chunk_sizes:
        metrics = overall_metrics[chunk_size]
        
        # Calculate summary metrics
        test_count = max(1, metrics["test_count"])
        poison_metrics = metrics["poison_detection"]
        
        # Calculate poison detection metrics
        poison_detection_accuracy = 0.0
        poison_detection_precision = 0.0
        poison_detection_recall = 0.0
        poison_detection_f1 = 0.0
        
        if poison_metrics["poisoned_chunks_selected"] > 0:
            poison_detection_accuracy = float(poison_metrics["poisoned_chunks_flagged"] / poison_metrics["poisoned_chunks_selected"])
        
        if (poison_metrics["correctly_flagged_as_poisoned"] + poison_metrics["incorrectly_flagged_as_poisoned"]) > 0:
            poison_detection_precision = float(poison_metrics["correctly_flagged_as_poisoned"] / (
                poison_metrics["correctly_flagged_as_poisoned"] + poison_metrics["incorrectly_flagged_as_poisoned"]
            ))
        
        if poison_metrics["poisoned_chunks_selected"] > 0:
            poison_detection_recall = float(poison_metrics["correctly_flagged_as_poisoned"] / poison_metrics["poisoned_chunks_selected"])
        
        if (poison_detection_precision + poison_detection_recall) > 0:
            poison_detection_f1 = float(2 * (poison_detection_precision * poison_detection_recall) / (
                poison_detection_precision + poison_detection_recall
            ))
        
        # Create summary for this chunk size
        metrics_summary[f"chunk_size_{chunk_size}"] = {
            "precision_before": float(metrics["precision_sum"] / test_count),
            "recall_before": float(metrics["recall_sum"] / test_count),
            "f1_before": float(metrics["f1_sum"] / test_count),
            "precision_after": float(metrics["precision_after_sum"] / test_count),
            "recall_after": float(metrics["recall_after_sum"] / test_count),
            "f1_after": float(metrics["f1_after_sum"] / test_count),
            "total_correct_chunks_found": int(metrics["correct_chunk_found_count"]),
            "total_tests": int(test_count),
            "avg_chunks_selected": float(metrics["chunk_count_sum"] / test_count),
            "avg_chunks_flagged": float(metrics["flagged_chunk_count_sum"] / test_count),
            "poison_detection": {
                "total_poisoned_instances": int(poison_metrics["total_poisoned_instances"]),
                "poisoned_chunks_selected": int(poison_metrics["poisoned_chunks_selected"]),
                "poisoned_chunks_flagged": int(poison_metrics["poisoned_chunks_flagged"]),
                "accuracy": float(poison_detection_accuracy),
                "precision": float(poison_detection_precision),
                "recall": float(poison_detection_recall),
                "f1": float(poison_detection_f1)
            }
        }
    
    # Prepare output data and ensure it's JSON serializable
    output_data = {
        "results": make_json_serializable(all_results),
        "metrics_summary": make_json_serializable(metrics_summary)
    }
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")

def main():
    # Configuration
    dataset_path = "./Data/poisoned_dataset_contract.json"
    output_path = "perplexity_defence_contract_results.json"
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load models
    logger.info("Loading models...")
    perplexity_model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Model for perplexity calculation
    perplexity_model = AutoModelForCausalLM.from_pretrained(perplexity_model_name, trust_remote_code=True).to(device)
    perplexity_tokenizer = AutoTokenizer.from_pretrained(perplexity_model_name, trust_remote_code=True)
    
    # Load SBERT model for semantic similarity
    sbert_model = SentenceTransformer('Stern5497/sbert-legal-xlm-roberta-base').to(device)
    
    # Load dataset
    logger.info(f"Loading dataset from {dataset_path}")
    data = load_legalbench_rag(dataset_path)
    
    # Different chunk sizes to test
    chunk_sizes = [128, 256, 512]
    
    # Maximum chunks for retrieval
    max_chunks = 5
    
    # Perplexity threshold (can be tuned)
    perplexity_threshold = 20.0
    
    # Initialize metrics tracking
    overall_metrics = initialize_metrics(chunk_sizes)
    
    # Track all results
    all_results = []
    
    # Process each test instance
    for test_idx, test in tqdm(enumerate(data), total=len(data), desc="Processing tests"):
        logger.info(f"Processing test {test_idx+1}/{len(data)}: {test['query']}")
        
        # Process each document in the test
        for doc_idx, document in enumerate(test["snippets"]):
            document_path = os.path.join("Data", document["file_path"])
            
            # Process the ground truth spans for this document
            ground_truth_spans = []
            
            # Check if this is a poisoned document
            is_poisoned = "is_poisoned" in document and document["is_poisoned"] == True
            poisoned_span = None
            
            # Extract spans based on the provided format
            if "span" in document:
                span_start, span_end = document["span"]
                ground_truth_spans.append({"start": span_start, "end": span_end})
            elif "spans" in document:
                for span in document["spans"]:
                    if span.get("label") == "correct":
                        ground_truth_spans.append({"start": span["start"], "end": span["end"]})
            
            # If no spans found but we have original_span field (for poisoned documents)
            if not ground_truth_spans and "original_span" in document:
                span_start, span_end = document["original_span"]
                ground_truth_spans.append({"start": span_start, "end": span_end})
            
            # Capture the poisoned span if it exists
            if is_poisoned and "poisoned_span" in document:
                poisoned_span = {
                    "start": document["poisoned_span"][0],
                    "end": document["poisoned_span"][1]
                }
            
            # Load document content
            doc_content = load_corpus(document_path)
            
            # Test with different chunk sizes
            for chunk_size in chunk_sizes:
                # Chunk the document
                chunks = chunk_text(doc_content, chunk_size)
                
                # Find which chunks contain the ground truth spans
                correct_chunks = []
                for span in ground_truth_spans:
                    span_chunks = find_spanning_chunks(span["start"], span["end"], chunks)
                    correct_chunks.extend(span_chunks)
                
                # Remove duplicates and sort
                correct_chunks = sorted(list(set(correct_chunks)))
                
                # Find which chunks contain the poisoned span
                poisoned_chunks = []
                if is_poisoned and poisoned_span:
                    poisoned_chunks = find_spanning_chunks(poisoned_span["start"], poisoned_span["end"], chunks)
                
                # Retrieve relevant chunks based on the query
                retrieval_result = retrieve_relevant_chunks(test["query"], chunks, sbert_model, max_chunks, device)
                selected_chunks = retrieval_result["selected_chunks"]
                
                # Check if poisoned chunks were selected
                poisoned_chunks_selected = [chunk_idx for chunk_idx in poisoned_chunks if chunk_idx in selected_chunks]
                
                # Calculate metrics before applying perplexity defense
                metrics_before = calculate_precision_recall(selected_chunks, correct_chunks, chunks)
                f1_before = calculate_f1_score(metrics_before["precision"], metrics_before["recall"])
                
                # Apply perplexity-based defense to detect poisoned chunks
                logger.info(f"Applying perplexity-based defense to {len(selected_chunks)} chunks...")
                detection_result = detect_poisoned_chunks(
                    chunks, 
                    selected_chunks, 
                    perplexity_model, 
                    perplexity_tokenizer, 
                    perplexity_threshold, 
                    device
                )
                flagged_chunks = detection_result["flagged_chunks"]
                
                # Calculate metrics after applying defense (using non-flagged chunks)
                non_flagged_chunks = [chunk_idx for chunk_idx in selected_chunks if chunk_idx not in flagged_chunks]
                metrics_after = calculate_precision_recall(non_flagged_chunks, correct_chunks, chunks)
                f1_after = calculate_f1_score(metrics_after["precision"], metrics_after["recall"])
                
                # Update overall metrics
                update_metrics(
                    overall_metrics,
                    chunk_size,
                    metrics_before,
                    metrics_after,
                    f1_before,
                    f1_after,
                    selected_chunks,
                    non_flagged_chunks,
                    flagged_chunks,
                    is_poisoned,
                    poisoned_chunks,
                    poisoned_chunks_selected
                )
                
                # Save result for this test case
                result = {
                    "test_idx": int(test_idx),
                    "query": test["query"],
                    "document_path": document_path,
                    "chunk_size": int(chunk_size),
                    "correct_chunks": [int(c) for c in correct_chunks],
                    "selected_chunks": [int(c) for c in selected_chunks],
                    "flagged_chunks": [int(c) for c in flagged_chunks],
                    "is_poisoned": bool(is_poisoned),
                    "poisoned_chunks": [int(c) for c in poisoned_chunks] if is_poisoned else [],
                    "poisoned_chunks_selected": [int(c) for c in poisoned_chunks_selected] if is_poisoned else [],
                    "perplexity_scores": {int(k): float(v) for k, v in detection_result["perplexity_scores"].items()},
                    "perplexity_threshold": float(detection_result["threshold"]),
                    "metrics_before_defense": {
                        "precision": float(metrics_before["precision"]),
                        "recall": float(metrics_before["recall"]),
                        "f1": float(f1_before),
                        "correct_chunk_found": bool(metrics_before["correct_chunk_found"])
                    },
                    "metrics_after_defense": {
                        "precision": float(metrics_after["precision"]),
                        "recall": float(metrics_after["recall"]),
                        "f1": float(f1_after),
                        "correct_chunk_found": bool(metrics_after["correct_chunk_found"])
                    }
                }
                
                all_results.append(result)
                
                # Print progress stats
                logger.info(f"Document {doc_idx+1}, Chunk size {chunk_size}:")
                logger.info(f"  Before defense: P={metrics_before['precision']:.3f}, R={metrics_before['recall']:.3f}, F1={f1_before:.3f}")
                logger.info(f"  After defense: P={metrics_after['precision']:.3f}, R={metrics_after['recall']:.3f}, F1={f1_after:.3f}")
                logger.info(f"  Total chunks: {len(selected_chunks)}, Flagged: {len(flagged_chunks)}")
                
                # Print poison detection stats if applicable
                if is_poisoned:
                    correct_detection = sum(1 for c in poisoned_chunks_selected if c in flagged_chunks)
                    detection_rate = correct_detection / len(poisoned_chunks_selected) if poisoned_chunks_selected else 0
                    logger.info(f"  Poisoned document detected!")
                    logger.info(f"  Poisoned chunks: {poisoned_chunks}")
                    logger.info(f"  Poisoned chunks selected: {poisoned_chunks_selected} ({len(poisoned_chunks_selected)}/{len(poisoned_chunks)} selected)")
                    logger.info(f"  Poisoned chunks correctly flagged: {correct_detection} ({detection_rate:.3f})")
        
        # Save intermediate results
        save_results(output_path, all_results, overall_metrics, chunk_sizes)
    
    # Final save
    save_results(output_path, all_results, overall_metrics, chunk_sizes)

if __name__ == "__main__":
    main()
