import os
import json
import torch
import numpy as np
import argparse
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

class Contriever:
    def __init__(self, model_name="facebook/contriever", device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device
        self.model.eval()
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def encode(self, sentences, convert_to_tensor=True, batch_size=8):
        all_embeddings = []
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            encoded_input = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            
            embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings)
        
        if len(all_embeddings) > 0:
            all_embeddings = torch.cat(all_embeddings, dim=0)
            
        return all_embeddings

def load_dataset(dataset_path, dataset_type):
    """Load and format different datasets into a common structure."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    formatted_data = []
    
    if dataset_type.lower() == "maud":
        # MAUD format is already handled
        return data["tests"]
    
    elif dataset_type.lower() == "contractnli":
        for item in data:
            formatted_item = {
                "query": item.get("hypothesis", ""),
                "snippets": [{
                    "file_path": item.get("document_name", ""),
                    "span": [item.get("span_start", 0), item.get("span_end", 0)] 
                    if "span_start" in item and "span_end" in item else None
                }]
            }
            formatted_data.append(formatted_item)
    
    elif dataset_type.lower() == "privacy_qa":
        for item in data:
            formatted_item = {
                "query": item.get("question", ""),
                "snippets": [{
                    "file_path": item.get("policy_id", ""),
                    "span": [item.get("segment_start", 0), item.get("segment_end", 0)]
                    if "segment_start" in item and "segment_end" in item else None
                }]
            }
            formatted_data.append(formatted_item)
    
    elif dataset_type.lower() == "cuad":
        for item in data:
            formatted_item = {
                "query": item.get("question", ""),
                "snippets": [{
                    "file_path": item.get("contract_name", ""),
                    "span": [item.get("answer_start", 0), item.get("answer_end", 0)]
                    if "answer_start" in item and "answer_end" in item else None
                }]
            }
            formatted_data.append(formatted_item)
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return formatted_data

def load_corpus(document_path):
    if not os.path.exists(document_path):
        raise FileNotFoundError(f"Document not found at {document_path}")
    
    with open(document_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    return content

def get_text_between_positions(file_path, start, end):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        if start < 0 or end > len(content) or start > end:
            raise IndexError("Invalid range: Ensure 0 ≤ start ≤ end ≤ document length.")
        return content[start:end]
    except FileNotFoundError:
        print(f"The file at {file_path} was not found.")
        return None
    except IndexError as e:
        print(e)
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def chunk_text(text, chunk_size):
    words = text.split()
    chunks = []
    word_positions = []
    
    current_pos = 0
    for word in words:
        word_positions.append(current_pos)
        current_pos += len(word) + 1
    
    overlap = chunk_size // 10
    step_size = chunk_size - overlap
    
    for i in range(0, len(words), step_size):
        end_idx = min(i + chunk_size, len(words))
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
        
        if end_idx >= len(words):
            break
    
    return chunks

def find_spanning_chunks(span_text, span_start, span_end, chunks):
    if span_text is None or span_start is None or span_end is None:
        return []
    
    spanning_chunks = []
    for i, chunk in enumerate(chunks):
        if chunk["start_pos"] <= span_end and chunk["end_pos"] >= span_start:
            spanning_chunks.append(i)
    
    if spanning_chunks:
        min_chunk = min(spanning_chunks)
        max_chunk = max(spanning_chunks)
        expected_chunks = list(range(min_chunk, max_chunk + 1))
        if sorted(spanning_chunks) != expected_chunks:
            print(f"Warning: Non-consecutive chunks detected: {spanning_chunks}")
            spanning_chunks = expected_chunks
    
    return spanning_chunks

def calculate_precision_recall(selected_chunks, correct_chunks):
    if not correct_chunks:
        return {"precision": 0.0, "recall": 0.0, "accuracy": 0.0}
    
    correct_chunks_set = set(correct_chunks)
    relevant_retrieved = len(selected_chunks.intersection(correct_chunks_set))
    
    precision = relevant_retrieved / len(selected_chunks) if selected_chunks else 0.0
    recall = relevant_retrieved / len(correct_chunks_set) if correct_chunks_set else 0.0
    accuracy = 1.0 if relevant_retrieved > 0 else 0.0
    
    return {"precision": precision, "recall": recall, "accuracy": accuracy}

def retrieve_with_sbert(query, chunks, model, k=5, device='cpu'):
    chunk_texts = [chunk["text"] for chunk in chunks]

    query_embedding = model.encode(query, convert_to_tensor=True, device=device)
    chunk_embeddings = model.encode(chunk_texts, convert_to_tensor=True, device=device, show_progress_bar=False)

    cosine_scores = util.cos_sim(query_embedding, chunk_embeddings)[0]
    top_k = min(k, len(chunks))
    top_k_indices = torch.topk(cosine_scores, k=top_k).indices.cpu().numpy()

    return set(top_k_indices.tolist())

def retrieve_with_contriever(query, chunks, model, k=5):
    chunk_texts = [chunk["text"] for chunk in chunks]

    query_embedding = model.encode([query], convert_to_tensor=True)
    chunk_embeddings = model.encode(chunk_texts, convert_to_tensor=True)

    cosine_scores = util.cos_sim(query_embedding, chunk_embeddings)[0]
    top_k = min(k, len(chunks))
    top_k_indices = torch.topk(cosine_scores, k=top_k).indices.cpu().numpy()

    return set(top_k_indices.tolist())

def retrieve_with_cross_encoder(query, chunks, model, k=5):
    chunk_texts = [chunk["text"] for chunk in chunks]
    
    # Prepare sentence pairs for cross-encoder
    sentence_pairs = [[query, text] for text in chunk_texts]
    
    # Score the pairs
    scores = model.predict(sentence_pairs)
    
    # Get top-k indices based on scores
    top_k = min(k, len(chunks))
    top_k_indices = np.argsort(scores)[-top_k:].tolist()
    
    return set(top_k_indices)

def load_model(model_type, model_name, device):
    """Load the appropriate model based on the model_type."""
    print(f"Loading {model_type} model: {model_name}")
    if model_type.lower() == "sbert":
        return SentenceTransformer(model_name).to(device)
    elif model_type.lower() == "cross-encoder":
        return CrossEncoder(model_name, device=device)
    elif model_type.lower() == "contriever":
        return Contriever(model_name, device=device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_default_model_name(model_type):
    """Get the default model name for the given model type."""
    if model_type.lower() == "sbert":
        return 'Stern5497/sbert-legal-xlm-roberta-base'
    elif model_type.lower() == "cross-encoder":
        return 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    elif model_type.lower() == "contriever":
        return 'facebook/contriever'
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def main():
    parser = argparse.ArgumentParser(description='Legal RAG Evaluation with Different Models and Datasets')
    parser.add_argument('--dataset_path', required=True, help='Path to the dataset JSON file')
    parser.add_argument('--dataset_type', required=True, choices=['maud', 'contractnli', 'privacy_qa', 'cuad'], 
                        help='Type of dataset')
    parser.add_argument('--model_type', required=True, choices=['sbert', 'cross-encoder', 'contriever'], 
                        help='Type of embedding model to use')
    parser.add_argument('--model_name', help='Name or path of the model')
    parser.add_argument('--data_dir', default='./Data', help='Directory where documents are stored')
    parser.add_argument('--chunk_sizes', type=int, nargs='+', default=[128, 256, 512], 
                        help='Chunk sizes to evaluate')
    parser.add_argument('--k_values', type=int, nargs='+', default=[1, 2, 3, 4, 5, 8, 16, 32, 64], 
                        help='k values (number of retrieved chunks) to evaluate')
    
    args = parser.parse_args()
    
    # Set model name if not provided
    if not args.model_name:
        args.model_name = get_default_model_name(args.model_type)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load appropriate model
    model = load_model(args.model_type, args.model_name, device)
    
    # Load dataset
    data = load_dataset(args.dataset_path, args.dataset_type)
    print(f"Loaded {len(data)} instances from {args.dataset_type} dataset")
    
    chunk_sizes = args.chunk_sizes
    k_values = args.k_values

    overall_metrics = {
        chunk_size: {
            k: {"precision_sum": 0.0, "recall_sum": 0.0, "accuracy_sum": 0.0, "count": 0}
            for k in k_values
        } for chunk_size in chunk_sizes
    }

    all_results = []
    output_path = f"{args.model_type}_{args.dataset_type}_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"results": [], "overall_metrics": {}}, f, indent=4)

    for i, sample in enumerate(data):
        print(f"\nProcessing sample {i+1}/{len(data)}")
        query = sample["query"]
        print(f"Query: {query}")
        
        doc_info = sample["snippets"][0]
        # Handle different path formats in different datasets
        doc_path = doc_info["file_path"]
        if not os.path.exists(doc_path) and not os.path.isabs(doc_path):
            doc_path = os.path.join(args.data_dir, doc_info["file_path"])
        
        span_text, span_start, span_end = None, None, None
        if "span" in doc_info and isinstance(doc_info["span"], list) and len(doc_info["span"]) == 2:
            span_start = doc_info["span"][0]
            span_end = doc_info["span"][1]
            span_text = get_text_between_positions(doc_path, span_start, span_end)
        
        try:
            corpus_text = load_corpus(doc_path)
        except FileNotFoundError:
            print(f"Document not found at {doc_path}, skipping sample")
            continue
            
        sample_results = []

        for chunk_size in chunk_sizes:
            print(f"\nUsing chunk size: {chunk_size}")
            chunks = chunk_text(corpus_text, chunk_size)
            print(f"Document chunked into {len(chunks)} chunks")
            
            correct_chunk_numbers = find_spanning_chunks(span_text, span_start, span_end, chunks)
            print(f"Correct answer spans chunks: {correct_chunk_numbers}")
            
            span_containing_chunks = []
            for chunk_idx in correct_chunk_numbers:
                if 0 <= chunk_idx < len(chunks):
                    span_containing_chunks.append({
                        "chunk_number": chunk_idx,
                        "text": chunks[chunk_idx]["text"]
                    })
            
            chunk_size_results = {
                "chunk_size": chunk_size,
                "total_chunks": len(chunks),
                "span_containing_chunks": span_containing_chunks,
                "k_values_results": {}
            }

            for k in k_values:
                print(f"Testing with k={k}")
                
                # Use the appropriate retrieval method based on model type
                if args.model_type.lower() == "sbert":
                    selected_chunks = retrieve_with_sbert(query, chunks, model, k=k, device=device)
                elif args.model_type.lower() == "cross-encoder":
                    selected_chunks = retrieve_with_cross_encoder(query, chunks, model, k=k)
                elif args.model_type.lower() == "contriever":
                    selected_chunks = retrieve_with_contriever(query, chunks, model, k=k)
                
                print(f"Selected chunks (k={k}): {selected_chunks}")
                
                metrics = calculate_precision_recall(selected_chunks, set(correct_chunk_numbers))
                print(f"k={k} - Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
                
                if correct_chunk_numbers:
                    overall_metrics[chunk_size][k]["precision_sum"] += metrics["precision"]
                    overall_metrics[chunk_size][k]["recall_sum"] += metrics["recall"]
                    overall_metrics[chunk_size][k]["accuracy_sum"] += metrics["accuracy"]
                    overall_metrics[chunk_size][k]["count"] += 1
                
                chunk_size_results["k_values_results"][k] = {
                    "selected_chunks": list(selected_chunks),
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "accuracy": metrics["accuracy"],
                    "correct_chunks": correct_chunk_numbers
                }
            
            sample_results.append(chunk_size_results)

        sample_result = {
            "sample_index": i,
            "query": query,
            "span_text": span_text,
            "results_by_chunk_size": sample_results
        }
        all_results.append(sample_result)

        try:
            with open(output_path, "r", encoding="utf-8") as f:
                current_data = json.load(f)

            current_data["results"].append(sample_result)
            current_averages = {}

            for chunk_size in chunk_sizes:
                current_averages[str(chunk_size)] = {}
                for k in k_values:
                    count = overall_metrics[chunk_size][k]["count"]
                    if count > 0:
                        current_averages[str(chunk_size)][str(k)] = {
                            "avg_precision": overall_metrics[chunk_size][k]["precision_sum"] / count,
                            "avg_recall": overall_metrics[chunk_size][k]["recall_sum"] / count,
                            "avg_accuracy": overall_metrics[chunk_size][k]["accuracy_sum"] / count,
                            "sample_count": count
                        }

            current_data["overall_metrics"] = current_averages
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(current_data, f, indent=4)

            print(f"Updated results in {output_path} after sample {i+1}")
        except Exception as e:
            print(f"Error updating JSON file: {e}")

    final_output = {"results": all_results, "overall_metrics": {}}
    for chunk_size in chunk_sizes:
        final_output["overall_metrics"][str(chunk_size)] = {}
        for k in k_values:
            count = overall_metrics[chunk_size][k]["count"]
            if count > 0:
                final_output["overall_metrics"][str(chunk_size)][str(k)] = {
                    "avg_precision": overall_metrics[chunk_size][k]["precision_sum"] / count,
                    "avg_recall": overall_metrics[chunk_size][k]["recall_sum"] / count,
                    "avg_accuracy": overall_metrics[chunk_size][k]["accuracy_sum"] / count,
                    "sample_count": count
                }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4)

    print(f"Final results saved to {output_path}")
    print(f"Total instances processed: {len(data)}")
    print(f"\nSummary of results:")
    for chunk_size in chunk_sizes:
        for k in k_values:
            count = overall_metrics[chunk_size][k]["count"]
            if count > 0:
                avg_precision = overall_metrics[chunk_size][k]["precision_sum"] / count
                avg_recall = overall_metrics[chunk_size][k]["recall_sum"] / count
                avg_accuracy = overall_metrics[chunk_size][k]["accuracy_sum"] / count
                print(f"Chunk size: {chunk_size}, k={k} - Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, Accuracy: {avg_accuracy:.4f}")

if __name__ == "__main__":
    main()
