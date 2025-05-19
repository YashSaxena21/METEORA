import os
import json
import torch
import numpy as np
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import copy
from datasets import load_dataset
import tempfile

def load_legalbench_rag(dataset_path):
    """
    Load the LegalBench RAG benchmark dataset from a JSON file.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data["tests"]  # Extracting the list of test cases

def save_document_to_temp_file(document_text):
    temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8', suffix='.txt')
    temp_file.write(document_text)
    temp_file.close()
    return temp_file.name

def prepare_legalbench_dataset():
    # Similar structure to prepare_finqa_dataset but for legal documents
    legalbench_dataset = load_legalbench_rag("./Data/privacy_qa.json")
    return legalbench_dataset

def generate_rationales_batch(model, tokenizer, queries, device, batch_size=4):
    results = []
    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i:i+batch_size]
        batch_responses = []
        for query in batch_queries:
            few_shot_examples = [
                {
                    "query": "Consider the Promotion Agreement between MiddleBrook Pharmaceuticals, Inc. and DoctorDirectory.com, Inc. for MOXATAG; Is there an anti-assignment clause in this contract?",
                    "response": """
        <rationale_1> [Precise semantic search strategy] Look for provisions that restrict either party's ability to transfer or assign their contractual obligations or rights without permission. </rationale_1>
        <rationale_2> [Alternative information extraction approach] Identify any clauses discussing "assignment," "assigns," or "successors," particularly those conditioned on written consent or restrictions on third-party transfers. </rationale_2>
        <rationale_3> [Legal keyword anchoring] Search for typical anti-assignment clause language such as "shall not assign," "without prior written consent," or "consent shall not be unreasonably withheld." </rationale_3>
        <rationale_4> [Contextual clause grouping] Locate sections typically placed near "Governing Law," "Term and Termination," or "Miscellaneous" — which is where assignment clauses are often found. </rationale_4>
        <rationale_5> [Contractual continuity trigger] Scan for clauses that specify the agreement binds "successors and assigns," followed by conditions that limit or prohibit assignment — a common anti-assignment signal. </rationale_5>
        <rationale_6> [Transactional control concern] Detect any stipulations that require one party to obtain consent before assigning the agreement, which reflects concerns over change in contractual control. </rationale_6>
        <rationale_7> [Consent-based transfer filter] Search for clauses that allow assignment but only with prior written approval — suggesting a partial anti-assignment clause that limits unauthorized transfers. </rationale_7>
        <rationale_8> [Risk allocation rationale] Look for language that protects either party from being bound to unknown or unintended assignees, which typically motivates inclusion of anti-assignment provisions. </rationale_8>
        <rationale_9> [Contractual integrity preservation] Find sections aiming to preserve the contractual relationship's original structure by limiting reassignment of duties or benefits to other entities. </rationale_9>
        <rationale_10> [Standard legal boilerplate detection] Identify standardized legal text often used in agreements under "Assignment/Change of Control" sections, which include assignment restrictions. </rationale_10>
        """
                },
                {
                    "query": "Consider the Cooperation Agreement between Beike Internet Security Technology Co., Ltd. and Baidu Online Network Technology (Beijing) Co., Ltd. for Internet Search Services; What licenses are granted under this contract?",
                    "response": """
        <rationale_1> [Usage restriction identification] Search for clauses that place limits or conditions on how Party A can use information or technical resources provided by Party B — these usually signal what is and isn't licensed. </rationale_1>
        <rationale_2> [Licensing boundary detection] Identify provisions that explicitly prohibit repurposing or commercialization of shared data or tools — such language suggests restrictions on granted licenses. </rationale_2>
        <rationale_3> [Reverse inference approach] Instead of looking for "grants," locate text that implies **what is *****not***** licensed**, such as prohibitions on commercial use, modification, or redistribution. </rationale_3>
        <rationale_4> [Function and asset usage clause trigger] Scan for any language referencing the use of Party B's "functions" or "information" — commonly discussed in license terms or data usage policies. </rationale_4>
        <rationale_5> [Commercial usage restriction filter] Search for statements that deny Party A the right to use provided services or content for commercial activities — a strong indicator of limited license rights. </rationale_5>
        <rationale_6> [Implied IP license control] Identify text where Party B controls or restricts how Party A can use Party B's deliverables, especially if it refers to search services, data, or embedded tools. </rationale_6>
        <rationale_7> [Clause overlap with licensing scope] Locate clauses where Party A is limited to specific use cases (e.g., internal integration) and explicitly excluded from others (e.g., resale, public offering). </rationale_7>
        <rationale_8> [Service limitation via non-commercial clause] Spot language that forbids commercial application of provided services — such statements clarify that Party A has a non-commercial license only. </rationale_8>
        <rationale_9> [Embedded content restriction detection] Search for sections that forbid changes, redistribution, or unauthorized applications of embedded tools, which often signals a narrow license grant. </rationale_9>
        <rationale_10> [Keyword pattern for denial of rights] Look for combinations like "shall not use," "without permission," or "not authorized" within sentences describing Party A's use of Party B's technology or information. </rationale_10>
        """
                },
                {
                    "query": "Consider the Hosting and Management Agreement between HealthGate Data Corp., Blackwell Science Limited, and Munksgaard A/S; What happens in the event of a change of control of one of the parties in this contract?",
                    "response": """
        <rationale_1> [Ownership transition trigger] Look for clauses that specify what rights or actions are triggered when there's a **change in control** or **ownership** of a contracting party. </rationale_1>
        <rationale_2> [Change of control as termination event] Search for conditions under which one party may terminate the agreement if the other party undergoes a **change in corporate control or ownership structure**. </rationale_2>
        <rationale_3> [Clause with unilateral rights upon acquisition] Find language granting one party **unilateral discretion** to terminate the contract if there's a merger, acquisition, or controlling stake change in the other party. </rationale_3>
        <rationale_4> [Risk mitigation for partner instability] Locate provisions that protect one party from instability or loss of control in the other — e.g., through **early termination rights upon control shifts**. </rationale_4>
        <rationale_5> [Corporate structure sensitivity clauses] Identify contract sections that mention changes in **holding companies**, **shareholders**, or **company control**, especially linked to potential liability shifts. </rationale_5>
        <rationale_6> [Control-based early exit option] Target clauses that allow a party to exit the agreement without financial penalty if the other party experiences a **change of control**, especially without obligation to justify losses. </rationale_6>
        <rationale_7> [Anti-assignment and control overlap] Scan for clauses that are similar in structure to **anti-assignment provisions**, but that deal specifically with **control changes** as triggers for termination rights. </rationale_7>
        <rationale_8> [Protective clause for business continuity] Search for statements enabling a party to safeguard their business interest if their partner's **ownership or management materially changes**. </rationale_8>
        <rationale_9> [Standard "change in control" definitions] Search for boilerplate definitions of "control" — e.g., holding >50% of voting rights — as these often appear near relevant enforcement clauses. </rationale_9>
        <rationale_10> [Clause containing "at their own option" or "may terminate"] Spot clauses using language like "may terminate at their discretion" in connection with events such as mergers, acquisitions, or control transfers. </rationale_10>
        """
                }
            ]

            # Construct the few-shot prompt with a general instruction and multiple examples
            prompt = """
            Instructions for Generating Precise Semantic Legal Rationales:

            Your task is to generate a series of semantic search strategies for extracting specific legal information from contract documents. Each rationale must:
            - Capture a UNIQUE semantic search dimension
            - Be concrete and specifically tailored to the query
            - Focus on extracting precise, targeted legal information
            - Use concise, strategically worded approaches

            Format Requirements:
            - Use XML-style tags: <rationale_1>, <rationale_2>, etc.
            - Include a brief descriptive label in square brackets
            - Provide a precise, strategic search approach

            Example 1:
            Query: Consider the Promotion Agreement between MiddleBrook Pharmaceuticals, Inc. and DoctorDirectory.com, Inc. for MOXATAG; Is there an anti-assignment clause in this contract?
            Response:
        {example1}

            Example 2:
            Query: Consider the Cooperation Agreement between Beike Internet Security Technology Co., Ltd. and Baidu Online Network Technology (Beijing) Co., Ltd. for Internet Search Services; What licenses are granted under this contract?
            Response:
        {example2}

            Example 3:
            Query: Consider the Hosting and Management Agreement between HealthGate Data Corp., Blackwell Science Limited, and Munksgaard A/S; What happens in the event of a change of control of one of the parties in this contract?
            Response:
        {example3}

            Your Task:
            Query: {query}
            Generate a set of nuanced, strategically crafted rationales to guide semantic search for the specific legal question.
            """.format(
                example1=few_shot_examples[0]["response"],
                example2=few_shot_examples[1]["response"],
                example3=few_shot_examples[2]["response"],
                query=query
            )
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=512,  # Increased to accommodate more rationales
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7  # Slight randomness to encourage diversity
                )
            
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            batch_responses.append(response)
        
        results.extend(batch_responses)
    return results


def extract_rationales_with_regex(response):
    """
    Extract rationales that occur after the last Query: statement in the text.
    Handles multiple formats of rationale extraction.
    Returns a list of tuples: (rationale_number, rationale_text)
    """
    # Step 1: Get everything after the last 'Query:' block
    last_query_split = re.split(r'\n?Query:.*', response)
    if len(last_query_split) < 2:
        print("⚠️ No query found.")
        return []

    # Step 2: Extract the last block (after the last query)
    final_block = last_query_split[-1]

    # Define multiple regex patterns to capture different rationale formats
    extraction_patterns = [
        # XML-style rationales
        r'<rationale_(\d+)>\s*(?:\[[^\]]+\])?\s*(.*?)\s*</rationale_\1>',
        
        # Plain text rationales with XML-like tags (first block matching this pattern)
        r'First Rationale:\s*(?:\[[^\]]+\])?\s*(.*?)(?=\n\n)',
        
        # Numbered rationales with XML-like tags
        r'<rationale_(\d+)>\s*(?:\[[^\]]+\])?\s*(.*?)(?=\n\n|$)',
        
        # Plain text numbered rationales
        r'Rationale\s+(\d+):\s*(?:\[[^\]]+\])?\s*(.*?)(?=\n\n|$)'
    ]

    # Attempt to extract using each pattern
    matches = []
    for pattern in extraction_patterns:
        current_matches = re.findall(pattern, final_block, re.DOTALL | re.IGNORECASE)
        if current_matches:
            matches = current_matches
            break

    # Clean up matches
    cleaned_rationales = []
    for match in matches:
        # Handle different match formats (with or without explicit numbers)
        if isinstance(match, tuple):
            # If match has a tuple with number and text
            if len(match) == 2:
                num = match[0] if match[0] else len(cleaned_rationales) + 1
                text = match[1].strip()
            else:
                # Fallback if unexpected match format
                continue
        else:
            # For matches without explicit numbers (like First Rationale)
            num = 1
            text = match.strip()

        # Remove any leading/trailing whitespace and XML-like tags
        text = re.sub(r'^(\[[^\]]+\])\s*', '', text).strip()
        
        cleaned_rationales.append((int(num), text))

    # Sort rationales by number
    cleaned_rationales.sort(key=lambda x: x[0])

    if cleaned_rationales:
        print(f"✅ Extracted {len(cleaned_rationales)} rationale(s) after final query.")
    else:
        print("⚠️ No rationales found after final query.")

    return cleaned_rationales

def improved_retrieval(rationales, chunks, sbert_model, max_chunks=5):
    """
    Hyperparameter-free chunk retrieval strategy with:
    1. Each rationale selects a distinct chunk (no duplicates)
    2. Data-driven pooled embedding with statistical elbow detection
    3. Simple position-based sliding window of size 1
    """
    chunk_votes = {}
    chunk_similarity_scores = {}
    rationale_contributions = {}  # Track which rationales contributed to finding which chunks
    already_selected = set()  # Track chunks that have been selected by previous rationales

    chunk_texts = [chunk["text"] for chunk in chunks]
    chunk_embeddings = sbert_model.encode(chunk_texts, convert_to_tensor=True)

    selection_reasons = {
        "Pairing": set(),
        "Pooling": set(),
        "Expansion": set()
    }

    # Rationale-based voting with distinct chunk selection
    for rationale_num, rationale_text in rationales:
        try:
            rationale_embedding = sbert_model.encode(rationale_text, convert_to_tensor=True)
            similarity_scores = util.cos_sim(rationale_embedding, chunk_embeddings)[0]
            
            # Create a masked version of similarity scores to ignore already selected chunks
            masked_scores = similarity_scores.clone()
            for idx in already_selected:
                masked_scores[idx] = -1.0  # Set already selected chunks to lowest possible score
            
            # Get the most similar chunk that hasn't been selected yet
            top_index = torch.argmax(masked_scores).item()
            score = similarity_scores[top_index].item()
            
            if top_index not in chunk_votes:
                chunk_votes[top_index] = 0
                chunk_similarity_scores[top_index] = 0
                rationale_contributions[top_index] = []

            chunk_votes[top_index] += 1
            chunk_similarity_scores[top_index] = max(chunk_similarity_scores[top_index], score)
            selection_reasons["rationale_voting"].add(top_index)
            already_selected.add(top_index)  # Mark this chunk as selected
            
            # Track which rationale contributed to finding this chunk
            rationale_contributions[top_index].append(rationale_num)

        except Exception as e:
            print(f"Error processing rationale {rationale_num}: {e}")

    # Pooled Embedding with Statistical Elbow Detection
    pooled_selected_chunks = []
    elbow_idx = 0
    if rationales:  # Check if there are any rationales
        rationale_texts = [rat_text for _, rat_text in rationales]
        rationale_embeddings = [sbert_model.encode(rat_text, convert_to_tensor=True) for rat_text in rationale_texts]

        pooled_embedding = torch.mean(torch.stack(rationale_embeddings), dim=0)
        pooled_similarity_scores = util.cos_sim(pooled_embedding, chunk_embeddings)[0]

        # Sort similarities for elbow detection
        chunk_similarities = [(idx, score.item()) for idx, score in enumerate(pooled_similarity_scores)]
        chunk_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Statistical elbow detection without preset hyperparameters
        if len(chunk_similarities) > 2:  # Need at least 3 points for meaningful analysis
            scores = [score for _, score in chunk_similarities]
            diffs = [scores[i] - scores[i+1] for i in range(len(scores)-1)]
            
            if len(diffs) > 0:
                # Calculate statistical properties of the differences
                mean_diff = sum(diffs) / len(diffs)
                variance = sum((d - mean_diff) ** 2 for d in diffs) / len(diffs)
                std_dev = variance ** 0.5  # Standard deviation
                
                # Detect outliers in differences (statistically significant drops)
                # Using z-scores to identify significant drops
                z_scores = [(diff - mean_diff) / std_dev if std_dev > 0 else 0 for diff in diffs]
                
                # Find first point where z-score exceeds 1.0 (1 standard deviation)
                # This is a statistically significant drop without using preset thresholds
                significant_drops = [i for i, z in enumerate(z_scores) if z > 1.0]
                
                if significant_drops:
                    elbow_idx = significant_drops[0]  # First significant drop
                else:
                    # Fallback: if no clear statistical pattern, use rate of change method
                    # Find where the second derivative is maximized (curvature)
                    second_diffs = [diffs[i+1] - diffs[i] for i in range(len(diffs)-1)]
                    if second_diffs:
                        elbow_idx = second_diffs.index(max(second_diffs))
                    else:
                        elbow_idx = min(len(scores) // 3, max_chunks)  # Fallback
            else:
                elbow_idx = 0
                
            # Print elbow detection details
            print(f"Statistical elbow detection found elbow at position {elbow_idx} with score {scores[elbow_idx]}")
            print(f"Mean difference: {mean_diff}, Standard deviation: {std_dev if 'std_dev' in locals() else 'N/A'}")
            if 'z_scores' in locals() and len(z_scores) > 0:
                print(f"Z-scores around elbow: {z_scores[max(0, elbow_idx-2):min(len(z_scores), elbow_idx+3)]}")
            
            # Select chunks up to the elbow point
            pooled_selected_chunks = [idx for idx, _ in chunk_similarities[:elbow_idx+1]]
        else:
            # If we have very few points, just take the highest ones
            pooled_selected_chunks = [idx for idx, _ in chunk_similarities[:min(len(chunk_similarities), max_chunks)]]
            
        selection_reasons["pooled_embedding"] = set(pooled_selected_chunks)

        combined_selected_chunks = set(chunk_votes.keys()).union(pooled_selected_chunks)
    else:
        # Fallback if no rationales were provided
        chunk_similarities = []
        combined_selected_chunks = set()

    # Simple position-based sliding window of size 1
    sliding_window_chunks = set()
    for chunk_idx in combined_selected_chunks:
        # Add chunks immediately before and after (window size of 1)
        if chunk_idx - 1 >= 0:
            sliding_window_chunks.add(chunk_idx - 1)
            selection_reasons["sliding_window"].add(chunk_idx - 1)
            
        if chunk_idx + 1 < len(chunks):
            sliding_window_chunks.add(chunk_idx + 1)
            selection_reasons["sliding_window"].add(chunk_idx + 1)

    final_selected_chunks = combined_selected_chunks.union(sliding_window_chunks)
    sorted_final_chunks = sorted(final_selected_chunks)

    selection_details = {
        "chunk_votes": chunk_votes,
        "chunk_similarity_scores": chunk_similarity_scores,
        "selection_reasons": selection_reasons,
        "rationale_contributions": rationale_contributions,
        "elbow_index": elbow_idx  # Include the elbow index in the details
    }

    return {
        "Pairing": sorted(selection_reasons["rationale_voting"]),
        "Pairing_and_pooling": sorted(combined_selected_chunks),
        "all_stages": sorted_final_chunks
    }

def calculate_metrics(selected_chunks, correct_chunks):
    selected_set = set(selected_chunks)
    correct_set = set(correct_chunks)
    
    overlap = len(selected_set.intersection(correct_set))
    precision = overlap / len(selected_set) if selected_set else 0.0
    recall = overlap / len(correct_set) if correct_set else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correct_found": overlap > 0
    }

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
    """
    Find all chunks that contain any part of the span text based on character positions.
    Returns a list of chunk indices that contain the span.
    """
    if span_text is None or span_start is None or span_end is None:
        return []
    
    spanning_chunks = []
    
    for i, chunk in enumerate(chunks):
        # Check if there's any overlap between the span and the chunk
        if (chunk["start_pos"] <= span_end and chunk["end_pos"] >= span_start):
            spanning_chunks.append(i)
    
    return spanning_chunks

def process_batch(batch_instances, model, tokenizer, sbert_model, device, chunk_sizes):
    batch_queries = [instance["query"] for instance in batch_instances]
    batch_responses = generate_rationales_batch(model, tokenizer, batch_queries, device)
    
    batch_results = []
    
    for i, (instance, response) in enumerate(zip(batch_instances, batch_responses)):
        query = instance["query"]
        rationales = extract_rationales_with_regex(response)
        
        if not rationales:
            continue
        
        # Get document path and span information
        doc_info = instance["snippets"][0]
        doc_path = os.path.join("Data", doc_info["file_path"])
        
        # Extract span text if span information is available
        span_text = None
        span_start = None
        span_end = None
        if "span" in doc_info and isinstance(doc_info["span"], list) and len(doc_info["span"]) == 2:
            span_start = doc_info["span"][0]
            span_end = doc_info["span"][1]
            with open(doc_path, "r", encoding="utf-8") as f:
                corpus_text = f.read()
                span_text = corpus_text[span_start:span_end]
        
        instance_results = []
        
        for chunk_size in chunk_sizes:
            chunks = chunk_text(corpus_text, chunk_size)
            correct_chunk_numbers = find_spanning_chunks(span_text, span_start, span_end, chunks)
            
            if not correct_chunk_numbers:
                continue
            
            retrieval_results = improved_retrieval(rationales, chunks, sbert_model)
            
            stage_metrics = {}
            for stage_name, selected_chunks in retrieval_results.items():
                stage_metrics[stage_name] = calculate_metrics(selected_chunks, correct_chunk_numbers)
            
            result = {
                "query": query,
                "chunk_size": chunk_size,
                "total_chunks": len(chunks),
                "gold_chunks": correct_chunk_numbers,
                "metrics": stage_metrics,
                "selections": retrieval_results
            }
            
            instance_results.append(result)
        
        batch_results.append(instance_results)
    
    return batch_results

def main():
    output_path = "abalation_real_privacy.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    llm_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct").to(device)
    llm_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    sbert_model = SentenceTransformer('Stern5497/sbert-legal-xlm-roberta-base').to(device)
    
    dataset = prepare_legalbench_dataset()
    chunk_sizes = [128, 256, 512]
    batch_size = 5
    
    all_results = []
    
    # Fix: Initialize the overall_metrics structure properly with all required nested keys
    overall_metrics = {}
    for size in chunk_sizes:
        overall_metrics[size] = {}
        for stage in ["Pairing", "Pairing_and_pooling", "all_stages"]:
            overall_metrics[size][stage] = {
                "precision_sum": 0, 
                "recall_sum": 0, 
                "f1_sum": 0, 
                "count": 0,
                "correct_count": 0
            }
    
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        batch_results = process_batch(batch, llm_model, llm_tokenizer, sbert_model, device, chunk_sizes)
        
        for instance_results in batch_results:
            all_results.extend(instance_results)
            
            for result in instance_results:
                chunk_size = result["chunk_size"]
                for stage, metrics in result["metrics"].items():
                    # Ensure the stage exists in overall_metrics
                    if stage not in overall_metrics[chunk_size]:
                        overall_metrics[chunk_size][stage] = {
                            "precision_sum": 0, 
                            "recall_sum": 0, 
                            "f1_sum": 0, 
                            "count": 0,
                            "correct_count": 0
                        }
                    
                    overall_metrics[chunk_size][stage]["precision_sum"] += metrics["precision"]
                    overall_metrics[chunk_size][stage]["recall_sum"] += metrics["recall"]
                    overall_metrics[chunk_size][stage]["f1_sum"] += metrics["f1"]
                    overall_metrics[chunk_size][stage]["count"] += 1
                    if metrics["correct_found"]:
                        overall_metrics[chunk_size][stage]["correct_count"] += 1
        
        # Save intermediate results
        with open(output_path, "w") as f:
            avg_metrics = {}
            for size in chunk_sizes:
                avg_metrics[size] = {}
                for stage in overall_metrics[size]:
                    metrics = overall_metrics[size][stage]
                    if metrics["count"] > 0:
                        avg_metrics[size][stage] = {
                            "avg_precision": metrics["precision_sum"] / metrics["count"],
                            "avg_recall": metrics["recall_sum"] / metrics["count"],
                            "avg_f1": metrics["f1_sum"] / metrics["count"],
                            "correct_percentage": (metrics["correct_count"] / metrics["count"]) * 100
                        }
            
            json.dump({"results": all_results, "avg_metrics": avg_metrics}, f, indent=2)

if __name__ == "__main__":
    main()
