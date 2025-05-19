import re
import os
import json
import torch
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from sentence_transformers import SentenceTransformer, util
from datasets import Dataset
from trl import DPOTrainer, DPOConfig

# =======================
# USER CONFIGURATION
# =======================
DATASETS = {
    "contractnli": "/path/to/contractnli.json",
    "cuad": "/path/to/cuad.json",
    "maud": "/path/to/maud.json",
    "privacyqa": "/path/to/privacyqa.json",
}
REFINER_MODEL_NAME = "llama-3.1-8b-instruct"
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
EPOCHS = 3
MAX_CHUNKS = 5

# =======================
# DPO DATA PREPARATION
# =======================
def prepare_dpo_dataset_from_splits(train_examples, val_examples, tokenizer):
    train_chosen, train_rejected = [], []
    for ex in train_examples:
        if not isinstance(ex.get("effective_rationales"), list) or not isinstance(ex.get("ineffective_rationales"), list):
            continue
        if not ex["effective_rationales"] or not ex["ineffective_rationales"]:
            continue
        query = ex.get("query", "")
        if not query:
            continue
        eff = [str(r) for r in ex["effective_rationales"]]
        ine = [str(r) for r in ex["ineffective_rationales"]]
        train_chosen.append({"input": f"Query: {query}\n\nProvide effective rationales for retrieving relevant legal information:\n", "chosen": ", ".join(eff), "rejected": ", ".join(ine)})
        train_rejected.append({"input": f"Query: {query}\n\nProvide effective rationales for retrieving relevant legal information:\n", "chosen": ", ".join(eff), "rejected": ", ".join(ine)})
    val_chosen, val_rejected = [], []
    for ex in val_examples:
        if not isinstance(ex.get("effective_rationales"), list) or not isinstance(ex.get("ineffective_rationales"), list):
            continue
        if not ex["effective_rationales"] or not ex["ineffective_rationales"]:
            continue
        query = ex.get("query", "")
        if not query:
            continue
        eff = [str(r) for r in ex["effective_rationales"]]
        ine = [str(r) for r in ex["ineffective_rationales"]]
        val_chosen.append({"input": f"Query: {query}\n\nProvide effective rationales for retrieving relevant legal information:\n", "chosen": ", ".join(eff), "rejected": ", ".join(ine)})
        val_rejected.append({"input": f"Query: {query}\n\nProvide effective rationales for retrieving relevant legal information:\n", "chosen": ", ".join(eff), "rejected": ", ".join(ine)})
    if not val_chosen and len(train_chosen) > 10:
        val_chosen = train_chosen[-10:]
        val_rejected = train_rejected[-10:]
        train_chosen = train_chosen[:-10]
        train_rejected = train_rejected[:-10]
    if not train_chosen or not val_chosen:
        return None
    train_ds = Dataset.from_dict({
        "input": [x["input"] for x in train_chosen],
        "chosen": [x["chosen"] for x in train_chosen],
        "rejected": [x["rejected"] for x in train_rejected]
    })
    val_ds = Dataset.from_dict({
        "input": [x["input"] for x in val_chosen],
        "chosen": [x["chosen"] for x in val_chosen],
        "rejected": [x["rejected"] for x in val_rejected]
    })
    return {"train": train_ds, "validation": val_ds}

# =======================
# FEW-SHOT PROMPT & RATIONALE GENERATION
# =======================
def extract_rationales_with_regex(response):
    last_block = response.split("Query:")[-1]
    matches = re.findall(r'<rationale_(\d+)>\s*(.*?)\s*</rationale_\1>', last_block, re.DOTALL)
    return sorted([(int(num), text.strip()) for num, text in matches], key=lambda x: x[0])

def generate_rationales(model, tokenizer, query, device):
    """
    Generate rationales using Equall/Saul-7B-Instruct-v1 with few-shot prompt.
    """
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
    - Generate 8-10 distinct rationales
    - Avoid redundancy between rationales

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
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            # Encode the prompt and generate
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Set a reasonable max_new_tokens to prevent infinite generation
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=0.7,  # Lower temperature for more focused generations
                    max_new_tokens=512,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode the generated text
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the part after the prompt
            response_text = generated_text[len(prompt.strip()):]
            print(f"Generated response: {generated_text}")
            
            # Handle empty responses
            if not response_text.strip():
                print(f"Attempt {attempt+1}: Empty response, retrying...")
                continue
                
            return generated_text

def extract_rationales_with_regex(response):
    """
    Extract rationales that occur after the last Query: statement in the text.
    Handles multiple formats of rationale extraction including numbered list formats
    with bracketed headers or descriptors.
    Returns a list of tuples: (rationale_number, rationale_text)
    """
    # Step 1: Get everything after the last 'Query:' block
    last_query_split = re.split(r'\n?Query:.*', response)
    if len(last_query_split) < 2:
        print("⚠️ No query found.")
        return []  # Return empty list if no query is found
    
    # Step 2: Extract the last block (after the last query)
    final_block = last_query_split[-1]

    # First attempt: Try to extract numbered items with bracketed headers or descriptors
    numbered_points = re.findall(
        r'(\d+)\.\s*\[([^\]]+)\]\s*(.*?)(?=\n\s*\d+\.\s*\[|$)', 
        final_block, 
        re.DOTALL
    )
    
    if numbered_points:
        cleaned_rationales = [
            (int(num), f"[{header}] {content.strip()}") 
            for num, header, content in numbered_points
        ]
        print(f"✅ Extracted {len(cleaned_rationales)} numbered rationales with bracketed headers.")
        return sorted(cleaned_rationales, key=lambda x: x[0])
    
    # Try alternative formats
    # Format: "2. [Audit rights under...]"
    alt_format = re.findall(
        r'(\d+)\.\s*(\[[^\]]+\].*?)(?=\n\s*\d+\.\s*\[|$)', 
        final_block, 
        re.DOTALL
    )
    
    if alt_format:
        cleaned_rationales = [
            (int(num), content.strip()) 
            for num, content in alt_format
        ]
        print(f"✅ Extracted {len(cleaned_rationales)} numbered rationales (alternative format).")
        return sorted(cleaned_rationales, key=lambda x: x[0])
    
    # Try other common formats
    extraction_patterns = [
        # XML-style rationales
        (r'<rationale_(\d+)>\s*(.*?)\s*</rationale_\1>', lambda m: (int(m[0]), m[1])),
        
        # Plain text numbered rationales
        (r'Rationale\s+(\d+):\s*(.*?)(?=\n\nRationale|$)', lambda m: (int(m[0]), m[1])),
        
        # Simple numbered points
        (r'(\d+)\.\s*(.*?)(?=\n\s*\d+\.\s*|$)', lambda m: (int(m[0]), m[1])),
        
        # "First Rationale" type format
        (r'First Rationale:\s*(.*?)(?=\n\n|$)', lambda m: (1, m)),
        
        # Rationales with "Point X:" format
        (r'Point\s+(\d+):\s*(.*?)(?=\n\s*Point\s+\d+:|$)', lambda m: (int(m[0]), m[1]))
    ]
    
    for pattern, formatter in extraction_patterns:
        matches = re.findall(pattern, final_block, re.DOTALL | re.IGNORECASE)
        if matches:
            try:
                cleaned_rationales = [formatter(m) for m in matches]
                print(f"✅ Extracted {len(cleaned_rationales)} rationales using pattern: {pattern}")
                return sorted(cleaned_rationales, key=lambda x: x[0])
            except Exception as e:
                print(f"⚠️ Error processing matches with pattern {pattern}: {e}")
                continue
    
    # Last resort: Look for any numbered list items
    last_resort = re.findall(r'(\d+)[.):]\s*(.*?)(?=\n\s*\d+[.):]\s*|$)', final_block, re.DOTALL)
    if last_resort:
        cleaned_rationales = [(int(num), text.strip()) for num, text in last_resort]
        print(f"✅ Extracted {len(cleaned_rationales)} numbered items as rationales (last resort).")
        return sorted(cleaned_rationales, key=lambda x: x[0])
    
    print("⚠️ No rationales found after the last query.")
    return []


# For backward compatibility with existing code
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
        "rationale_voting": set(),
        "pooled_embedding": set(),
        "sliding_window": set()
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
        "selected_chunks": sorted_final_chunks,
        "selection_details": selection_details,
        "chunk_similarities": chunk_similarities
    }

def calculate_precision_recall(selected_chunks, correct_chunks, chunks):
    sel, corr = set(selected_chunks), set(correct_chunks)
    tp = len(sel & corr)
    precision = tp / len(sel) if sel else 0.0
    recall = tp / len(corr) if corr else 0.0
    return {"precision": precision, "recall": recall, "correct_chunk_found": float(tp>0)}

# =======================
# EVALUATION ON SPLIT
# =======================
def evaluate_model_on_split(model, tokenizer, examples, sbert_model, max_chunks, split_name, device):
    print(f"Evaluating on {split_name} split with {len(examples)} examples...")
    metrics = {"precision": [], "recall": [], "f1": [], "correct_chunk_found": [], "selected_chunks_count": []}
    results = []
    for ex in examples:
        query, chunks, correct = ex["query"], ex["document_chunks"], ex["correct_chunks"]
        rats = generate_rationales(model, tokenizer, query, device)
        ret = improved_retrieval(rats, chunks, sbert_model, max_chunks)
        sel = ret["selected_chunks"]
        m = calculate_precision_recall(sel, correct, chunks)
        m["f1"] = 2*m["precision"]*m["recall"]/(m["precision"]+m["recall"]) if (m["precision"]+m["recall"])>0 else 0
        metrics["precision"].append(m["precision"])
        metrics["recall"].append(m["recall"])
        metrics["f1"].append(m["f1"])
        metrics["correct_chunk_found"].append(m["correct_chunk_found"])
        metrics["selected_chunks_count"].append(len(sel))
        results.append({"test_id": ex.get("test_id"), "metrics": m, "selected_chunks": sel})
    summary = {k: sum(v)/len(v) if v else 0.0 for k,v in metrics.items()}
    print(f"{split_name} summary: {summary}")
    return {"summary": summary, "individual_results": results}

# =======================
# TEST DATA PROCESSING
# =======================
def prepare_test_data(test_data, tokenizer):
    processed = []
    for ex in test_data:
        if all(k in ex for k in ["query","document_chunks","correct_chunks"]):
            processed.append({"query": ex["query"], "document_chunks": ex["document_chunks"], "correct_chunks": ex["correct_chunks"]})
    return processed

# =======================
# MAIN PIPELINE
# =======================
def run_pipeline(name, path):
    print(f"\n=== Processing dataset: {name} ===")
    tokenizer = AutoTokenizer.from_pretrained(REFINER_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(REFINER_MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sbert_model = SentenceTransformer('Stern5497/sbert-legal-xlm-roberta-base').to(device)
    with open(path, 'r') as f: all_ex = json.load(f)
    total = len(all_ex)
    te, ve = int(total*TRAIN_SPLIT), int(total*(TRAIN_SPLIT+VAL_SPLIT))
    train, val, test = all_ex[:te], all_ex[te:ve], all_ex[ve:]
    dsets = prepare_dpo_dataset_from_splits(train, val, tokenizer)
    if not dsets:
        print(f"Skipping {name}, insufficient data.")
        return
    tokenizer.pad_token = tokenizer.eos_token_id
    cfg = DPOConfig(output_dir=f"./dpo_{name}", per_device_train_batch_size=1, per_device_eval_batch_size=1,
                    gradient_accumulation_steps=2, learning_rate=3e-5, lr_scheduler_type="cosine",
                    num_train_epochs=EPOCHS, warmup_ratio=0.1, logging_steps=10, do_eval=True,
                    eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
                    load_best_model_at_end=True, metric_for_best_model="eval_rewards/chosen",
                    greater_is_better=True, remove_unused_columns=False, beta=0.05)
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=dsets["train"], eval_dataset=dsets["validation"],
                         processing_class=tokenizer, callbacks=[get_print_logs_callback()])
    trainer.train()
    trainer.save_model(f"./dpo_{name}/final_model")
    print(f"DPO training completed for {name}")
    # --- EVALUATION ON TEST ---
    test_proc = prepare_test_data(test, tokenizer)
    eval_res = evaluate_model_on_split(trainer.model, tokenizer, test_proc, sbert_model, MAX_CHUNKS, "test", device)
    os.makedirs(f"./dpo_{name}", exist_ok=True)
    with open(f"./dpo_{name}/test_results.json", "w") as f:
        json.dump(eval_res, f, indent=2)
    print(f"Test evaluation saved for {name}")

if __name__ == "__main__":
    if not torch.cuda.is_available(): print("CUDA unavailable, using CPU!")
    for ds, p in DATASETS.items():
        if os.path.exists(p):
            run_pipeline(ds, p)
        else:
            print(f"Path not found: {p}")
