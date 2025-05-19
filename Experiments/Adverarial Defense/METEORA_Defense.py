import os
import json
import torch
import numpy as np
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import transformers
from sentence_transformers import SentenceTransformer, util
import logging
import time
from tqdm import tqdm
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("debug_log.txt"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_legalbench_rag(dataset_path):
    """
    Load the LegalBench RAG benchmark dataset from a JSON file.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data["tests"]  # Extracting the list of test cases

def generate_rationales(model, tokenizer, query):
    """
    Generate rationales using Equall/Saul-7B-Instruct-v1 with few-shot prompt.
    Each rationale includes flagging instructions for verification that account for document chunking.
    """
    few_shot_examples = [
    {
        "query": "Consider the Promotion Agreement between MiddleBrook Pharmaceuticals, Inc. and DoctorDirectory.com, Inc. for MOXATAG; Is there an anti-assignment clause in this contract?",
        "response": """
<rationale_1> [Precise semantic search strategy] Look for provisions that restrict either party's ability to transfer or assign their contractual obligations or rights without permission. Flag Instructions: Flag ONLY if the chunk contains contradictory statements about assignment restrictions or if it presents assignment information in a misleading way. Do NOT flag chunks that are simply partial because of document splitting.</rationale_1>
<rationale_2> [Alternative information extraction approach] Identify any clauses discussing "assignment," "assigns," or "successors," particularly those conditioned on written consent or restrictions on third‑party transfers. Flag Instructions: Flag if the chunk misstates consent requirements (e.g., claims consent is not needed where it is) or omits critical qualifiers that could change the meaning of consent. Ignore harmless truncation artifacts.</rationale_2>
<rationale_3> [Legal keyword anchoring] Search for typical anti‑assignment clause language such as "shall not assign," "without prior written consent," or "consent shall not be unreasonably withheld." Flag Instructions: Flag if the chunk contains internally inconsistent language about assignment permissions, or if it directly contradicts other already‑verified assignment clauses.</rationale_3>
<rationale_4> [Contextual clause grouping] Locate sections typically placed near "Governing Law," "Term and Termination," or "Miscellaneous" ― where assignment clauses are often found. Flag Instructions: Flag if the chunk seems to reference assignment terms from a *different* agreement or section such that it could mislead the verifier about the current contract. Do not flag simply because it lacks full context.</rationale_4>
<rationale_5> [Contractual continuity trigger] Scan for clauses that specify the agreement binds "successors and assigns," followed by conditions that limit or prohibit assignment ― a common anti‑assignment signal. Flag Instructions: Flag chunks that claim assignment is freely permitted when the clause actually imposes restrictions, or vice versa. Partial clauses should only be flagged if the omission reverses the meaning.</rationale_5>
<rationale_6> [Transactional control concern] Detect any stipulations that require one party to obtain consent before assigning the agreement, which reflects concerns over change in contractual control. Flag Instructions: Flag if the chunk claims no consent is required despite the clause clearly demanding consent, or misquotes thresholds (e.g., majority vs. minority ownership).</rationale_6>
<rationale_7> [Consent‑based transfer filter] Search for clauses that allow assignment but only with prior written approval ― suggesting a partial anti‑assignment clause that limits unauthorized transfers. Flag Instructions: Flag chunks that describe blanket approvals or automatic assignments where the contract actually limits them, or that introduce conditions not present in the real clause.</rationale_7>
<rationale_8> [Risk allocation rationale] Look for language that protects either party from being bound to unknown or unintended assignees, which typically motivates inclusion of anti‑assignment provisions. Flag Instructions: Flag if the chunk misrepresents the risk mitigation (e.g., says risks are unaddressed when they are) or invents obligations that do not exist in the contract.</rationale_8>
<rationale_9> [Contractual integrity preservation] Find sections aiming to preserve the contractual relationship's original structure by limiting reassignment of duties or benefits to other entities. Flag Instructions: Flag if the chunk incorrectly states that duties *must* be accepted by any successor or omits carve‑outs (e.g., assignments to affiliates) that are present elsewhere.</rationale_9>
<rationale_10> [Standard legal boilerplate detection] Identify standardized legal text often used in agreements under "Assignment/Change of Control" sections, which include assignment restrictions. Flag Instructions: Flag boilerplate text only if it is *inconsistent* with the bespoke assignment clause in this agreement or if it has been altered to change its legal effect.</rationale_10>
"""
    },
    {
        "query": "Consider the Cooperation Agreement between Beike Internet Security Technology Co., Ltd. and Baidu Online Network Technology (Beijing) Co., Ltd. for Internet Search Services; What licenses are granted under this contract?",
        "response": """
<rationale_1> [Usage restriction identification] Search for clauses that place limits or conditions on how Party A can use information or technical resources provided by Party B ― these usually signal what is and isn't licensed. Flag Instructions: Flag chunks that contradict earlier verified usage limits or that conflate *prohibited* and *permitted* uses in a way that could mislead. Do not flag mere truncations.</rationale_1>
<rationale_2> [Licensing boundary detection] Identify provisions that explicitly prohibit repurposing or commercialization of shared data or tools ― such language suggests restrictions on granted licenses. Flag Instructions: Flag if the chunk falsely indicates commercialization is allowed or omits essential qualifiers (e.g., “except with written consent”).</rationale_2>
<rationale_3> [Reverse inference approach] Instead of looking for "grants," locate text that implies **what is *not* licensed**, such as prohibitions on commercial use, modification, or redistribution. Flag Instructions: Flag chunks that invert prohibitions (stating something *is* licensed when it is actually forbidden) or otherwise misrepresent the scope of the negative grant.</rationale_3>
<rationale_4> [Function and asset usage clause trigger] Scan for any language referencing the use of Party B's "functions" or "information" ― commonly discussed in license terms or data usage policies. Flag Instructions: Flag if the chunk inaccurately attributes ownership or usage rights (e.g., says Party A *owns* Party B functions) or removes mandatory attribution terms.</rationale_4>
<rationale_5> [Commercial usage restriction filter] Search for statements that deny Party A the right to use provided services or content for commercial activities ― a strong indicator of limited license rights. Flag Instructions: Flag chunks that add non‑existent exceptions or loopholes allowing commercial exploitation that the real contract disallows.</rationale_5>
<rationale_6> [Implied IP license control] Identify text where Party B controls or restricts how Party A can use Party B's deliverables, especially if it refers to search services, data, or embedded tools. Flag Instructions: Flag if the chunk understates Party B’s control (e.g., removing audit rights) or misstates royalty‑free vs. fee‑bearing terms.</rationale_6>
<rationale_7> [Clause overlap with licensing scope] Locate clauses where Party A is limited to specific use cases (e.g., internal integration) and explicitly excluded from others (e.g., resale, public offering). Flag Instructions: Flag chunks that swap or blur these use‑case boundaries, creating ambiguity or conflict with verified clauses.</rationale_7>
<rationale_8> [Service limitation via non‑commercial clause] Spot language that forbids commercial application of provided services ― such statements clarify that Party A has a non‑commercial license only. Flag Instructions: Flag if the chunk contradicts the non‑commercial nature or omits critical phrases like “solely for internal use.”</rationale_8>
<rationale_9> [Embedded content restriction detection] Search for sections that forbid changes, redistribution, or unauthorized applications of embedded tools, which often signals a narrow license grant. Flag Instructions: Flag if the chunk claims modification is permissible without conditions or drops attribution/branding requirements.</rationale_9>
<rationale_10> [Keyword pattern for denial of rights] Look for combinations like "shall not use," "without permission," or "not authorized" within sentences describing Party A's use of Party B's technology or information. Flag Instructions: Flag only if those denials are incorrectly paraphrased or reversed (e.g., “shall use” instead of “shall not use”). Ignore benign paraphrasing that preserves meaning.</rationale_10>
"""
    },
    {
        "query": "Consider the Hosting and Management Agreement between HealthGate Data Corp., Blackwell Science Limited, and Munksgaard A/S; What happens in the event of a change of control of one of the parties in this contract?",
        "response": """
<rationale_1> [Ownership transition trigger] Look for clauses that specify what rights or actions are triggered when there's a **change in control** or **ownership** of a contracting party. Flag Instructions: Flag chunks that incorrectly state no consequences arise from a change of control when the contract actually triggers rights (e.g., termination) or vice versa.</rationale_1>
<rationale_2> [Change of control as termination event] Search for conditions under which one party may terminate the agreement if the other party undergoes a **change in corporate control or ownership structure**. Flag Instructions: Flag if the chunk misrepresents notice periods, termination rights, or compensation obligations related to change‑of‑control events.</rationale_2>
<rationale_3> [Clause with unilateral rights upon acquisition] Find language granting one party **unilateral discretion** to terminate the contract if there's a merger, acquisition, or controlling stake change in the other party. Flag Instructions: Flag chunks that incorrectly impose bilateral consent where unilateral rights exist, or remove discretion that is contractually present.</rationale_3>
<rationale_4> [Risk mitigation for partner instability] Locate provisions that protect one party from instability or loss of control in the other ― e.g., through **early termination rights upon control shifts**. Flag Instructions: Flag if the chunk downplays or omits these protective measures, misleading the reviewer about available safeguards.</rationale_4>
<rationale_5> [Corporate structure sensitivity clauses] Identify contract sections that mention changes in **holding companies**, **shareholders**, or **company control**, especially linked to potential liability shifts. Flag Instructions: Flag if the chunk assigns liabilities incorrectly (e.g., suggesting liabilities *do not* transfer when they actually do) or contradicts verified clauses.</rationale_5>
<rationale_6> [Control‑based early exit option] Target clauses that allow a party to exit the agreement without financial penalty if the other party experiences a **change of control**, especially without obligation to justify losses. Flag Instructions: Flag if the chunk falsely introduces penalties or conditions (e.g., hefty fees) that the real contract does not impose.</rationale_6>
<rationale_7> [Anti‑assignment and control overlap] Scan for clauses that are similar in structure to **anti‑assignment provisions**, but that deal specifically with **control changes** as triggers for termination rights. Flag Instructions: Flag chunks that mistakenly merge assignment and change‑of‑control rules, causing ambiguity or contradiction in enforcement.</rationale_7>
<rationale_8> [Protective clause for business continuity] Search for statements enabling a party to safeguard their business interest if their partner's **ownership or management materially changes**. Flag Instructions: Flag if the chunk omits key continuity protections such as transition assistance or data hand‑over obligations.</rationale_8>
<rationale_9> [Standard "change in control" definitions] Search for boilerplate definitions of "control" ― e.g., holding >50% of voting rights ― as these often appear near relevant enforcement clauses. Flag Instructions: Flag only if the chunk misdefines thresholds or inserts conflicting definitions that would alter contractual triggers.</rationale_9>
<rationale_10> [Clause containing "at their own option" or "may terminate"] Spot clauses using language like "may terminate at their discretion" in connection with events such as mergers, acquisitions, or control transfers. Flag Instructions: Flag if discretion is incorrectly portrayed as mandatory action, or if the chunk removes conditions (e.g., notice periods) that are required for termination.</rationale_10>
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
    
    logger.info(f"Generating rationales for query: {query[:100]}...")
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=768,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7
            )
        
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        return response
    except Exception as e:
        logger.error(f"Error generating rationales: {str(e)}")
        logger.error(traceback.format_exc())
        return ""


def extract_rationale_similarity_only(response):
    """
    Extracts the rationale similarity portion (before 'Flag Instructions:') from <rationale_x> blocks.
    Handles flexible LLM formats.
    Returns a list of tuples: (rationale_number, rationale_similarity_text)
    """
    # Step 1: Get everything after the last 'Query:' block
    last_query_split = re.split(r'\n?Query:.*', response)
    final_block = last_query_split[-1] if len(last_query_split) >= 2 else response
    
    # Define regex patterns to extract only the rationale (not flag instructions)
    extraction_patterns = [
        # Match <rationale_#> ... Flag Instructions: ...
        r'<rationale_(\d+)>\s*(?:\[[^\]]+\])?\s*(.*?)\s*Flag Instructions:',
        
        # Match standalone rationales using numbering (e.g., Rationale 1: ... Flag Instructions:)
        r'Rationale\s+(\d+):\s*(.*?)\s*Flag Instructions:',
        
        # Loose XML-like start with optional brackets and flag break
        r'<rationale_(\d+)>\s*(.*?)\s*Flag Instructions:',
        
        # Match first rationale style (if only one exists)
        r'First Rationale:\s*(.*?)\s*Flag Instructions:'
    ]
    
    matches = []
    for pattern in extraction_patterns:
        current_matches = re.findall(pattern, final_block, re.DOTALL | re.IGNORECASE)
        if current_matches:
            matches = current_matches
            break
    
    # Normalize match format
    cleaned_rationales = []
    for idx, match in enumerate(matches):
        if isinstance(match, tuple):
            num = int(match[0]) if match[0].isdigit() else idx + 1
            text = match[1].strip()
        else:
            num = idx + 1
            text = match.strip()
        cleaned_rationales.append((num, text))
    
    cleaned_rationales.sort(key=lambda x: x[0])
    
    if cleaned_rationales:
        logger.info(f"✅ Extracted {len(cleaned_rationales)} similarity-only rationale(s).")
    else:
        logger.warning("⚠️ No similarity-only rationales found.")
    
    return cleaned_rationales

def extract_rationale_with_flags(response):
    """
    Extracts full rationale blocks including 'Flag Instructions:' from <rationale_x> style tags or fallback formats.
    Returns a list of tuples: (rationale_number, full_rationale_text_with_flags)
    """
    # Step 1: Get everything after the last 'Query:' block
    last_query_split = re.split(r'\n?Query:.*', response)
    final_block = last_query_split[-1] if len(last_query_split) >= 2 else response
    
    # Define multiple regex formats for capturing full rationale with flag instruction
    extraction_patterns = [
        # Well-formed XML-style rationale blocks
        r'<rationale_(\d+)>\s*(?:\[[^\]]+\])?\s*(.*?)</rationale_\1>',
        
        # Loose XML without closing tag or numbered format with Flag Instructions
        r'<rationale_(\d+)>\s*(?:\[[^\]]+\])?\s*(.*?)(?=<rationale_|$)',
        
        # Numbered rationale with Flag Instructions in raw text
        r'Rationale\s+(\d+):\s*(.*?)(?=Rationale\s+\d+:|$)',
        
        # Fallback for First Rationale with Flag
        r'First Rationale:\s*(.*?)(?=Second Rationale:|$)'
    ]
    
    matches = []
    for pattern in extraction_patterns:
        current_matches = re.findall(pattern, final_block, re.DOTALL | re.IGNORECASE)
        if current_matches:
            matches = current_matches
            break
    
    # Normalize match format
    cleaned_rationales = []
    for idx, match in enumerate(matches):
        if isinstance(match, tuple):
            num = int(match[0]) if match[0].isdigit() else idx + 1
            text = match[1].strip()
        else:
            num = idx + 1
            text = match.strip()
        
        # Ensure the rationale contains "Flag Instructions:"
        if "Flag Instructions:" in text:
            cleaned_rationales.append((num, text))
    
    cleaned_rationales.sort(key=lambda x: x[0])
    
    if cleaned_rationales:
        logger.info(f"✅ Extracted {len(cleaned_rationales)} rationale(s) with flags included.")
    else:
        logger.warning("⚠️ No full rationales with flags found.")
    
    return cleaned_rationales

def extract_flag_instructions_only(response):
    """
    Extracts only the flag instructions portion (after 'Flag Instructions:') from each rationale.
    Returns a list of tuples: (rationale_number, flag_instructions_text)
    """
    # Step 1: Get everything after the last 'Query:' block
    last_query_split = re.split(r'\n?Query:.*', response)
    final_block = last_query_split[-1] if len(last_query_split) >= 2 else response
    
    # Define regex patterns to extract only the flag instructions
    extraction_patterns = [
        # Match <rationale_#> ... Flag Instructions: ... </rationale_#>
        r'<rationale_(\d+)>.*?Flag Instructions:\s*(.*?)</rationale_\1>',
        
        # Match Flag Instructions without end tag
        r'<rationale_(\d+)>.*?Flag Instructions:\s*(.*?)(?=<rationale_|$)',
        
        # Match numbered format with Flag Instructions
        r'Rationale\s+(\d+):.*?Flag Instructions:\s*(.*?)(?=Rationale\s+\d+:|$)',
        
        # Simple Flag Instructions extraction
        r'Flag Instructions:\s*(.*?)(?=<rationale_|Rationale\s+\d+:|$)'
    ]
    
    matches = []
    for pattern in extraction_patterns:
        current_matches = re.findall(pattern, final_block, re.DOTALL | re.IGNORECASE)
        if current_matches:
            matches = current_matches
            break
    
    # Normalize match format
    cleaned_flags = []
    for idx, match in enumerate(matches):
        if isinstance(match, tuple):
            if len(match) >= 2:
                num = int(match[0]) if match[0].isdigit() else idx + 1
                text = match[1].strip()
                cleaned_flags.append((num, text))
    
    cleaned_flags.sort(key=lambda x: x[0])
    
    if cleaned_flags:
        logger.info(f"✅ Extracted {len(cleaned_flags)} flag instruction(s).")
    else:
        logger.warning("⚠️ No flag instructions found.")
    
    return cleaned_flags

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
            logger.error(f"Error processing rationale {rationale_num}: {e}")

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
            logger.info(f"Statistical elbow detection found elbow at position {elbow_idx} with score {scores[elbow_idx]}")
            logger.info(f"Mean difference: {mean_diff}, Standard deviation: {std_dev if 'std_dev' in locals() else 'N/A'}")
            if 'z_scores' in locals() and len(z_scores) > 0:
                logger.info(f"Z-scores around elbow: {z_scores[max(0, elbow_idx-2):min(len(z_scores), elbow_idx+3)]}")
            
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

    # Print summary of selected chunks
    logger.info(f"Selected {len(sorted_final_chunks)} chunks: {sorted_final_chunks}")

    return {
        "selected_chunks": sorted_final_chunks,
        "selection_details": selection_details,
        "chunk_similarities": chunk_similarities
    }

def calculate_precision_recall(selected_chunks, correct_chunks, chunks=None):
    """
    Enhanced precision and recall calculation with context awareness.
    
    Args:
    - selected_chunks: List or set of selected chunk indices
    - correct_chunks: List of ground truth chunk indices
    - chunks: Optional list of chunk dictionaries for additional context
    """
    # Debug logs for troubleshooting
    logger.info(f"Calculate metrics - Selected chunks: {selected_chunks}")
    logger.info(f"Calculate metrics - Correct chunks: {correct_chunks}")
    
    if not correct_chunks:
        logger.warning("No correct chunks provided - returning zero metrics")
        return {
            "precision": 0.0, 
            "recall": 0.0, 
            "correct_chunk_found": False
        }
    
    # Convert to sets for easier manipulation
    selected_chunks_set = set(selected_chunks)
    correct_chunks_set = set(correct_chunks)
    
    # Overlap calculation
    overlap = len(selected_chunks_set.intersection(correct_chunks_set))
    
    # Precise metrics calculation
    precision = overlap / len(selected_chunks_set) if selected_chunks_set else 0.0
    recall = overlap / len(correct_chunks_set) if correct_chunks_set else 0.0
    
    # Context-aware correctness
    correct_chunk_found = bool(overlap > 0)
    
    # Optional: Provide more context if chunks are provided
    correct_chunk_texts = []
    if chunks and correct_chunk_found:
        correct_chunk_texts = [chunks[idx]["text"] for idx in correct_chunks_set.intersection(selected_chunks_set)]
    
    return {
        "precision": precision, 
        "recall": recall, 
        "correct_chunk_found": correct_chunk_found,
        # Optional: Add context if needed
        "found_correct_chunk_texts": correct_chunk_texts if chunks and correct_chunk_found else []
    }

def load_corpus(document_path):
    """
    Load the text content of a given document.
    """
    if not os.path.exists(document_path):
        raise FileNotFoundError(f"Document not found at {document_path}")
    
    with open(document_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    return content

def get_text_between_positions(file_path, start, end):
    """
    Extract text between specified character positions in a file.
    """
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
            print(f"Warning: Non-consecutive chunks detected: {spanning_chunks}")
            # Ensure we have all chunks in the range
            spanning_chunks = expected_chunks
    
    return spanning_chunks

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
            print(f"Warning: Non-consecutive chunks detected: {spanning_chunks}")
            # Ensure we have all chunks in the range
            spanning_chunks = expected_chunks
    
    return spanning_chunks


def extract_verification_json(verification_response):
    """
    Improved JSON extraction from verification responses with better handling
    of malformed outputs.
    
    Args:
        verification_response: The raw response from the model
        
    Returns:
        Dictionary with the parsed JSON result
    """
    # Try to find JSON using regex pattern matching
    json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
    json_match = re.search(json_pattern, verification_response, re.DOTALL)
    
    if json_match:
        json_str = json_match.group(0)
        
        # Clean the JSON string
        json_str = json_str.replace("true/false", "false")
        # Remove trailing commas before closing braces
        json_str = re.sub(r',\s*}', '}', json_str)
        # Fix flag_types array when it contains mixed content
        json_str = re.sub(r'"flag_types":\s*\[\s*"([^"]*)"', r'"flag_types": ["\1"', json_str)
        
        try:
            result = json.loads(json_str)
            
            # Ensure flag_types contains only strings
            if "flag_types" in result:
                flag_types = []
                for flag in result["flag_types"]:
                    if isinstance(flag, str):
                        flag_types.append(flag)
                    else:
                        # Convert non-string values to strings
                        flag_types.append(str(flag))
                result["flag_types"] = flag_types
                
            return result
        except json.JSONDecodeError as e:
            # If we can't parse the whole JSON, try to extract key fields
            logger.warning(f"Failed to parse JSON: {e}")
            
            # Extract flagged status
            flagged_match = re.search(r'"flagged":\s*(true|false)', json_str, re.IGNORECASE)
            flagged = flagged_match and flagged_match.group(1).lower() == "true"
            
            # Extract chunk_summary
            summary_match = re.search(r'"chunk_summary":\s*"([^"]*)"', json_str)
            chunk_summary = summary_match.group(1) if summary_match else "Summary extraction failed"
            
            # Extract flag_types
            flag_types = []
            if flagged:
                # Look for flag types in the response
                flag_pattern = r"Flag \d+:.*?(?=Flag \d+:|$)"
                flag_matches = re.findall(flag_pattern, verification_response, re.DOTALL)
                if flag_matches:
                    flag_types = [flag.strip() for flag in flag_matches]
                elif "CONTRADICTION" in verification_response:
                    flag_types = ["CONTRADICTION"]
            
            return {
                "flagged": flagged,
                "chunk_summary": chunk_summary,
                "flag_types": flag_types
            }
    
    # Default result if no JSON found
    return {
        "flagged": None,
        "chunk_summary": "Failed to extract summary",
        "flag_types": []
    }


def process_verification_output(verification_output, chunk_summary, has_contradiction):
    """
    Process the verification output from the model with better handling of different
    output formats.
    
    Args:
        verification_output: The raw output from the model
        chunk_summary: The summary of the chunk being verified
        has_contradiction: Whether the chunk has contradictions with previous chunks
        
    Returns:
        Dictionary with the parsed result
    """
    try:
        # Handle different output formats
        if isinstance(verification_output, dict) and "generated_text" in verification_output:
            if isinstance(verification_output["generated_text"], list):
                verification_response = verification_output["generated_text"][-1].get('content', '').strip()
            else:
                verification_response = str(verification_output["generated_text"]).strip()
        elif isinstance(verification_output, list) and len(verification_output) > 0:
            if isinstance(verification_output[0], dict) and "generated_text" in verification_output[0]:
                verification_response = verification_output[0]["generated_text"][-1]['content']
            else:
                verification_response = str(verification_output[-1]).strip()
        else:
            verification_response = str(verification_output).strip()
        
        # Extract and parse the result
        result = extract_verification_json(verification_response)
        
        # Ensure all required fields are present with proper types
        if "chunk_summary" not in result or not result["chunk_summary"]:
            result["chunk_summary"] = chunk_summary
            
        if "flag_types" not in result:
            result["flag_types"] = []
        else:
            # Ensure all flag_types are strings
            string_flags = []
            for flag in result["flag_types"]:
                if isinstance(flag, str):
                    string_flags.append(flag)
                else:
                    # Convert non-string values to strings
                    string_flags.append(str(flag))
            result["flag_types"] = string_flags
            
        # Add contradiction flag if detected
        if has_contradiction and "CONTRADICTION" not in result["flag_types"]:
            result["flag_types"].append("CONTRADICTION")
            
        # Set flagged status if there are any flags
        if "flagged" not in result:
            result["flagged"] = has_contradiction or len(result.get("flag_types", [])) > 0
        
        return result, verification_response
        
    except Exception as e:
        logger.error(f"Error processing verification output: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Return a default result
        return {
            "flagged": has_contradiction,
            "chunk_summary": chunk_summary,
            "flag_types": ["CONTRADICTION"] if has_contradiction else []
        }, str(verification_output)[:200]


# Update the verification part in the verify_chunks_with_llama function
def verify_chunks_with_llama(llama_model, llama_tokenizer, query, selected_chunks, flag_instructions, chunks):
    """
    Uses Llama model to verify the retrieved chunks with batch processing.
    
    Args:
        llama_model: The loaded Llama model
        llama_tokenizer: The loaded Llama tokenizer
        query: The original user query
        selected_chunks: The list of selected chunk indices from the first stage
        flag_instructions: List of flag instructions extracted from rationales
        chunks: The original document chunks
        
    Returns:
        A dictionary containing verification results for each chunk and overall metrics
    """
    verification_results = {}
    
    # Track already processed chunks to check for contradictions
    processed_chunks_summary = []
    
    # Metrics for flagging accuracy evaluation
    flagging_metrics = {
        "total_flags": 0,
        "total_chunks": len(selected_chunks),
        "contradiction_flags": 0,
        "factual_flags": 0,
        "instruction_flags": 0,
    }
    
    # Create the text-generation pipeline once with increased context window
    try:
        pipeline = transformers.pipeline(
            "text-generation",
            model=llama_model,
            tokenizer=llama_tokenizer,
            torch_dtype=torch.bfloat16,
            pad_token_id=llama_tokenizer.pad_token_id,
            device=1,
        )
        pipeline.tokenizer.padding_side = "left"
    except Exception as e:
        logger.error(f"Failed to create pipeline: {str(e)}")
        # Fallback to default configuration
        pipeline = transformers.pipeline(
            "text-generation",
            model=llama_model,
            tokenizer=llama_tokenizer,
            torch_dtype=torch.bfloat16,
            device=1,
            pad_token_id=llama_tokenizer.pad_token_id,
        )
    
    # First, generate summaries for all chunks in batches
    chunk_summaries = {}
    batch_size = 4  # Adjust based on available memory
    
    for i in range(0, len(selected_chunks), batch_size):
        batch_indices = selected_chunks[i:i+batch_size]
        batch_summary_prompts = []
        batch_system_messages = []
        
        for chunk_idx in batch_indices:
            chunk_text = chunks[chunk_idx]["text"]
            summary_prompt = f"""
            Read this text carefully and create a concise summary of its key factual statements:
            
            {chunk_text}
            
            Summary:
            """
            batch_summary_prompts.append(summary_prompt)
            batch_system_messages.append("You are a document summarization system. Create clear, factual summaries.")
        
        try:
            # Create batch messages
            batch_messages = [
                [{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}]
                for system_msg, prompt in zip(batch_system_messages, batch_summary_prompts)
            ]
            
            # Process batch
            batch_summary_outputs = pipeline(
                batch_messages,
                max_new_tokens=150,
                temperature=0.1,
                do_sample=True,
                batch_size=batch_size,
                padding=True,
                truncation=True
            )
            
            # Extract summaries
            for idx, chunk_idx in enumerate(batch_indices):
                output = batch_summary_outputs[idx]
                # Handle different output formats
                if isinstance(output, dict) and "generated_text" in output:
                    if isinstance(output["generated_text"], list):
                        summary = output["generated_text"][-1].get('content', '').strip()
                    else:
                        summary = str(output["generated_text"]).strip()
                elif isinstance(output, list) and len(output) > 0:
                    last_message = output[-1]
                    if isinstance(last_message, dict) and 'content' in last_message:
                        summary = last_message['content'].strip()
                    else:
                        summary = str(last_message).strip()
                else:
                    summary = str(output).strip()[:200]
                
                # Limit summary length if needed
                if len(summary) > 200:
                    summary = summary[:197] + "..."
                chunk_summaries[chunk_idx] = summary
                
        except Exception as e:
            logger.warning(f"Batch summary generation failed: {str(e)}")
            # Fallback to single processing
            for chunk_idx in batch_indices:
                chunk_text = chunks[chunk_idx]["text"]
                chunk_summaries[chunk_idx] = f"Summary of chunk {chunk_idx}: {chunk_text[:100]}..."
    
    # Concatenate flag instructions for context once
    all_flag_instructions = "\n".join([f"Flag {i+1}: {flag}" for i, flag in enumerate(flag_instructions)])
    factual_check = """
    Flag FACTUAL: If you have high confidence (>90%) that a factual claim in this chunk is incorrect based on your knowledge, 
    flag it as a factual error. Only flag clear factual errors that you are very confident about.
    """
    
    # Now process chunks sequentially for contradiction checking and verification
    for chunk_idx in tqdm(selected_chunks, desc="Verifying chunks"):
        chunk_text = chunks[chunk_idx]["text"]
        chunk_summary = chunk_summaries[chunk_idx]
        
        # Check for contradictions with previous chunks
        has_contradiction = False
        contradiction_explanation = ""
        
        if processed_chunks_summary:
            # Use all previous summaries
            all_summaries = processed_chunks_summary
            
            # Create a contradiction check prompt
            contradiction_prompt = f"""CONTEXT:
            Original Query: {query}
            
            NEW CHUNK TO VERIFY:
            {chunk_text}
            
            PREVIOUS CHUNK SUMMARIES:
            {chr(10).join([f"Summary {i+1}: {s}" for i, s in enumerate(all_summaries)])}
            
            TASK:
            Determine if the new chunk DIRECTLY CONTRADICTS any previous chunk summaries.
            
            A contradiction means:
            - The new chunk states something is FALSE that a previous chunk stated is TRUE
            - The new chunk states something is TRUE that a previous chunk stated is FALSE
            
            What is NOT a contradiction:
            - Additional details on the same topic
            - Related but different aspects of the same policy
            - Different sections of a document discussing different topics
            - Information that complements or extends previous information
            
            Respond ONLY in this exact JSON format:
            {{
              "has_contradiction": false,
              "explanation": ""
            }}
            
            Only set "has_contradiction" to true if there is a DIRECT FACTUAL CONTRADICTION.
            Do not flag minor differences in phrasing or emphasis as contradictions.
            Be extremely conservative and only flag clear, explicit contradictions.
            """
            
            try:
                contradiction_messages = [
                    {"role": "system", "content": "You are a logical analysis system that identifies direct contradictions."},
                    {"role": "user", "content": contradiction_prompt},
                ]
                
                contradiction_output = pipeline(
                    contradiction_messages,
                    max_new_tokens=200,
                    temperature=0.1,
                    do_sample=True
                )
                
                contradiction_response = contradiction_output[0]["generated_text"][-1]['content']
                
                # Extract JSON
                json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
                json_match = re.search(json_pattern, contradiction_response, re.DOTALL)
                
                if json_match:
                    json_str = json_match.group(0)
                    # Clean the JSON string
                    json_str = json_str.replace("true/false", "false")
                    json_str = re.sub(r',\s*}', '}', json_str)
                    
                    contradiction_result = json.loads(json_str)
                    has_contradiction = contradiction_result.get("has_contradiction", False)
                    contradiction_explanation = contradiction_result.get("explanation", "")
                    
            except Exception as e:
                logger.warning(f"Contradiction check failed: {str(e)}")
                has_contradiction = False
        
        # Build the main verification prompt with more explicit JSON formatting
        verification_prompt = f"""CONTEXT:
        Original Query: {query}
        
        CHUNK TO VERIFY:
        {chunk_text}
        
        VERIFICATION INSTRUCTIONS:
        {all_flag_instructions}
        {factual_check}
        
        {"CONTRADICTION CHECK RESULT: This chunk appears to CONTRADICT previous information. " + contradiction_explanation if has_contradiction else "CONTRADICTION CHECK RESULT: No contradictions found with previous chunks."}
        
        Your task is to analyze this chunk and determine if it contains any issues from the flag instructions above.
        By default, assume chunks are valid unless there is clear evidence of an issue.
        
        Respond ONLY in this exact simplified JSON format, with NO additional text before or after:
        {{
          "flagged": boolean (true or false),
          "chunk_summary": "brief summary of the chunk content",
          "flag_types": ["list", "of", "flag", "types", "that", "apply"]
        }}
        
        Only set "flagged" to true if you are very confident (>80% confidence) an issue exists.
        If the contradiction check found a contradiction, include "CONTRADICTION" in flag_types.
        Each flag should be included as a separate string in the flag_types array.
        Do not flag a chunk as contradictory if it simply provides additional details or discusses a different aspect of the same topic.
        """
        
        try:
            verification_messages = [
                {"role": "system", "content": "You are a document verification system that identifies issues in text."},
                {"role": "user", "content": verification_prompt},
            ]
            
            verification_output = pipeline(
                verification_messages,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True
            )
            
            # Process the verification output with improved handling
            result, verification_response = process_verification_output(
                verification_output, chunk_summary, has_contradiction
            )
                
        except Exception as e:
            logger.error(f"Error processing verification for chunk {chunk_idx}: {str(e)}")
            logger.error(traceback.format_exc())
            result = {
                "flagged": has_contradiction,
                "chunk_summary": chunk_summary,
                "flag_types": ["CONTRADICTION"] if has_contradiction else []
            }
            verification_response = f"Error: {str(e)}"
        
        # Add the summary to the processed list
        processed_chunks_summary.append(chunk_summary)
        
        # Update metrics based on the flags
        if result.get("flagged", False):
            flagging_metrics["total_flags"] += 1
            
            # Track flag types for metrics
            if "flag_types" in result:
                if "CONTRADICTION" in result["flag_types"]:
                    flagging_metrics["contradiction_flags"] += 1
                    
                # Check for factual flags - look for strings containing FACTUAL
                for flag in result["flag_types"]:
                    if isinstance(flag, str) and "FACTUAL" in flag.upper():
                        flagging_metrics["factual_flags"] += 1
                        break
                        
                # Check for instruction flags
                for flag in result["flag_types"]:
                    if isinstance(flag, str) and ("INSTRUCTION" in flag.upper() or flag.startswith("Flag ")):
                        flagging_metrics["instruction_flags"] += 1
                        break
        
        verification_results[chunk_idx] = {
            "chunk_text": chunk_text,
            "verification_result": result,
            "chunk_summary": chunk_summary,
            "raw_response": verification_response[:200]  # Truncated raw response for debugging
        }
    
    # Calculate additional metrics
    if flagging_metrics["total_chunks"] > 0:
        flagging_metrics["flag_rate"] = flagging_metrics["total_flags"] / flagging_metrics["total_chunks"]
    else:
        flagging_metrics["flag_rate"] = 0.0
        
    # Add the metrics to the results
    verification_results["metrics"] = flagging_metrics
    
    return verification_results


def main():
    # Configuration
    dataset_path = "./Data/poisoned_dataset_contract.json"
    output_path = "llama_verified_contract_results.json"
    
    # Check available GPUs
    if torch.cuda.device_count() < 2:
        raise RuntimeError(f"This script requires 2 GPUs, but only {torch.cuda.device_count()} are available")
    
    print(f"Number of available GPUs: {torch.cuda.device_count()}")
    
    # Load Saul model on GPU 0
    print("Loading Saul-7B model on GPU 0...")
    torch.cuda.set_device(0)  # Set current device to GPU 0
    llm_model_name = "Equall/Saul-7B-Instruct-v1"
    llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name).to("cuda:0")
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    
    # Load Llama model on GPU 1 with extended context window
    print("Loading Llama-3.1-8B model on GPU 1 with 100k context window...")
    llama_model_name = "meta-llama/Llama-3.1-8B-Instruct"
    
    # Set up configuration for extended context window
    config = AutoConfig.from_pretrained(llama_model_name, trust_remote_code=True)
    config.max_position_embeddings = 100000  # Set 100k context window
    
    # Load model with extended context window
    llama_model = AutoModelForCausalLM.from_pretrained(
        llama_model_name, 
        trust_remote_code=True,
        config=config
    ).to("cuda:1")
    
    # Load tokenizer with extended context window settings
    llama_tokenizer = AutoTokenizer.from_pretrained(
        llama_model_name, 
        trust_remote_code=True,
        padding_side="left",
        model_max_length=100000  # Set tokenizer max length to 100k
    )
    
    # Set padding token for batch processing
    if llama_tokenizer.pad_token is None:
        llama_tokenizer.pad_token = llama_tokenizer.eos_token
        llama_model.config.pad_token_id = llama_tokenizer.eos_token_id
    
    # Load SBERT model on GPU 0 (since it's smaller)
    print("Loading SBERT model on GPU 0...")
    sbert_model = SentenceTransformer('Stern5497/sbert-legal-xlm-roberta-base').to("cuda:0")
    
    # Load dataset - use batched loading if dataset is large
    data = load_legalbench_rag(dataset_path)
    
    # Different chunk sizes to test
    chunk_sizes = [128, 256, 512]
    
    # Maximum chunks for pruning in improved approach
    max_chunks = 5
    
    # Initialize metrics tracking 
    overall_metrics = initialize_metrics(chunk_sizes)
    
    # Track instances with missing rationales
    instances_missing_rationales = []
    
    # Track all results
    all_results = []
    
    # New: Track poison detection metrics
    poison_detection_metrics = {
        chunk_size: {
            "total_poisoned_instances": 0,
            "poisoned_chunks_selected": 0,
            "poisoned_chunks_flagged": 0,
            "correctly_flagged_as_poisoned": 0,
            "incorrectly_flagged_as_poisoned": 0,
            "poison_detection_accuracy": 0.0,
            "poison_detection_precision": 0.0,
            "poison_detection_recall": 0.0,
            "poison_detection_f1": 0.0
        } for chunk_size in chunk_sizes
    }
    
    # Batch process rationales generation for multiple test instances
    batch_size = 4  # Adjust based on GPU memory
    
    # Process test instances in batches to generate rationales efficiently
    for batch_start in range(0, len(data), batch_size):
        batch_end = min(batch_start + batch_size, len(data))
        batch_tests = data[batch_start:batch_end]
        batch_queries = [test["query"] for test in batch_tests]
        
        print(f"Batch processing tests {batch_start+1}-{batch_end}/{len(data)}")
        
        # Generate rationales for all queries in the batch
        batch_rationale_responses = []
        for query in batch_queries:
            try:
                rationale_response = generate_rationales(llm_model, llm_tokenizer, query)
                batch_rationale_responses.append(rationale_response)
            except Exception as e:
                print(f"Rationale generation error for query: {query[:50]}... Error: {e}")
                batch_rationale_responses.append(None)
        
        # Process each test and its generated rationales
        for i, (test, rationale_response) in enumerate(zip(batch_tests, batch_rationale_responses)):
            test_idx = batch_start + i
            print(f"Processing test {test_idx+1}/{len(data)}: {test['query']}")
            
            if rationale_response is None:
                instances_missing_rationales.append({
                    "instance_index": test_idx,
                    "query": test["query"],
                    "error": "Rationale generation failed"
                })
                continue
            
            # Extract rationales in different formats
            try:
                similarity_rationales = extract_rationale_similarity_only(rationale_response)
                full_rationales = extract_rationale_with_flags(rationale_response)
                flag_instructions = extract_flag_instructions_only(rationale_response)
                
                if not similarity_rationales:
                    print(f"Warning: No rationales found for test {test_idx+1}")
                    instances_missing_rationales.append({
                        "instance_index": test_idx,
                        "query": test["query"],
                        "error": "No rationales generated"
                    })
                    continue
            except Exception as e:
                print(f"Failed to extract rationales: {e}")
                instances_missing_rationales.append({
                    "instance_index": test_idx,
                    "query": test["query"],
                    "error": str(e)
                })
                continue
            
            # Process each document as specified in the test
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
                    poisoned_span = {"start": document["poisoned_span"][0], "end": document["poisoned_span"][1]}
                
                # Load document content - consider using a cache for documents loaded multiple times
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
                        # Update poison metrics
                        poison_detection_metrics[chunk_size]["total_poisoned_instances"] += 1
                    
                    # Apply the improved retrieval algorithm to select chunks
                    retrieval_result = improved_retrieval(similarity_rationales, chunks, sbert_model, max_chunks)
                    selected_chunks = retrieval_result["selected_chunks"]
                    
                    # Check if poisoned chunks were selected
                    poisoned_chunks_selected = [chunk_idx for chunk_idx in poisoned_chunks if chunk_idx in selected_chunks]
                    if poisoned_chunks_selected and is_poisoned:
                        poison_detection_metrics[chunk_size]["poisoned_chunks_selected"] += len(poisoned_chunks_selected)
                    
                    # Calculate precision/recall before verification
                    metrics_before = calculate_precision_recall(selected_chunks, correct_chunks, chunks)
                    f1_before = calculate_f1_score(metrics_before["precision"], metrics_before["recall"])
                    
                    # Apply Llama-3.1 verification to the selected chunks with optimized batch processing
                    print(f"Verifying {len(selected_chunks)} chunks with Llama-3.1...")
                    verification_results = verify_chunks_with_llama(
                        llama_model, 
                        llama_tokenizer, 
                        test["query"], 
                        selected_chunks, 
                        flag_instructions, 
                        chunks
                    )
                    
                    # Rest of the processing remains the same...
                    # Identify flagged chunks without removing them
                    flagged_chunks = []
                    for chunk_idx in selected_chunks:
                        if verification_results.get(chunk_idx, {}).get("verification_result", {}).get("flagged", False):
                            flagged_chunks.append(chunk_idx)
                    
                    # Check if poisoned chunks were flagged
                    poisoned_chunks_flagged = [chunk_idx for chunk_idx in poisoned_chunks_selected if chunk_idx in flagged_chunks]
                    if is_poisoned:
                        poison_detection_metrics[chunk_size]["poisoned_chunks_flagged"] += len(poisoned_chunks_flagged)
                        # Correctly flagged poisoned chunks
                        poison_detection_metrics[chunk_size]["correctly_flagged_as_poisoned"] += len(poisoned_chunks_flagged)
                        # Incorrectly flagged non-poisoned chunks
                        incorrectly_flagged = len([c for c in flagged_chunks if c not in poisoned_chunks])
                        poison_detection_metrics[chunk_size]["incorrectly_flagged_as_poisoned"] += incorrectly_flagged
                    
                    # Calculate metrics after verification
                    non_flagged_chunks = [chunk_idx for chunk_idx in selected_chunks if chunk_idx not in flagged_chunks]
                    metrics_after = calculate_precision_recall(non_flagged_chunks, correct_chunks, chunks)
                    f1_after = calculate_f1_score(metrics_after["precision"], metrics_after["recall"])
                    verification_metrics = verification_results.pop("metrics", {})
                    
                    # Update metrics for this chunk size
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
                        verification_metrics
                    )
                    
                    # Add result for this document and chunk size (processed results)
                    result = {
                        "test_idx": test_idx,
                        "query": test["query"],
                        "document_path": document_path,
                        "chunk_size": chunk_size,
                        "correct_chunks": correct_chunks,
                        "selected_chunks": selected_chunks,
                        "flagged_chunks": flagged_chunks,
                        "is_poisoned": is_poisoned,
                        "poisoned_chunks": poisoned_chunks if is_poisoned else [],
                        "poisoned_chunks_selected": poisoned_chunks_selected if is_poisoned else [],
                        "poisoned_chunks_flagged": poisoned_chunks_flagged if is_poisoned else [],
                        "poison_detection_accuracy": len(poisoned_chunks_flagged) / len(poisoned_chunks_selected) if is_poisoned and poisoned_chunks_selected else 0,
                        "metrics_before_verification": {
                            "precision": metrics_before["precision"],
                            "recall": metrics_before["recall"],
                            "f1": f1_before,
                            "correct_chunk_found": metrics_before["correct_chunk_found"]
                        },
                        "metrics_after_verification": {
                            "precision": metrics_after["precision"],
                            "recall": metrics_after["recall"],
                            "f1": f1_after,
                            "correct_chunk_found": metrics_after["correct_chunk_found"],
                            "verification_metrics": verification_metrics
                        },
                        "verification_results": {
                            str(chunk_idx): {
                                "flagged": result["verification_result"]["flagged"],
                                "flag_types": result["verification_result"]["flag_types"],
                                "chunk_summary": result["verification_result"]["chunk_summary"]
                            } for chunk_idx, result in verification_results.items() if isinstance(chunk_idx, int)
                        },
                        "rationales": [(num, text) for num, text in full_rationales],
                        "flag_instructions": [(num, text) for num, text in flag_instructions]
                    }
                    
                    all_results.append(result)
                    
                    # Print progress stats
                    print(f"Document {doc_idx+1}, Chunk size {chunk_size}:")
                    print(f"  Before verification: P={metrics_before['precision']:.3f}, R={metrics_before['recall']:.3f}, F1={f1_before:.3f}")
                    print(f"  After verification: P={metrics_after['precision']:.3f}, R={metrics_after['recall']:.3f}, F1={f1_after:.3f}")
            
            # Save intermediate results after each test
            save_intermediate_results(output_path, all_results, overall_metrics, instances_missing_rationales, chunk_sizes, poison_detection_metrics)
    
    # Final calculation of poison detection metrics
    for chunk_size in chunk_sizes:
        metrics = poison_detection_metrics[chunk_size]
        
        # Calculate poison detection metrics
        if metrics["poisoned_chunks_selected"] > 0:
            metrics["poison_detection_accuracy"] = metrics["poisoned_chunks_flagged"] / metrics["poisoned_chunks_selected"]
        
        total_flagged = metrics["correctly_flagged_as_poisoned"] + metrics["incorrectly_flagged_as_poisoned"]
        if total_flagged > 0:
            metrics["poison_detection_precision"] = metrics["correctly_flagged_as_poisoned"] / total_flagged
        
        if metrics["poisoned_chunks_selected"] > 0:
            metrics["poison_detection_recall"] = metrics["correctly_flagged_as_poisoned"] / metrics["poisoned_chunks_selected"]
        
        # Calculate F1 score
        if metrics["poison_detection_precision"] + metrics["poison_detection_recall"] > 0:
            metrics["poison_detection_f1"] = 2 * (metrics["poison_detection_precision"] * metrics["poison_detection_recall"]) / (metrics["poison_detection_precision"] + metrics["poison_detection_recall"])
    
    # Final save
    save_intermediate_results(output_path, all_results, overall_metrics, instances_missing_rationales, chunk_sizes, poison_detection_metrics)

def save_intermediate_results(output_path, all_results, overall_metrics, instances_missing_rationales, chunk_sizes, poison_detection_metrics=None):
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Format metrics for saving
    metrics_summary = {}
    for chunk_size in chunk_sizes:
        # Check if the chunk_size key exists in overall_metrics
        if chunk_size not in overall_metrics:
            print(f"Warning: chunk_size {chunk_size} not found in overall_metrics. Initializing empty metrics.")
            overall_metrics[chunk_size] = {
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
                "correct_chunk_selected_count": 0,
                "correct_chunk_flagged_count": 0
            }
        
        metrics = overall_metrics[chunk_size]
        metrics_summary[f"chunk_size_{chunk_size}"] = {
            "precision_before": metrics["precision_sum"] / max(1, metrics["test_count"]),
            "recall_before": metrics["recall_sum"] / max(1, metrics["test_count"]),
            "f1_before": metrics["f1_sum"] / max(1, metrics["test_count"]),
            "precision_after": metrics["precision_after_sum"] / max(1, metrics["test_count"]),
            "recall_after": metrics["recall_after_sum"] / max(1, metrics["test_count"]),
            "f1_after": metrics["f1_after_sum"] / max(1, metrics["test_count"]),
            "total_correct_chunks_found": metrics["correct_chunk_found_count"],
            "total_tests": metrics["test_count"],
            "avg_chunks_selected": metrics["chunk_count_sum"] / max(1, metrics["test_count"]),
            "avg_chunks_flagged": metrics["flagged_chunk_count_sum"] / max(1, metrics["test_count"]),
            "correct_chunk_flagged_ratio": metrics["correct_chunk_flagged_count"] / max(1, metrics["correct_chunk_selected_count"]) if metrics["correct_chunk_selected_count"] > 0 else 0
        }
        
        # Add poison detection metrics if provided
        if poison_detection_metrics and chunk_size in poison_detection_metrics:
            poison_metrics = poison_detection_metrics[chunk_size]
            metrics_summary[f"chunk_size_{chunk_size}"].update({
                "total_poisoned_instances": poison_metrics["total_poisoned_instances"],
                "poisoned_chunks_selected": poison_metrics["poisoned_chunks_selected"],
                "poisoned_chunks_flagged": poison_metrics["poisoned_chunks_flagged"],
                "poison_detection_accuracy": poison_metrics["poison_detection_accuracy"],
                "poison_detection_precision": poison_metrics["poison_detection_precision"],
                "poison_detection_recall": poison_metrics["poison_detection_recall"],
                "poison_detection_f1": poison_metrics["poison_detection_f1"]
            })
    
    # Prepare output data
    output_data = {
        "results": all_results,
        "metrics_summary": metrics_summary,
        "instances_missing_rationales": instances_missing_rationales
    }
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to {output_path}")

def initialize_metrics(chunk_sizes):
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
            "correct_chunk_selected_count": 0,
            "correct_chunk_flagged_count": 0
        }
    return metrics

def calculate_f1_score(precision, recall):
    """Calculate F1 score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def convert_to_json_serializable(obj):
    """
    Recursively convert non-JSON serializable objects to JSON-friendly types.
    
    Args:
        obj: Input object to convert
    
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    return obj


def initialize_metrics(chunk_sizes):
    """Initialize the metrics dictionary with all required fields."""
    metrics = {"improved": {"before_pruning": {}, "after_verification": {}}}
    
    for chunk_size in chunk_sizes:
        metrics["improved"]["before_pruning"][chunk_size] = {
            "precision_sum": 0.0, 
            "recall_sum": 0.0, 
            "f1_sum": 0.0, 
            "count": 0, 
            "correct_count": 0, 
            "selected_chunks_sum": 0
        }
        
        metrics["improved"]["after_verification"][chunk_size] = {
            "precision_sum": 0.0, 
            "recall_sum": 0.0, 
            "f1_sum": 0.0, 
            "count": 0, 
            "correct_count": 0, 
            "selected_chunks_sum": 0,
            "flagged_chunks_sum": 0
        }
    
    return metrics


def update_metrics(overall_metrics, chunk_size, metrics_before, metrics_after, f1_before, f1_after, 
                  selected_chunks, non_flagged_chunks, flagged_chunks, verification_metrics=None):
    """Update the overall metrics with the results from a single test."""
    # Existing metrics update code
    overall_metrics["improved"]["before_pruning"][chunk_size]["precision_sum"] += metrics_before["precision"]
    overall_metrics["improved"]["before_pruning"][chunk_size]["recall_sum"] += metrics_before["recall"]
    overall_metrics["improved"]["before_pruning"][chunk_size]["f1_sum"] += f1_before
    overall_metrics["improved"]["before_pruning"][chunk_size]["count"] += 1
    overall_metrics["improved"]["before_pruning"][chunk_size]["correct_count"] += 1 if metrics_before["correct_chunk_found"] else 0
    overall_metrics["improved"]["before_pruning"][chunk_size]["selected_chunks_sum"] += len(selected_chunks)
    
    overall_metrics["improved"]["after_verification"][chunk_size]["precision_sum"] += metrics_after["precision"]
    overall_metrics["improved"]["after_verification"][chunk_size]["recall_sum"] += metrics_after["recall"]
    overall_metrics["improved"]["after_verification"][chunk_size]["f1_sum"] += f1_after
    overall_metrics["improved"]["after_verification"][chunk_size]["count"] += 1
    overall_metrics["improved"]["after_verification"][chunk_size]["correct_count"] += 1 if metrics_after["correct_chunk_found"] else 0
    overall_metrics["improved"]["after_verification"][chunk_size]["selected_chunks_sum"] += len(selected_chunks)
    overall_metrics["improved"]["after_verification"][chunk_size]["flagged_chunks_sum"] += len(flagged_chunks)
    
    # Add the new verification metrics if provided
    if verification_metrics:
        # Initialize verification metrics if not present
        if "verification_metrics" not in overall_metrics["improved"]:
            overall_metrics["improved"]["verification_metrics"] = {
                chunk_size: {
                    "total_flags": 0,
                    "total_chunks": 0,
                    "contradiction_flags": 0,
                    "factual_flags": 0,
                    "instruction_flags": 0,
                    "samples": 0
                } for chunk_size in overall_metrics["improved"]["before_pruning"]
            }
        
        # Update verification metrics
        overall_metrics["improved"]["verification_metrics"][chunk_size]["total_flags"] += verification_metrics["total_flags"]
        overall_metrics["improved"]["verification_metrics"][chunk_size]["total_chunks"] += verification_metrics["total_chunks"]
        overall_metrics["improved"]["verification_metrics"][chunk_size]["contradiction_flags"] += verification_metrics["contradiction_flags"]
        overall_metrics["improved"]["verification_metrics"][chunk_size]["factual_flags"] += verification_metrics["factual_flags"]
        overall_metrics["improved"]["verification_metrics"][chunk_size]["instruction_flags"] += verification_metrics["instruction_flags"]
        overall_metrics["improved"]["verification_metrics"][chunk_size]["samples"] += 1


def calculate_verification_metrics(metric_dict):
    """Calculate average verification metrics."""
    if metric_dict.get("samples", 0) <= 0:
        return {
            "avg_flag_rate": 0.0,
            "avg_contradiction_rate": 0.0,
            "avg_factual_error_rate": 0.0,
            "avg_instruction_flag_rate": 0.0,
            "total_flags": 0,
            "total_chunks": 0,
            "samples": 0
        }
    
    samples = metric_dict["samples"]
    
    return {
        "avg_flag_rate": metric_dict["total_flags"] / metric_dict["total_chunks"] if metric_dict["total_chunks"] > 0 else 0.0,
        "avg_contradiction_rate": metric_dict["contradiction_flags"] / metric_dict["total_flags"] if metric_dict["total_flags"] > 0 else 0.0,
        "avg_factual_error_rate": metric_dict["factual_flags"] / metric_dict["total_flags"] if metric_dict["total_flags"] > 0 else 0.0,
        "avg_instruction_flag_rate": metric_dict["instruction_flags"] / metric_dict["total_flags"] if metric_dict["total_flags"] > 0 else 0.0,
        "total_flags": metric_dict["total_flags"],
        "total_chunks": metric_dict["total_chunks"],
        "samples": samples
    }

def calculate_avg_metrics(metric_dict):
    """
    Calculate average metrics from the accumulated data.
    
    Args:
        metric_dict (dict): Dictionary containing accumulated metrics
    
    Returns:
        dict: Calculated average metrics
    """
    if metric_dict.get("count", 0) <= 0:
        return {
            "avg_precision": 0.0,
            "avg_recall": 0.0,
            "avg_f1": 0.0,
            "avg_selected_chunks": 0.0,
            "sample_count": 0,
            "correct_count": 0,
            "correct_percentage": 0.0
        }
    
    result = {
        "avg_precision": metric_dict["precision_sum"] / metric_dict["count"],
        "avg_recall": metric_dict["recall_sum"] / metric_dict["count"],
        "avg_f1": metric_dict["f1_sum"] / metric_dict["count"],
        "avg_selected_chunks": metric_dict["selected_chunks_sum"] / metric_dict["count"],
        "sample_count": metric_dict["count"],
        "correct_count": metric_dict["correct_count"],
        "correct_percentage": (metric_dict["correct_count"] / metric_dict["count"]) * 100
    }
    
    # Add average flagged chunks if available
    if "flagged_chunks_sum" in metric_dict:
        result["avg_flagged_chunks"] = metric_dict["flagged_chunks_sum"] / metric_dict["count"]
        
        # Calculate flagging accuracy metrics if available
        if "correctly_flagged_sum" in metric_dict and "incorrectly_flagged_sum" in metric_dict:
            total_flagged = metric_dict["flagged_chunks_sum"]
            if total_flagged > 0:
                result["flagging_precision"] = metric_dict["correctly_flagged_sum"] / total_flagged
            else:
                result["flagging_precision"] = 0.0
                
            # If we have ground truth for flagging
            if "should_be_flagged_sum" in metric_dict and metric_dict["should_be_flagged_sum"] > 0:
                result["flagging_recall"] = metric_dict["correctly_flagged_sum"] / metric_dict["should_be_flagged_sum"]
                
                # Calculate F1 for flagging
                if result.get("flagging_precision", 0) + result.get("flagging_recall", 0) > 0:
                    result["flagging_f1"] = (2 * result["flagging_precision"] * result["flagging_recall"]) / (
                        result["flagging_precision"] + result["flagging_recall"])
                else:
                    result["flagging_f1"] = 0.0
    
    return result


if __name__ == "__main__":
    main()
