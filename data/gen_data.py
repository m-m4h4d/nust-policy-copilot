import json
import random

# --- CONFIGURATION ---
INPUT_FILE = "nust_corpus_preprocessed.json"
OUTPUT_FILE = "eval_set.json"
TARGET_TOTAL = 120
TOPICS = {
    "Attendance": ["attendance", "absent", "leave", "shortage"],
    "Grading": ["GPA", "CGPA", "grading", "exam", "probation", "transcript", "credit hour"],
    "Hostel": ["hostel", "mess", "curfew", "accommodation", "laundry", "boarding"],
    "Fees": ["fee", "refund", "tuition", "dues", "scholarship", "financial aid", "security deposit"],
    "Curriculum": ["project", "degree", "internship", "course", "elective", "software engineering", "FYP"]
}

# Diverse Templates to avoid repetition
TEMPLATES = [
    "What are the official rules regarding {keyword}?",
    "How does the handbook define the policy for {keyword}?",
    "Can you explain the procedure for {keyword}?",
    "What specific requirements are listed for {keyword}?",
    "If a student has an issue with {keyword}, what is the protocol?",
    "What are the consequences related to {keyword}?",
    "Is there any mention of {keyword} in the regulations?",
    "Detail the guidelines provided for {keyword}.",
    "What is the NUST stance on {keyword}?",
    "Provide a summary of the {keyword} section."
]

def load_corpus():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_balanced_dataset():
    corpus = load_corpus()
    eval_set = []
    used_doc_ids = set()
    
    # Calculate target per topic (approx 24 per topic for 5 topics)
    target_per_topic = TARGET_TOTAL // len(TOPICS)
    
    print(f"--- Generating {TARGET_TOTAL} Questions ({target_per_topic} per topic) ---")

    for topic, keywords in TOPICS.items():
        topic_count = 0
        
        # Shuffle corpus to avoid getting only the first few chapters
        random.shuffle(corpus)
        
        for chunk in corpus:
            if topic_count >= target_per_topic: 
                break
                
            doc_id = chunk['metadata']['doc_id']
            content = chunk['page_content'].lower()
            
            # Skip if we already used this exact chunk (prevents duplicate answers)
            if doc_id in used_doc_ids:
                continue
                
            # Check for keyword match
            matched_keyword = None
            for k in keywords:
                if k.lower() in content:
                    matched_keyword = k
                    break
            
            if matched_keyword:
                # Select a random template
                template = random.choice(TEMPLATES)
                question_text = template.format(keyword=matched_keyword)
                
                # Add to dataset
                eval_set.append({
                    "question": question_text,
                    "gold_ids": [doc_id],
                    "topic": topic,
                    "source_preview": chunk['page_content'][:100] + "..."
                })
                
                used_doc_ids.add(doc_id)
                topic_count += 1
        
        print(f"  Topic '{topic}': Generated {topic_count} questions.")

    # FALLBACK: If we are short of 120, fill with random general questions
    while len(eval_set) < TARGET_TOTAL:
        chunk = random.choice(corpus)
        if chunk['metadata']['doc_id'] not in used_doc_ids:
            eval_set.append({
                "question": f"Summarize the policy details found in section {chunk['metadata'].get('page', '?')}.",
                "gold_ids": [chunk['metadata']['doc_id']],
                "topic": "General/Misc",
                "source_preview": chunk['page_content'][:100] + "..."
            })
            used_doc_ids.add(chunk['metadata']['doc_id'])
    
    # Shuffle final set so topics are mixed
    random.shuffle(eval_set)
    
    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, indent=4)
        
    print(f"--- Success! Saved {len(eval_set)} unique questions to {OUTPUT_FILE} ---")

if __name__ == "__main__":
    generate_balanced_dataset()