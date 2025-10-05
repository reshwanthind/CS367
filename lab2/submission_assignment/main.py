import text_processor
import a_star_search
import alignment
from cost_functions import char_level_edit_distance

# download("en_core_web_sm")

# --- Configuration ---
DOC1_PATH = "doc1.txt"
DOC2_PATH = "doc2.txt"
PLAGIARISM_THRESHOLD = 15  # Edit distance below which is considered potential plagiarism

def main():
    """Main function to run the plagiarism detection process."""
    print("--- 1. Processing Documents ---")
    doc1_sentences = text_processor.process_text_file(DOC1_PATH)
    doc2_sentences = text_processor.process_text_file(DOC2_PATH)

    if not doc1_sentences or not doc2_sentences:
        print("Error: One or both documents are empty or could not be read.")
        return

    print(f"Document 1 has {len(doc1_sentences)} sentences.")
    print(f"Document 2 has {len(doc2_sentences)} sentences.\n")

    # --- 2. Running A* Search ---
    print("--- 2. Starting A* Search for Optimal Alignment ---")
    start_state = (0, 0, -1)  # (idx_doc1, idx_doc2, move_type) - -1 for start
    goal_state = (len(doc1_sentences), len(doc2_sentences), -1)
    
    path = a_star_search.a_star(start_state, goal_state, doc1_sentences, doc2_sentences)
    
    if not path:
        print("Could not find an alignment path.")
        return
        
    print("Optimal path found.\n")

    # --- 3. Analyzing Results ---
    print("--- 3. Plagiarism Analysis Report ---")
    aligned_pairs = alignment.reconstruct_alignment(path, doc1_sentences, doc2_sentences)

    print(f"{'Operation':<10} | {'Document 1 Sentence':<60} | {'Document 2 Sentence'}")
    print("-" * 120)

    for sent1, sent2, op in aligned_pairs:
        print(f"{op:<10} | {sent1:<60} | {sent2}")

        if op == "ALIGN":
            distance = char_level_edit_distance(sent1, sent2)
            if distance <= PLAGIARISM_THRESHOLD:
                print(f"  -> \033[91mPOTENTIAL PLAGIARISM DETECTED!\033[0m (Edit Distance: {distance})\n")
            else:
                print(f"  -> No significant similarity. (Edit Distance: {distance})\n")
        else:
            print() # Add a newline for spacing

if __name__ == "__main__":
    main()