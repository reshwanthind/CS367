from typing import List, Tuple

def reconstruct_alignment(path: List[Tuple[int, int, int]], doc1: List[str], doc2: List[str]) -> List[Tuple[str, str, str]]:
    """
    Reconstructs the sentence alignment based on the A* path.

    Returns:
        A list of tuples, where each tuple contains:
        (sentence_from_doc1, sentence_from_doc2, operation)
    """
    aligned_pairs = []
    
    for state in path:
        idx1, idx2, move = state
        
        if idx1 == 0 and idx2 == 0: # Skip start state
            continue

        sent1 = ""
        sent2 = ""
        operation = ""

        if move == 0: # Alignment
            sent1 = doc1[idx1 - 1]
            sent2 = doc2[idx2 - 1]
            operation = "ALIGN"
        elif move == 1: # Insertion
            sent1 = "---" # Represents a gap
            sent2 = doc2[idx2 - 1]
            operation = "INSERT"
        elif move == 2: # Deletion
            sent1 = doc1[idx1 - 1]
            sent2 = "---" # Represents a gap
            operation = "DELETE"
            
        aligned_pairs.append((sent1, sent2, operation))
        
    return aligned_pairs