import random

def generate_k_sat_problem(k, m, n):
    """
    Generates a uniform random k-SAT problem with m clauses over n elements and k literals per clause.
    """
    
    if k > n:
        raise ValueError("k cannot be greater than n") # Not enough elements to form clauses

    clauses = set() # set to avoid duplicate clauses
    elements = range(1, n + 1)

    while len(clauses) < m:
        # Choose k distinct elements
        clause_vars = random.sample(elements, k)
        
        # Randomly negate each element
        clause = tuple(var * random.choice([-1, 1]) for var in clause_vars)
        clauses.add(clause)
        
    return clauses

# #Example usage:
# clauses = generate_k_sat_problem(3, 5, 4)
# print(clauses)