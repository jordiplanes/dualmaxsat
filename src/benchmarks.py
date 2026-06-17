import random
from pysat.formula import WCNF
from example import compare_wcnf

def generate_random_max3sat(n_vars: int, n_clauses: int, seed: int = 42, weighted: bool = False) -> WCNF:
    """Generate a random Max-3SAT instance where all clauses are soft."""
    rng = random.Random(seed)
    wcnf = WCNF()
    for _ in range(n_clauses):
        # Pick 3 distinct variables
        vars = rng.sample(range(1, n_vars + 1), 3)
        # Randomly negate
        clause = [v if rng.random() > 0.5 else -v for v in vars]
        # Weight 1, soft clause
        weight = rng.randint(1, 100) if weighted else 1
        wcnf.append(clause, weight=weight)
    return wcnf

def generate_vertex_cover(n_nodes: int, p_edge: float, seed: int = 42, weighted: bool = False) -> WCNF:
    """
    Generate a Minimum Vertex Cover instance on a random Erdos-Renyi graph G(n, p).
    Formulated as MaxSAT:
    - Hard clauses: for each edge (u, v), u or v must be in the cover (u v v).
    - Soft clauses: for each node u, we prefer u not to be in the cover (-u), weight 1.
    """
    rng = random.Random(seed)
    wcnf = WCNF()
    edges = 0
    
    # Hard clauses for edges
    for i in range(1, n_nodes + 1):
        for j in range(i + 1, n_nodes + 1):
            if rng.random() < p_edge:
                wcnf.append([i, j]) # Hard clause
                edges += 1
                
    # Soft clauses to minimize cover size (maximize nodes NOT in cover)
    for i in range(1, n_nodes + 1):
        weight = rng.randint(1, 100) if weighted else 1
        wcnf.append([-i], weight=weight)
        
    return wcnf, edges

def generate_set_cover(n_elements: int, n_sets: int, p_contain: float, seed: int = 42, weighted: bool = False) -> WCNF:
    """
    Generate a Minimum Set Cover instance.
    - Variables: 1 to n_sets (representing whether each set is chosen).
    - Hard clauses: For each element, at least one set containing it must be chosen.
    - Soft clauses: We prefer not to choose a set (-S), weight 1.
    """
    rng = random.Random(seed)
    wcnf = WCNF()
    
    element_to_sets = {e: [] for e in range(1, n_elements + 1)}
    
    for s in range(1, n_sets + 1):
        for e in range(1, n_elements + 1):
            if rng.random() < p_contain:
                element_to_sets[e].append(s)
                
    # If any element is in no sets, add it to a random set to ensure satisfiability of hard clauses
    for e in range(1, n_elements + 1):
        if not element_to_sets[e]:
            s = rng.randint(1, n_sets)
            element_to_sets[e].append(s)
            
    # Hard clauses: each element covered
    for e in range(1, n_elements + 1):
        wcnf.append(element_to_sets[e])
        
    # Soft clauses: minimize sets chosen
    for s in range(1, n_sets + 1):
        weight = rng.randint(1, 100) if weighted else 1
        wcnf.append([-s], weight=weight)
        
    return wcnf

def main():
    print("Running Parameterized Benchmarks...")
    
    # 1. Random Max-3SAT
    n_vars, n_clauses = 20, 80
    wcnf_3sat = generate_random_max3sat(n_vars, n_clauses, seed=123)
    compare_wcnf(
        f"Random Max-3SAT (V={n_vars}, C={n_clauses})", 
        wcnf_3sat, 
        f"max3sat_{n_vars}_{n_clauses}"
    )
    
    # 2. Vertex Cover
    n_nodes, p_edge = 20, 0.2
    wcnf_vc, edges = generate_vertex_cover(n_nodes, p_edge, seed=123)
    compare_wcnf(
        f"Vertex Cover on G(n={n_nodes}, p={p_edge}) [Edges={edges}]", 
        wcnf_vc, 
        f"vc_{n_nodes}_{p_edge}"
    )
    
    # 3. Set Cover
    n_elements, n_sets, p_contain = 20, 15, 0.2
    wcnf_sc = generate_set_cover(n_elements, n_sets, p_contain, seed=123)
    compare_wcnf(
        f"Set Cover (Elements={n_elements}, Sets={n_sets}, p={p_contain})", 
        wcnf_sc, 
        f"sc_{n_elements}_{n_sets}_{p_contain}"
    )

if __name__ == "__main__":
    main()
