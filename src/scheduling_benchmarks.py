import random
from pysat.formula import WCNF, IDPool
from pysat.card import CardEnc, EncType
from example import compare_wcnf

def generate_task_allocation(n_tasks: int, m_machines: int, capacity: int, seed: int = 42) -> WCNF:
    """
    1. Task-Machine Allocation
    Variables: x_{t,m} = task t assigned to machine m
    Hard: exactly one machine per task, at most `capacity` tasks per machine
    Soft: cost c_{t,m} to assign t to m. Minimize cost -> maximize weight of -x_{t,m}
    """
    rng = random.Random(seed)
    wcnf = WCNF()
    vpool = IDPool()
    
    def var(t, m):
        return vpool.id(f"x_{t}_{m}")
    
    # Hard: exactly 1 machine per task
    for t in range(1, n_tasks + 1):
        vars_t = [var(t, m) for m in range(1, m_machines + 1)]
        cnf = CardEnc.equals(lits=vars_t, bound=1, vpool=vpool, encoding=EncType.seqcounter)
        for clause in cnf.clauses:
            wcnf.append(clause)
            
    # Hard: at most `capacity` tasks per machine
    for m in range(1, m_machines + 1):
        vars_m = [var(t, m) for t in range(1, n_tasks + 1)]
        cnf = CardEnc.atmost(lits=vars_m, bound=capacity, vpool=vpool, encoding=EncType.seqcounter)
        for clause in cnf.clauses:
            wcnf.append(clause)

    # Soft: assignment costs
    for t in range(1, n_tasks + 1):
        for m in range(1, m_machines + 1):
            cost = rng.randint(10, 100)
            # Minimize cost => soft clause is to NOT assign, with weight = cost
            wcnf.append([-var(t, m)], weight=cost)
            
    return wcnf

def generate_gate_allocation(n_flights: int, m_gates: int, conflict_prob: float, seed: int = 42) -> WCNF:
    """
    2. Airport Gate Allocation
    Hard: exactly one gate per flight. Conflicting flights cannot share a gate.
    Soft: preference cost for each gate.
    """
    rng = random.Random(seed)
    wcnf = WCNF()
    vpool = IDPool()
    
    def var(f, g):
        return vpool.id(f"x_{f}_{g}")
        
    # Hard: exactly 1 gate per flight
    for f in range(1, n_flights + 1):
        vars_f = [var(f, g) for g in range(1, m_gates + 1)]
        cnf = CardEnc.equals(lits=vars_f, bound=1, vpool=vpool, encoding=EncType.seqcounter)
        for clause in cnf.clauses:
            wcnf.append(clause)
            
    # Hard conflicts
    for f1 in range(1, n_flights + 1):
        for f2 in range(f1 + 1, n_flights + 1):
            if rng.random() < conflict_prob:
                for g in range(1, m_gates + 1):
                    # NOT both f1 and f2 at gate g
                    wcnf.append([-var(f1, g), -var(f2, g)])
                    
    # Soft: preference cost
    for f in range(1, n_flights + 1):
        for g in range(1, m_gates + 1):
            cost = rng.randint(10, 100)
            wcnf.append([-var(f, g)], weight=cost)
            
    return wcnf

def generate_shift_scheduling(n_emps: int, n_days: int, req_per_day: int, seed: int = 42) -> WCNF:
    """
    3. Employee Shift Scheduling
    Hard: exactly `req_per_day` workers each day. No employee works 3 consecutive days.
    Soft: penalty/preference cost for each assignment.
    """
    rng = random.Random(seed)
    wcnf = WCNF()
    vpool = IDPool()
    
    def var(e, d):
        return vpool.id(f"x_{e}_{d}")
        
    # Hard: exactly req workers per day
    for d in range(1, n_days + 1):
        vars_d = [var(e, d) for e in range(1, n_emps + 1)]
        cnf = CardEnc.equals(lits=vars_d, bound=req_per_day, vpool=vpool, encoding=EncType.seqcounter)
        for clause in cnf.clauses:
            wcnf.append(clause)
            
    # Hard: no 3 consecutive days
    for e in range(1, n_emps + 1):
        for d in range(1, n_days - 1):
            wcnf.append([-var(e, d), -var(e, d+1), -var(e, d+2)])
            
    # Soft: preference cost
    for e in range(1, n_emps + 1):
        for d in range(1, n_days + 1):
            cost = rng.randint(10, 100)
            wcnf.append([-var(e, d)], weight=cost)
            
    return wcnf

def main():
    print("Generating and running scheduling and allocation experiments...")
    
    # 1. Task Allocation
    n_tasks, m_machines, capacity = 6, 3, 3
    wcnf_task = generate_task_allocation(n_tasks, m_machines, capacity, seed=123)
    compare_wcnf(
        f"Task Allocation (T={n_tasks}, M={m_machines}, Cap={capacity})", 
        wcnf_task, 
        f"task_{n_tasks}_{m_machines}_{capacity}"
    )
    
    # 2. Gate Allocation
    n_flights, m_gates, conflict_prob = 6, 3, 0.3
    wcnf_gate = generate_gate_allocation(n_flights, m_gates, conflict_prob, seed=123)
    compare_wcnf(
        f"Gate Allocation (F={n_flights}, G={m_gates}, Conf={conflict_prob})", 
        wcnf_gate, 
        f"gate_{n_flights}_{m_gates}_{conflict_prob}"
    )
    
    # 3. Shift Scheduling
    n_emps, n_days, req_per_day = 4, 4, 2
    wcnf_shift = generate_shift_scheduling(n_emps, n_days, req_per_day, seed=123)
    compare_wcnf(
        f"Shift Scheduling (E={n_emps}, D={n_days}, Req={req_per_day})", 
        wcnf_shift, 
        f"shift_{n_emps}_{n_days}_{req_per_day}"
    )

if __name__ == "__main__":
    main()
