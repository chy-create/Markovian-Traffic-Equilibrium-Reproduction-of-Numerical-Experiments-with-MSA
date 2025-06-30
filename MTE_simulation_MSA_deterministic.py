import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Union
import re
import time

# ------------------------------
# Network Data Parser
# ------------------------------

def load_tntp_network(filepath: str):
    arcs = []
    t0, ba, ca, pa = [], [], [], []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('~') or line.strip() == '' or 'Init' in line:
                continue
            parts = line.strip().strip(';').split()
            if len(parts) < 6:
                continue
            i, j = int(parts[0]) - 1, int(parts[1]) - 1
            arcs.append((i, j))
            ca.append(float(parts[2]))
            t0.append(float(parts[4]))
            ba.append(float(parts[5]))
            pa.append(float(parts[6]))
    num_nodes = max(max(i, j) for i, j in arcs) + 1
    return arcs, np.array(t0), np.array(ba), np.array(ca), np.array(pa), num_nodes

def load_tntp_demand(filepath: str):
    od_demand = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
    origin = None
    for line in lines:
        if line.lower().startswith("origin"):
            origin = int(line.split()[1]) - 1
        elif ':' in line and origin is not None:
            parts = line.strip().strip(';').split(';')
            for part in parts:
                if ':' in part:
                    dest, val = part.split(':')
                    od_demand[(origin, int(dest) - 1)] = float(val)
    return od_demand

def load_tntp_flow(filepath: str, arcs: List[Tuple[int, int]]) -> np.ndarray:
    """Load benchmark flow values from SiouxFalls_flow.tntp"""
    arc_dict = {(i, j): idx for idx, (i, j) in enumerate(arcs)}
    benchmark_w = np.zeros(len(arcs))
    started = False

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '' or line.startswith('~'):
                continue
            if not started:
                if 'from' in line.lower() and 'to' in line.lower():
                    started = True
                continue
            parts = line.strip(';').split()
            if len(parts) < 3:
                continue
            try:
                i, j = int(parts[0]) - 1, int(parts[1]) - 1
                vol = float(parts[2])
                if (i, j) in arc_dict:
                    benchmark_w[arc_dict[(i, j)]] = vol
            except ValueError:
                continue

    return benchmark_w


# ------------------------------
# Network Class (Modified for Sioux Falls)
# ------------------------------

class Network:
    def __init__(self, num_nodes: int, arcs: List[Tuple[int, int]], 
                 t0: np.ndarray, ba: np.ndarray, ca: np.ndarray, pa: np.ndarray, 
                 od_demand: Dict[Tuple[int, int], float]):
        self.num_nodes = num_nodes
        self.arcs = arcs
        self.arc_indices = {(i, j): idx for idx, (i, j) in enumerate(arcs)}
        self.t0 = t0
        self.ba = ba
        self.ca = ca
        self.pa = pa
        self.od_demand = od_demand

        # Build forward/backward star
        self.A_plus = {i: [] for i in range(num_nodes)}
        self.A_minus = {j: [] for j in range(num_nodes)}
        for idx, (i, j) in enumerate(arcs):
            self.A_plus[i].append(idx)
            self.A_minus[j].append(idx)

    def sa(self, w: np.ndarray) -> np.ndarray:
        """BPR link performance function"""
        return self.t0 * (1 + self.ba * (w / self.ca)**self.pa)

    def sa_inv(self, t: np.ndarray) -> np.ndarray:
        """Inverse BPR function"""
        return self.ca * (((t / self.t0) - 1) / self.ba)**(1/self.pa)

# ------------------------------
# Deterministic Model Functions
# ------------------------------

def phi(z: np.ndarray) -> float:
    """Deterministic choice: return minimum cost"""
    z = np.array(z)
    return np.min(z)

def d_phi(z: np.ndarray) -> np.ndarray:
    """Deterministic gradient: 1 at argmin, 0 elsewhere"""
    z = np.array(z)
    grad = np.zeros_like(z)
    min_indices = np.where(z == np.min(z))[0]
    # If multiple minima, evenly split gradient
    grad[min_indices] = 1.0 / len(min_indices)
    return grad

# ------------------------------
# Fixed-point Iteration for Tau
# ------------------------------

def fixed_point_tau(network: Network, t: np.ndarray, destination: int, 
                   beta: float = 0.5, max_iter: int = 1000, tol: float = 1e-6) -> np.ndarray:
    """Compute expected minimum cost tau via fixed-point iteration"""
    num_nodes = network.num_nodes
    tau = np.zeros(num_nodes)
    
    for _ in range(max_iter):
        tau_new = np.zeros_like(tau)
        for i in range(num_nodes):
            if i == destination:
                continue
            z = [t[arc_idx] + tau[j] 
                 for arc_idx in network.A_plus[i] 
                 for _, j in [network.arcs[arc_idx]]]
            if z:
                tau_new[i] = phi(z)
        
        if np.max(np.abs(tau_new - tau)) < tol:
            break
        tau = tau_new
    
    return tau

# ------------------------------
# Flow Computation
# ------------------------------

def compute_vd(network: Network, t: np.ndarray, tau: np.ndarray, 
              destination: int, beta: float = 0.5) -> np.ndarray:
    """Compute equilibrium flows for given destination"""
    num_arcs = len(network.arcs)
    num_nodes = network.num_nodes
    vd = np.zeros(num_arcs)
    
    # Compute trip generation
    g = np.zeros(num_nodes)
    for (o, d), val in network.od_demand.items():
        if d == destination:
            g[o] += val
    
    # Compute choice probabilities
    P = np.zeros((num_nodes, num_nodes))
    Q = np.zeros((num_nodes, num_arcs))  # Q[i, a]: probability node i chooses arc a
    for i in range(num_nodes):
        if i == destination:
            continue
        z = [t[arc_idx] + tau[j] 
             for arc_idx in network.A_plus[i] 
             for _, j in [network.arcs[arc_idx]]]
        if z:
            prob = d_phi(z)
            for arc_idx, p in zip(network.A_plus[i], prob):
                _, j = network.arcs[arc_idx]
                P[i, j] += p
                Q[i, arc_idx] = p
    
    # Solve linear system for node flows
    I = np.eye(num_nodes)
    x = np.linalg.solve((I - P.T), g)
    
    vd = Q.T @ x
    
    return vd

# ------------------------------
# Main Experiment with Sioux Falls
# ------------------------------

def run_sioux_falls_convergence_log():
    # Load data from TNTP files
    arcs, t0, ba, ca, pa, num_nodes = load_tntp_network("SiouxFalls_net.tntp")
    od_demand = load_tntp_demand("SiouxFalls_trips.tntp")
    
    # Create network (Sioux Falls has 24 nodes)
    net = Network(
    num_nodes=num_nodes,
    arcs=arcs,
    t0=t0,
    ba=ba,
    ca=ca,
    pa=pa,
    od_demand=od_demand
    )

    beta = 0.4 #0.5
    num_arcs = len(net.arcs)
    t = net.sa(np.ones(num_arcs))
    w = net.sa_inv(t)

    log_gap_list = []

    for k in range(1, 241): #241
        vd_all = np.zeros(num_arcs)
        for d in range(num_nodes):
            tau_d = fixed_point_tau(net, t, destination=d, beta=beta)
            vd_d = compute_vd(net, t, tau_d, destination=d, beta=beta)
            vd_all += vd_d
        w_new = vd_all
        w_prev = w.copy()
        gap = np.linalg.norm(w_new - w_prev)
        log_gap = np.log10(gap)
        log_gap_list.append(log_gap)
        step_size = 1 / (k + 1)#max(1 / (k + 1), 0.125)
        w = (1 - step_size) * w + step_size * w_new
        t = net.sa(w)
        if log_gap < -9:
            print(f'Reaches 1e-9 at {k}')

    # Plot log-scale convergence
    plt.figure(figsize=(8, 5))
    plt.plot(log_gap_list, marker='o', linewidth=0.5, markersize=0.7)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel(r"$\log_{10} \left\| \hat{w}^{(k)} - w^{(k)} \right\|_2$", fontsize=12)
    plt.title("Convergence of MTE (beta = 0.5)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("mte_convergence_MSA_deterministic.png", dpi=300)
    plt.show()
    print("Simulation completed.")
    
    return w

if __name__ == "__main__":
    print("Starting MTE simulation on Sioux Falls network")
    start_time = time.time()
    w = run_sioux_falls_convergence_log()
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.2f} seconds")
    # print(w)
    print("Simulation completed.")

    arcs, _, _, _, _, _ = load_tntp_network("SiouxFalls_net.tntp")
    benchmark_w = load_tntp_flow("SiouxFalls_flow.tntp", arcs)
    l2_error = np.linalg.norm(w - benchmark_w)
    rel_error = l2_error / (np.linalg.norm(benchmark_w) + 1e-6)
    print(f"L2 error between computed flow and benchmark: {l2_error:.4f}")
    print(f"Relative error: {rel_error:.4%}")
