# runner.py
"""
Phi-SOC Module B — Experiment Runner
=======================================
Executa qualquer fase do artigo através de um preset (JSON) com semente fixa.
Uso como script autónomo:
  python runner.py --preset fase6 --seed 42 --output results/
Uso como biblioteca:
  from runner import run_experiment
  resultado = run_experiment(preset="fase6", seed=42)
"""

import os
import json
import argparse
import numpy as np
import torch

# NOTA: O ficheiro pipeline_completo.py deve estar no mesmo diretório
# ou ser instalado como pacote.  Aqui assumimos importação direta.
from phisoc.pipeline import *

# ----------------------------------------------------------------------
# Carregamento de presets
# ----------------------------------------------------------------------
def load_preset(name: str) -> dict:
    """Lê o ficheiro JSON de presets e devolve o dicionário com o nome dado."""
    preset_path = os.path.join(os.path.dirname(__file__), "configs", "presets.json")
    if not os.path.exists(preset_path):
        raise FileNotFoundError(f"Ficheiro de presets não encontrado em {preset_path}")
    with open(preset_path) as f:
        presets = json.load(f)
    if name not in presets:
        raise ValueError(f"Preset '{name}' desconhecido. Presets disponíveis: {list(presets.keys())}")
    return presets[name]

# ----------------------------------------------------------------------
# Pipeline principal
# ----------------------------------------------------------------------
def run_experiment(preset: str, seed: int = 42) -> dict:
    """
    Executa a experiência completa para um determinado preset e semente.
    Retorna um dicionário com todas as métricas.
    """
    config = load_preset(preset)

    # Parâmetros com defaults de publicação
    defaults = {
        "m": 2,
        "T_equil": 3000,
        "T_sample": 10000,
        "snap_interval": 200,
        "EPSILON_BLOCK": 0.01,
        "max_cycles": 50,
        "dissipative": True,
        "frustration": False,
        "tear_prob": 0.0,
        "inversion_prob": 0.3,
        "add_edge_prob": 0.0,
        "competition_strength": 0.0,
        "max_avalanche_size": None,
        "adaptive_lambda": True
    }
    params = {**defaults, **config, "seed": seed}

    print(f"\n=== Executando preset '{preset}' com seed {seed} ===")
    print("Parâmetros:", json.dumps(params, indent=2))

    # 1. Grafos e acoplamento
    graphs_raw = generate_graphs(params["sizes"], m=params["m"], seed=seed)
    G_global, node_graph, nodes_per_graph = couple_graphs(
        graphs_raw, cross_edges_per_pair=params.get("cross_edges_initial", 5), seed=seed
    )

    # 2. Simulação SOC
    snapshots, lambda_block, R0_block = run_soc_simulation(
        G_global, node_graph, nodes_per_graph,
        EPSILON_CROSS=params["EPSILON_CROSS"],
        T_equil=params["T_equil"],
        T_sample=params["T_sample"],
        snap_interval=params["snap_interval"],
        dissipative=params["dissipative"],
        adaptive_lambda=params["adaptive_lambda"],
        frustration=params["frustration"],
        tear_prob=params["tear_prob"],
        inversion_prob=params["inversion_prob"],
        add_edge_prob=params["add_edge_prob"],
        competition_strength=params["competition_strength"],
        max_avalanche_size=params["max_avalanche_size"],
        seed=seed
    )

    # 3. Ciclos e clusters (topologia de referência, fixada antes da simulação)
    cycles_list = [find_fundamental_cycles(G) for G in graphs_raw]
    clusters, all_cycles, cycle_to_graph = build_clusters(cycles_list, max_per_graph=params["max_cycles"])
    edge_maps = build_edge_maps(all_cycles)

    # 4. Tensores
    S_series, F_raw, g, F_lie, offsets = build_SOC_tensors(
        snapshots, clusters, all_cycles, cycle_to_graph, edge_maps,
        EPSILON_BLOCK=params["EPSILON_BLOCK"]
    )

    # 5. Poisson empírico
    # Normalização z‑score e PCA dinâmico (dimensão ativa da Killing)
    S_mean = S_series.mean(axis=0)
    S_std = S_series.std(axis=0) + 1e-12
    S_norm = (S_series - S_mean) / S_std
    Cov_S = np.cov(S_norm.T)
    eigvals_cov, eigvecs_cov = np.linalg.eigh(Cov_S)
    idx_cov = np.argsort(eigvals_cov)[::-1]

    # Calcula C_full para detectar dimensão ativa
    omega_full = poisson_bracket_omega(S_norm)
    C_full = estimate_derivatives_C(S_norm)
    F_full = torch.tensor(C_full, dtype=DTYPE, device=device)
    d_active, _, _ = active_dimension(F_full)

    # Projeção segura (nunca mais do que o número de componentes)
    d_proj = min(d_active, len(idx_cov), 8)   # para manter comparabilidade com o artigo
    V_pca = eigvecs_cov[:, idx_cov[:d_proj]]
    S_red = (S_norm - S_norm.mean(axis=0)) @ V_pca

    omega_red = poisson_bracket_omega(S_red)
    C_red = estimate_derivatives_C(S_red)
    jac_real = poisson_jacobi_direct(omega_red, C_red)
    jac_shuffle = shuffle_baseline(S_red, seed=seed)
    ratio = jac_real / (jac_shuffle + 1e-12)

    # 6. Classificação da álgebra
    F_tensor = torch.tensor(C_red, dtype=DTYPE, device=device)
    best, score, results, dim_alg = classify_algebra(F_tensor)

    # 7. Identificação final (retorna também as métricas textuais)
    antisym, jac_rel, dist_su3 = identify_algebra(C_red, jac_shuffle=jac_shuffle)

    # 8. Métricas adicionais
    F_tensor_red = torch.tensor(C_red, dtype=DTYPE, device=device)
    K_red = torch.einsum('acd,bcd->ab', F_tensor_red, F_tensor_red)
    eigvals_K = torch.linalg.eigvalsh(K_red).abs()
    rank_eff = (eigvals_K > 1e-6).sum().item()

    # ---- Monta o dicionário de resultados ----
    resultado = {
        "preset": preset,
        "seed": seed,
        "params": params,
        "lambda_final": lambda_block.tolist(),  # converte ndarray para lista
        "R0_final": R0_block.tolist(),
        "n_snapshots": len(snapshots),
        "n_cycles_total": len(all_cycles),
        "d_active (Poisson)": d_active,
        "d_proj (PCA)": d_proj,
        "Jacobi_real": jac_real,
        "Jacobi_shuffle": jac_shuffle,
        "ratio_J_shuffle": ratio,
        "algebra_best": best,
        "algebra_score": score,
        "algebra_scores": {k: v.get("score", v) for k, v in results.items()},
        "algebra_dim": dim_alg,
        "antisymmetry_error": antisym,
        "Jacobi_relative": jac_rel,
        "dist_SU3_GellMann": dist_su3,
        "rank_eff": rank_eff
    }

    print(f"\nResultado principal: {json.dumps({k: resultado[k] for k in ['Jacobi_real', 'algebra_best', 'algebra_dim', 'dist_SU3_GellMann']}, indent=2)}")
    return resultado
# ----------------------------------------------------------------------
# Execução com múltiplas seeds
# ----------------------------------------------------------------------
def run_multiple_seeds(preset: str, n_seeds: int = 10, base_seed: int = 42):
    """
    Executa o mesmo preset para múltiplas seeds e agrega estatísticas.
    """
    resultados = []

    print(f"\n=== MULTI-SEED RUN: {n_seeds} execuções ===")

    for i in range(n_seeds):
        seed = base_seed + i
        print(f"\n--- Seed {seed} ---")
        res = run_experiment(preset=preset, seed=seed)
        resultados.append(res)

    # -----------------------------
    # Agregação estatística
    # -----------------------------
    def collect(key):
        vals = []
        for r in resultados:
            v = r.get(key)
            if isinstance(v, (float, np.floating)):
                vals.append(float(v))
        return np.array(vals)

    summary = {}

    keys_to_aggregate = [
        "Jacobi_real",
        "Jacobi_shuffle",
        "ratio_J_shuffle",
        "Jacobi_relative",
        "dist_SU3_GellMann",
        "rank_eff",
        "d_active (Poisson)"
    ]

    for k in keys_to_aggregate:
        vals = collect(k)
        if len(vals) > 0:
            summary[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals))
            }

    print("\n=== RESUMO ESTATÍSTICO ===")
    print(json.dumps(summary, indent=2))

    return {
        "preset": preset,
        "n_seeds": n_seeds,
        "results": resultados,
        "summary": summary
    }

# ----------------------------------------------------------------------
# Interface de linha de comandos
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Phi-SOC Module B — Runner")
    parser.add_argument("--preset", type=str, required=True, help="Nome do preset (ex: fase6)")
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatória")
    parser.add_argument("--multi-seeds", type=int, default=None, help="Número de seeds para execução múltipla")
    parser.add_argument("--output", type=str, default="results", help="Directório de saída para os JSONs")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.multi_seeds is not None:
        resultado = run_multiple_seeds(
            preset=args.preset,
            n_seeds=args.multi_seeds,
            base_seed=args.seed
        )
        fname = os.path.join(args.output, f"{args.preset}_multiseed.json")

    else:
        resultado = run_experiment(
            preset=args.preset,
            seed=args.seed
        )
        fname = os.path.join(args.output, f"{args.preset}_seed{args.seed}.json")

    with open(fname, "w") as f:
        json.dump(resultado, f, indent=2)

    print(f"Resultados guardados em: {fname}")


if __name__ == "__main__":
    main()