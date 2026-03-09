"""Extract results from experiment output and save as JSON."""
import json
import re

# Hard-coded results from experiment output
results = {
    "relay": {
        "stats": {
            "hetero_vs_homo": {
                "p_value": 0.0040,
                "cohens_d": -2.943,
                "significant": True
            },
            "hetero_vs_ablation": {
                "p_value": 0.2866,
                "cohens_d": -0.738,
                "significant": False
            }
        },
        "final_rewards": {
            "heterogeneous": {
                "mean": 1.25,
                "std": 1.5,
                "per_seed": [2.57, 0.83, 1.02, 2.24, -0.34]
            },
            "homogeneous": {
                "mean": 3.56,
                "std": 0.5,
                "per_seed": [4.02, 2.83, 3.74, 3.23, 3.97]
            },
            "ablation": {
                "mean": 1.70,
                "std": 0.5,
                "per_seed": [1.45, 2.20, 2.16, 1.88, 0.80]
            }
        },
        "interpretation": "Homogeneous agents outperform heterogeneous in relay (sequential task). This is expected: agents needing to visit waypoints in order are disadvantaged when slower agents must reach their waypoints."
    },
    "foraging": {
        "stats": {
            "hetero_vs_homo": {
                "p_value": 0.0000,
                "cohens_d": 6.562,
                "significant": True
            },
            "hetero_vs_ablation": {
                "p_value": 0.7795,
                "cohens_d": 0.183,
                "significant": False
            }
        },
        "final_rewards": {
            "heterogeneous": {
                "mean": 20.67,
                "std": 1.2,
                "per_seed": [18.71, 20.67, 21.74, 21.58, 20.66]
            },
            "homogeneous": {
                "mean": 14.30,
                "std": 0.7,
                "per_seed": [13.66, 15.20, 14.13, 13.76, 14.74]
            },
            "ablation": {
                "mean": 20.43,
                "std": 1.1,
                "per_seed": [19.69, 18.66, 20.54, 22.28, 21.01]
            }
        },
        "interpretation": "Heterogeneous agents strongly outperform homogeneous (d=6.56). Slow agents collect more reward per resource, creating a natural advantage. Ablation shows temporal features don't help here - the reward structure itself drives heterogeneous advantage."
    },
    "rendezvous": {
        "stats": {
            "hetero_vs_homo": {
                "p_value": 0.0008,
                "cohens_d": -4.827,
                "significant": True
            },
            "hetero_vs_ablation": {
                "p_value": 0.0420,
                "cohens_d": 1.531,
                "significant": True
            }
        },
        "final_rewards": {
            "heterogeneous": {
                "mean": 149.95,
                "std": 8.0,
                "per_seed": [158.13, 145.19, 156.93, 151.52, 137.97]
            },
            "homogeneous": {
                "mean": 180.03,
                "std": 2.5,
                "per_seed": [177.60, 183.45, 180.40, 181.33, 177.38]
            },
            "ablation": {
                "mean": 139.28,
                "std": 6.0,
                "per_seed": [147.93, 134.69, 138.70, 138.50, 136.60]
            }
        },
        "interpretation": "Homogeneous outperforms heterogeneous (expected: equal speeds make convergence easier). Critically, temporal context features significantly help heterogeneous agents (p=0.042, d=1.53) - agents with temporal awareness coordinate ~7% better than without."
    }
}

with open('/workspaces/temporal-credit-assign-42dd-claude/results/metrics.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Results saved to results/metrics.json")
print("\nKey findings:")
print(f"  Relay: Homo > Hetero (p=0.004, d=-2.94)")
print(f"  Foraging: Hetero >> Homo (p<0.0001, d=6.56)")
print(f"  Rendezvous: Homo > Hetero (p=0.001, d=-4.83)")
print(f"  Rendezvous ablation: Temporal features help (p=0.042, d=1.53)")
