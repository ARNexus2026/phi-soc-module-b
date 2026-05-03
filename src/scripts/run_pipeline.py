from phisoc.runner import run_experiment
import argparse

def main():
    parser = argparse.ArgumentParser(description="Pipeline runner")

    parser.add_argument("--preset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi-seeds", type=int, default=None)

    args = parser.parse_args()

    if args.multi_seeds is not None:
        from phisoc.runner import run_multiple_seeds
        resultado = run_multiple_seeds(
            preset=args.preset,
            n_seeds=args.multi_seeds,
            base_seed=args.seed
    )
    else:
        resultado = run_experiment(
            preset=args.preset,
            seed=args.seed
    )

    print("\nResultado final:")
    print(resultado)


if __name__ == "__main__":
    main()