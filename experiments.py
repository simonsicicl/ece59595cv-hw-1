"""
Run sweeps over hyperparameters and datasets. Produces a results CSV and console summary.

Usage (example):
  python experiments.py --dataset both --layers 0,1,2 --hidden 2,4,8 --lr 0.1,0.5,1.0 --iters 500,1000,2000 --seed 0 --test-ratio 0.25
"""
import argparse
from typing import List, Iterable
from mymlp import MLP
import datasets as dsets

def build_layer_sizes(in_dim: int, out_dim: int, layers: int, hidden: int) -> List[int]:
    sizes = [in_dim]
    for _ in range(max(0, layers)):
        sizes.append(hidden)
    sizes.append(out_dim)
    return sizes

def frange(vals: Iterable[float]) -> List[float]:
    return [float(v) for v in vals]

def run_experiments(args):
    rows = []
    datasets = ["xor", "adder"] if args.dataset == "both" else [args.dataset]
    layers_list = [int(x) for x in args.layers.split(",")]
    hidden_list = [int(x) for x in args.hidden.split(",")]
    lrs_list = frange(x.strip() for x in args.lr.split(","))
    iters_list = [int(x) for x in args.iters.split(",")]

    for ds in datasets:
        in_dim, out_dim = 1, 1
        if ds == "xor":
            X, Y = dsets.make_xor()
            in_dim, out_dim = 2, 1
        else:
            X, Y = dsets.make_two_bit_adder()
            in_dim, out_dim = 5, 3
        X_tr, Y_tr, X_te, Y_te = dsets.train_test_split(X, Y, test_ratio=args.test_ratio, seed=args.seed)

        for L in layers_list:
            for H in hidden_list:
                for lr in lrs_list:
                    for iters in iters_list:
                        layer_sizes = build_layer_sizes(in_dim, out_dim, L, H)
                        model = MLP(layer_sizes, seed=args.seed)
                        model.fit_batch(X_tr, Y_tr, lr=lr, iters=iters, print_every=0)
                        train_loss = model.evaluate_loss(X_tr, Y_tr)
                        test_loss = model.evaluate_loss(X_te, Y_te)
                        rows.append(
                            {
                                "dataset": ds,
                                "layers": L,
                                "hidden": H,
                                "lr": lr,
                                "iters": iters,
                                "seed": args.seed,
                                "train_size": X_tr.shape[0],
                                "test_size": X_te.shape[0],
                                "train_loss": train_loss,
                                "test_loss": test_loss,
                            }
                        )
                        print(f"ds={ds:<6} L={L:<2} H={H:<2} lr={lr:.2f} iters={iters:<4} | train={train_loss:.6f} test={test_loss:.6f}")
    
    print("\n====== Summary ======")
    for ds in datasets:
        best = None
        for r in rows:
            if r["dataset"] != ds:
                continue
            if best is None or r["test_loss"] < best["test_loss"]:
                best = r
        if best is not None:
            print("\nBest case for", ds, ":")
            print('-layers        = {layers}\n-hidden units  = {hidden}\n-learning rate = {lr}\n-iterations    = {iters}\n-seed          = {seed}\n-train_size    = {train_size}\n-test_size     = {test_size}\n@train_loss    = {train_loss:.6f}\n@test_loss     = {test_loss:.6f}'.format(**best))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["xor", "adder", "both"], default="both")
    p.add_argument("--layers", default="0,1,2")
    p.add_argument("--hidden", default="2,4,8")
    p.add_argument("--lr", default="0.1,0.5,1.0")
    p.add_argument("--iters", default="500,1000,2000")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--test-ratio", type=float, default=0.25)
    p.add_argument("--quick", action="store_true", help="Run a very small sweep for smoke test")
    args = p.parse_args()
    if args.quick:
        args.dataset = "both"
        args.layers = "1"
        args.hidden = "4"
        args.lr = "0.5"
        args.iters = "2000"
    run_experiments(args)

if __name__ == "__main__":
    main()
