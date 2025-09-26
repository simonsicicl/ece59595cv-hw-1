"""
CLI to train a single MLP configuration on a specified dataset and report train/test loss.

Usage (example):
  python train.py --dataset xor --layers 2 --hidden 4 --lr 0.5 --iters 2000 --seed 0 --test-ratio 0.25 --print-every 100
"""
import argparse
from mymlp import MLP
import datasets as dsets


def build_layer_sizes(in_dim: int, out_dim: int, layers: int, hidden: int) -> list[int]:
    sizes = [in_dim]
    for _ in range(max(0, layers)):
        sizes.append(hidden)
    sizes.append(out_dim)
    return sizes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["xor", "adder"], default="xor")
    p.add_argument("--layers", type=int, default=1, help="number of hidden layers")
    p.add_argument("--hidden", type=int, default=4, help="hidden units per layer")
    p.add_argument("--lr", type=float, default=0.5, help="learning rate")
    p.add_argument("--iters", type=int, default=2000, help="training iterations")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--test-ratio", type=float, default=0.25)
    p.add_argument("--print-every", type=int, default=0)
    args = p.parse_args()

    if args.dataset == "xor":
        X, Y = dsets.make_xor()
        in_dim, out_dim = 2, 1
    else:
        X, Y = dsets.make_two_bit_adder()
        in_dim, out_dim = 5, 3

    X_tr, Y_tr, X_te, Y_te = dsets.train_test_split(X, Y, test_ratio=args.test_ratio, seed=args.seed)

    layer_sizes = build_layer_sizes(in_dim, out_dim, args.layers, args.hidden)
    model = MLP(layer_sizes, seed=args.seed)

    model.fit_batch(X_tr, Y_tr, lr=args.lr, iters=args.iters, print_every=args.print_every)

    train_loss = model.evaluate_loss(X_tr, Y_tr)
    test_loss = model.evaluate_loss(X_te, Y_te)

    print("=== Results ===")
    print(f"dataset={args.dataset} layers={args.layers} hidden={args.hidden} lr={args.lr} iters={args.iters} seed={args.seed}")
    print(f"train_size={X_tr.shape[0]} test_size={X_te.shape[0]}")
    print(f"train_loss={train_loss:.6f}")
    print(f"test_loss={test_loss:.6f}")


if __name__ == "__main__":
    main()
