import argparse
import sympy
from sympy import lambdify
import numpy as np


def generate_data(expression_str, range_dict, num_samples):
    expr = sympy.sympify(expression_str)
    symbols_list = sorted(expr.free_symbols, key=lambda s: s.name)
    f = lambdify(symbols_list, expr, 'numpy')
    covs = []
    for sym in symbols_list:
        if str(sym) not in range_dict:
            range_dict[str(sym)] = (0, 1)
        covs.append(np.random.uniform(range_dict[str(sym)][0], range_dict[str(sym)][1], num_samples).reshape(-1, 1))
    X = np.hstack(covs)
    y = f(*[X[:, i] for i in range(X.shape[1])])
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Generate data from a mathematical expression.")
    parser.add_argument("--exp", type=str, required=True, help="Mathematical expression to generate data for.")
    parser.add_argument("-n", type=int, required=True, help="Number of samples to generate.")
    parser.add_argument("--ranges", nargs='+', action='append',
                        help="Ranges for variables in the format: -var_name min max. This option can be repeated.")
    parser.add_argument("-s", "--save", type=str, help="Path to save the data.", default="data/data.npz")

    args = parser.parse_args()

    # Process ranges into a dictionary
    range_dict = {}
    for range_arg in args.ranges:
        var_name, min_val, max_val = range_arg
        range_dict[var_name] = (float(min_val), float(max_val))

    # Generate data
    X, y = generate_data(args.exp, range_dict, args.n)

    # For demonstration, print the generated data
    print("Generate covariates (X) of shape", X.shape)
    print("Generate response (y) of shape:", y.shape)

    np.savez(args.save, X=X, y=y)
    print("Saving data to:", args.save)


if __name__ == "__main__":
    main()
