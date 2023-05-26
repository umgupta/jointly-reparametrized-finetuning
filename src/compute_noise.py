"""
Invokes compute-dp-epsilon to compute the noise multiplier for a given epsilon.

Usage: python compute_noise.py --num_samples 100000 --batch_size 32 --epsilon 1.0 --epochs 10 --delta 1e-6 --lowest_noise 0.2 --highest_noise 2.0 

The above command will search for noise multiplier between lowest noise and highest noise. Other parameters are self-explanatory.
"""


import argparse
import subprocess
import sys


def run_command(sampling_probability, noise, delta, steps):
    # command = f"compute-dp-epsilon
    result = subprocess.check_output(
        f"compute-dp-epsilon --sampling-probability {sampling_probability} --noise-multiplier {noise} --delta {delta} --num-compositions {steps}",
        shell=True,
        stderr=subprocess.PIPE,
    )
    # print(str(result))
    # print(str(result).split("\\n")[-3].strip().split(" "))
    return float(str(result).split("\\n")[-3].strip().split(" ")[-1])


if __name__ == "__main__":
    EPS = 1e-6
    parser = argparse.ArgumentParser()

    parser.add_argument("-N", "--num_samples", required=True, type=int)
    parser.add_argument("-B", "--batch_size", required=True, type=int)
    parser.add_argument("-D", "--delta", required=False, type=float)
    parser.add_argument("--epsilon", required=True, type=float)
    parser.add_argument("-E", "--epochs", required=True, type=int)

    parser.add_argument("--lowest_noise", required=False, type=float, default=0.2)
    parser.add_argument("--highest_noise", required=False, type=float, default=2.0)

    args = parser.parse_args()

    if args.delta is None:
        args.delta = 1 / (2 * args.num_samples)
    sampling_probability = args.batch_size / args.num_samples
    num_compositions = args.epochs * (args.num_samples // args.batch_size)

    print(sampling_probability, num_compositions, args.delta)

    # binary search
    low_noise = args.lowest_noise
    high_noise = args.highest_noise

    high_eps = run_command(sampling_probability, low_noise, args.delta, num_compositions)
    low_eps = run_command(sampling_probability, high_noise, args.delta, num_compositions)

    if high_eps < args.epsilon:
        sys.exit("Use a lower seed noise")

    if low_eps > args.epsilon:
        sys.exit("Use a higher seed noise")

    while high_noise - low_noise > EPS:
        print(f"noise diff {high_noise-low_noise}")
        mid_noise = (low_noise + high_noise) / 2

        mid_eps = run_command(sampling_probability, mid_noise, args.delta, num_compositions)

        if mid_eps < args.epsilon:
            high_noise = mid_noise
            low_eps = mid_eps
        else:
            low_noise = mid_noise
            high_eps = mid_eps

        print(f"mid eps  is {mid_eps}")
    print(f"final noise is {mid_noise}")
