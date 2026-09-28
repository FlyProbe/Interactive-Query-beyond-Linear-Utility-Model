import argparse
import AT
import RS

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./datasets/4d_100000.txt')
    parser.add_argument('--alg', choices=('AT', 'RS'), default='AT')
    parser.add_argument('--data_size', type=int, default=10000)
    parser.add_argument('--rounds',  type=int, default=100)
    parser.add_argument("--eps", type=float, default=0.0)
    # AT uses the selected candidate information-gain strategy.
    parser.add_argument('--dtype', choices=('float32', 'float64'), default="float32")
    parser.add_argument('--save-json', action=argparse.BooleanOptionalAction, default=True,
                        help='Export AT monitoring routes and results as JSON (default: enabled)')
    parser.add_argument('--output-dir', default=None,
                        help='Directory for AT JSON output (default: experiments/AT/run_<timestamp>)')
    # RS sampling budget.
    parser.add_argument('--N', type=int, default=5000)

    return parser



def main():
    parser = parse_args()
    args = parser.parse_args()
    if args.alg == "AT":
        AT.main(args)
    elif args.alg == "RS":
        RS.main(args)

if __name__ == "__main__":
    main()
