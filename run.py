import argparse
import AT
import RS

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./datasets/4d_100000.txt')
    parser.add_argument('--alg', type=str, default='AT')
    parser.add_argument('--data_size', type=int, default=10000)
    parser.add_argument('--rounds',  type=int, default=100)
    parser.add_argument("--eps", type=float, default=0.0)
    # for AT
    parser.add_argument('--dtype',  type=str, default="float32")
    # for RS
    parser.add_argument('--N', type=int, default=5000)

    return parser



def main():
    parser = parse_args()
    args = parser.parse_args()
    if args.alg == "AT":
        AT.main(args)
    elif args.alg == "RS":
        RS.main(args)
    else:
        Exception("Unknown algorithm")

if __name__ == "__main__":
    main()