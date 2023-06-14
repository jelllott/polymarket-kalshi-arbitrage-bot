import argparse

def main():
    p = argparse.ArgumentParser("speechkv-trim")
    p.add_argument("--pruner", required=True)
    p.add_argument("--budget", type=int, default=1024)
    args = p.parse_args()
    print(args)

if __name__ == "__main__":
    main()
