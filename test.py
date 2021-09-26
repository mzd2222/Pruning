import argparse

parser = argparse.ArgumentParser(description="test test  test")

# nargs=‘+’ 表示可以连续输入多个 required=True表示为必须
parser.add_argument("integers", type=str, nargs='+', required=True, default="mzd", help="__help test help test")


args = parser.parse_args()

print("recieve: ", args.integers)