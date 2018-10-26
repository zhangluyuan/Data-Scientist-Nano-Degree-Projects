import argparse
# define commanline Argument
parser = argparse.ArgumentParser()
parser.add_argument('--a', type=int, help='a the first int')
parser.add_argument('--b', type=int, help='b the second int')

args, _ = parser.parse_known_args()

# laod and build the model
def sum(a,b):
    checkpoint = 'my_model.pth'
    print(a+b)
    print('model checkpoint is saved to ', checkpoint)
    return a+b

if args.a:
    a=args.a
    b=args.b
    sum(a,b)
