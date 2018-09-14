import sys
from utils import plot_history


def test(model_name):
    plot_history(model_name)


if __name__ == "__main__":
    model = str(sys.argv[1])
    test(model)