from model import run_cl

import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('yaml_path', metavar='path', type=str, help='path to yaml file (embeddings)')
args = parser.parse_args()


def main():
    """Main method which reads a yaml file and starts the training of the model with the specified parameters.
    """
    with open(args.yaml_path) as y:
        conf = yaml.load(y.read(), Loader=yaml.FullLoader)

        run_cl(**conf["classifier"])


if __name__ == "__main__":
    main()
