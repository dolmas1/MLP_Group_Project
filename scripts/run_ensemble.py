from ensembles import run_ensemble

import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('yaml_path', metavar='path', type=str, help='path to yaml file (embeddings)')
args = parser.parse_args()


def main():
    """Main method which reads a yaml file and builds/evaluates an ensemble using the specified parameters.
    """
    with open(args.yaml_path) as y:
        conf = yaml.load(y.read(), Loader=yaml.FullLoader)

        run_ensemble(**conf)


if __name__ == "__main__":
    main()
