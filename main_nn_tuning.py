from tuning.nn_robustness import nn_tuning
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NN tuning for single features')
    parser.add_argument('feature_name', type=str, help='Name of feature')
    args = parser.parse_args()
    name = args.feature_name

    nn_tuning(feature_names=[name])

print('Done with tuning.')
