import warnings
warnings.filterwarnings("ignore", message="The package pingouin is out of date.")
from data.dataloading import train_test_split, load_features
import torch
from models.nn import Net
from models.linear import LinearLayer
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from metrics.evaluation import compute_eval_metrics
from training.trainer import Trainer
import argparse
import secrets
import pandas as pd


class ConditionalRequiredAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)


def parse_from_command_line():
    feature_names = ['mean', 'var', 'std', 'skew', 'kurt']
    feature_names += ['ar1', 'ar2', 'ar3', 'ar4', 'ar5', 'ar6', 'ar7', 'ar8', 'ar9']
    models = ['linear', 'neuralnet']

    # Parse feature name from command line
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument(
        '--model_path',
        type=str,
        help=f'Specify the model path.',
        required=False  # Ensure the argument is required
    )

    parser.add_argument(
        '--feature',
        type=str,
        choices=feature_names,
        help=f'Name of feature (must be one of {feature_names})',
        required=True,
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=models,
        help=f'The model to be used for training (must be one of {models})',
        required=True,
    )
    parser.add_argument(
        '--eps',
        type=float,
        help=f'The value for epsilon (float, default=100.)',
        default=100.,
    )
    parser.add_argument(
        '--alpha',
        type=float,
        help=f'The value for alpha (float, default=1.)',
        default=1.,
    )
    parser.add_argument(
        '--width',
        type=int,
        help='Width of the neural net. Only required when model is neuralnet.',
        action=ConditionalRequiredAction,
        default=None,
    )
    parser.add_argument(
        '--train_test_split',
        type=float,
        help=f'The train/test ratio. Default = 0.75',
        default=0.75,
    )
    parser.add_argument(
        '--seed',
        type=int,
        help=f'The seeding for train/test split. If unspecified, a random seed is chosen. Must be a 30 bit integer.',
        default=None,
    )
    parser.add_argument(
        '--epochs',
        type=int,
        help=f'The number of training epochs. Default is 3000',
        default=3000,
    )

    args = parser.parse_args()

    if not args.seed:
        args.seed = secrets.randbits(30)

    # Custom logic for conditional requirement
    if args.model == 'neuralnet':
        if not args.width:
            args.width = 256
    elif args.model != 'neuralnet' and args.width:
        parser.error("--width can only be set when --model is 'neuralnet'")

    return args


if __name__ == '__main__':
    time_run_started = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

    # Parse from command line and set up model and features
    args = parse_from_command_line()

    # General hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 512
    feature_names = [args.feature]
    model = {'linear': LinearLayer, 'neuralnet': Net}
    model = model[args.model]
    compute_loss_within_batch = False  # Compute loss over entire train set if False, only within batch if True
    num_epochs = args.epochs

    # Hyperparameters
    if args.model == 'neuralnet':
        model_params = {
            'dim': 360,
            'width': args.width,
            'depth': 1,
            'nenc': 1,
        }
    elif args.model == 'linear':
        model_params = {
            'in_dim': 360,
            'out_dim': 1,
        }
    else:
        raise ValueError(f"We should not be here. model was misspecified. Got {args.model}")
    loss_params = {
        'eps': args.eps,  # default 100.
        'alpha': args.alpha,  # default 1.
    }

    print(f"Using device {device}")

    # Load data
    data = load_features("data", feature_names)
    # Train test split with deterministic RNG
    data_split = train_test_split(data, perc=args.train_test_split, seed=args.seed)
    del data

    if not args.model_path:
        # Training
        trainer = Trainer(
            model=model,
            model_params=model_params,
            loss_params=loss_params,
            labels=data_split['train']['label'],
            features=data_split['train']['features'],
            device=device,
            lr=1e-3,
            batch_size=batch_size,
            within_batch=compute_loss_within_batch,
        )
        losses = trainer.train(num_epochs)

        # Save losses to txt file
        filename = f"./logs/loss_{str(trainer.model)}_autocorr_{time_run_started}.txt"
        with open(filename, "w") as f:
            np.savetxt(f, np.array(losses))
        # Plot losses and save to figure
        filename = f"./logs/loss_{str(trainer.model)}_autocorr_{time_run_started}.png"
        plt.plot(losses)
        plt.title('Losses (averaged over batches) by epoch')
        plt.savefig(filename)
        plt.close()

        net = trainer.model

    if args.model_path:
        # Load model
        with open(args.model_path, "rb") as f:
            net = model
            net = net(**model_params)
            net.load_state_dict(torch.load(f, map_location=torch.device('cpu')))
            net.eval()

    # Evaluating model performance
    metrics = compute_eval_metrics(
        data=data_split,
        model=net,
        device=device,
        batch_size=batch_size,
        metric='euclidean',
        feature_name='+'.join(feature_names),
        create_figures=True,
    )

    print(f"Performance on training set ({int(args.train_test_split*100)}%)...")
    print(pd.Series(metrics['train']))
    print(f"Performance on validation set ({int(100-args.train_test_split*100)}%)...")
    print(pd.Series(metrics['val']))

    # Save model
    if not args.model_path:
        filename = f"outputs/model_{str(net)}"
        if args.model == 'neuralnet':
            filename += f"_WIDTH-{model_params['width']}_DEPTH-{model_params['depth']}"
        filename += f"_FEATURES-{'+'.join(feature_names)}.pt"
        torch.save(net.state_dict(), filename)

    # Done
    print(f'Finished executing.')
    if not args.model_path:
        print(f"Model saved to {filename}.")

