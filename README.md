## Installation

After cloning the repo to your local machine, change into the directory.

Then run `pip install -r requirements.txt`

Good to go!

## Running

First, run `python main.py --help`

This will print out all the options that can (or need to) be specified to run.

For instance, in order to train a linear model on the mean feature for 2000 epochs with the epsilon coefficient in the loss set to 50, run

`python main.py --feature mean --model linear --epochs 2000 --eps 50`

The script will output into two directories (which are expected to exist before running, otherwise the script will fail!):

- `outputs`: this directory is expected to exist at the same level as the main.py script. It will contain the saved model.
- `figures`: this directory is expected to exist at the same level as the main.py script. Here we save plots from the evaluation of the trained model.
