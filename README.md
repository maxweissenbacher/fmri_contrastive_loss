## Installation

After cloning the repo to your local machine, change into the directory.

Then run `pip install -r requirements.txt`

Good to go!

## Running

First, run `python main.py --help`

This will print out all the options that can (or need to) be specified to run.

For instance, in order to train a linear model on the mean feature for 2000 epochs with the epsilon coefficient in the loss set to 50, run

`python main.py --feature mean --model linear --epochs 2000 --eps 50`
