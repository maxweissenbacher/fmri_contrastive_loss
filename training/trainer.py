from data.dataloading import ourDataset
import torch
from torch.utils.data import DataLoader
from models.losses.contrastive_losses import contr_loss_simple
from tqdm import tqdm
import numpy as np
import time


class Trainer:
    def __init__(self, model, model_params, loss_params, data, device, lr, batch_size):
        self.model = model(**model_params).to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5)
        # Load data
        self.same_subject = data['same_subject']
        self.diff_subject = data['diff_subject']
        self.features = data['autocorrelation_and_variation']
        # Convert to dataset and dataloader
        dataset = ourDataset(self.features, device=device)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # Loss hyperparameters
        self.eps = loss_params['eps']
        self.alpha = loss_params['alpha']
        self.device = device

    def train(self, num_epochs):
        # Testing model output
        test_num = min(10, self.features.shape[0])
        output = self.model(self.features[:test_num].to(self.device))
        print(f'Tested successfully. self.model output has shape {output.shape}')

        # Set up for logging training metrics
        losses = []

        # Training loop
        start_time = time.time()
        pbar = tqdm(range(num_epochs))
        for _ in pbar:
            avg_loss = []  # average loss across batches
            avg_gn = []  # average gradient norm across batches
            # iterate over all data
            for (d, batch_idx) in self.dataloader:
                batch_idx = batch_idx.detach().numpy()
                # get submatrices of same and diff
                # ----------
                # TO-DO: Check if using the same_subject matrix as opposed to the same_subject_train matrix is correct!
                # ----------
                same = torch.tensor(self.same_subject[batch_idx[:, None], batch_idx[None, :]]).to(self.device)
                diff = torch.tensor(self.diff_subject[batch_idx[:, None], batch_idx[None, :]]).to(self.device)
                output = self.model.forward(d)
                loss = contr_loss_simple(output, same, diff, self.eps, self.alpha, metric='cosine')
                self.optimizer.zero_grad()
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100.)
                self.optimizer.step()
                # Remember loss and gradient norm per batch
                avg_loss.append(loss.detach().item())
                avg_gn.append(gn.detach().item())

            # Evaluate model on validation set
            # with torch.no_grad():
            #    avg_loss = []  # We hope this approximates the true loss well but likely misses lots of interactions
            #    for (d, batch_idx) in dataloader_val:
            #        batch_idx = batch_idx.detach().numpy()
            #        same = torch.tensor(same_subject[batch_idx[:, None], batch_idx[None, :]]).to(device)
            #        diff = torch.tensor(diff_subject[batch_idx[:, None], batch_idx[None, :]]).to(device)
            #        output = self.model.forward(d)
            #        loss = contr_loss_simple(output, same, diff, eps)
            #        avg_loss.append(loss.detach().item())

            #    val_loss = np.array(avg_loss).mean()

            # Update learning rate
            # self.scheduler.step(val_loss)

            # Update progress bar
            description = (
                f'Loss {np.array(avg_loss).mean():.2f} | '
                f'grad norm {np.array(avg_gn).mean():.2f} | '
                f'learning rate {self.optimizer.param_groups[0]["lr"]:.9f}'
            )
            pbar.set_description(description)

            # Logging
            losses.append(np.array(avg_loss).mean())

        end_time = time.time()

        print(
            f"Training loop ({num_epochs} epochs) executed in {end_time - start_time:.2f}s, or {(end_time - start_time) / num_epochs:.2f}s per epoch.")

        return losses

