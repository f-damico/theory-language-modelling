import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def train(model, dataloader, accumulation, criterion, optimizer, scheduler,
          epoch=None, post_update_callback=None):
    """
    Train the model for one epoch.

    Returns:
        Average loss over batches
    """
    model.train()
    optimizer.zero_grad()
    running_loss = 0.

    num_updates_epoch = len(dataloader) // accumulation
    update = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item()
        loss /= accumulation
        loss.backward()

        if ((batch_idx + 1) % accumulation == 0):
            optimizer.step()
            optimizer.zero_grad()

            update += 1

            if post_update_callback is not None:
                post_update_callback(
                    epoch=epoch,
                    update=update,
                    num_updates_epoch=num_updates_epoch,
                    batch_idx=batch_idx + 1,
                )

    scheduler.step()

    return running_loss / (batch_idx + 1)