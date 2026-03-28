def train(model, dataloader, accumulation, criterion, optimizer, scheduler):
    """
    Train the model for one epoch.

    Returns:
        Average loss over batches
    """
    model.train()
    optimizer.zero_grad()
    running_loss = 0.

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item()
        loss = loss / accumulation
        loss.backward()

        if (batch_idx + 1) % accumulation == 0:
            optimizer.step()
            optimizer.zero_grad()

    scheduler.step()

    return running_loss / (batch_idx + 1)