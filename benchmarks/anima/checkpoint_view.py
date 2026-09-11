"""Save the model and optimizer in the same true-weight view."""


def checkpoint_true_view(optimizer, write):
    was_training = optimizer._train_mode
    optimizer.eval()
    try:
        return write()
    finally:
        if was_training:
            optimizer.train()
