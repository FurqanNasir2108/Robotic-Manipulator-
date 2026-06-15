"""
Learning rate scheduler utilities.
"""

import torch.optim as optim


def get_scheduler(optimizer, scheduler_type='cosine', **kwargs):
    """
    Create a learning rate scheduler.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    scheduler_type : str
        'cosine', 'step', 'plateau', or 'none'.
    **kwargs
        Additional scheduler parameters.

    Returns
    -------
    scheduler or None
    """
    if scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 200),
            eta_min=kwargs.get('eta_min', 1e-6),
        )
    elif scheduler_type == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 50),
            gamma=kwargs.get('gamma', 0.5),
        )
    elif scheduler_type == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 10),
        )
    elif scheduler_type == 'none':
        return None
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")