def setup_optimizer(optim_cfg, model):
    """
    Setup the optimizer. Return the optimizer.
    Only includes parameters with requires_grad=True to avoid wasting memory
    on frozen parameters (e.g. DINO backbone).
    """
    from torch import optim

    optimizer = eval(optim_cfg.type)
    model_trainable_params = get_named_trainable_params(model)
    trainable_params = [p for (name, p) in model_trainable_params]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)
    frozen_count = total_params - trainable_count
    print(
        f"Trainable parameters: {trainable_count / 1e6:.1f}M"
        f" (frozen: {frozen_count / 1e6:.1f}M, total: {total_params / 1e6:.1f}M)",
    )
    return optimizer(trainable_params, **optim_cfg.params)


def setup_lr_scheduler(optimizer, scheduler_cfg):
    import torch.optim as optim
    import torch.optim.lr_scheduler as lr_scheduler
    from .lr_scheduler import CosineAnnealingLRWithWarmup
    from torch.optim.lr_scheduler import CosineAnnealingLR

    sched = eval(scheduler_cfg.type)
    if sched is None:
        return None
    return sched(optimizer, **scheduler_cfg.params)

def get_named_trainable_params(model):
    return [
        (name, param) for name, param in model.named_parameters() if param.requires_grad
    ]