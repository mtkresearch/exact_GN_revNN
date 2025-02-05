import torch


def log_layer_weights(logger, model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.logkv(f"weight_variance/{name}", param.data.var().cpu().numpy())
            logger.logkv(f"weight_L2_norm/{name}", torch.norm(param.data).cpu().numpy())


def log_layer_losses(logger, iter, model, labels, images, layer_loss_func):
    logger.logkv("iter", iter)

    full_targets = model.reverse_verbose(labels)
    full_outputs = model.forward_verbose(images)

    # Loss for component (1) of layer output
    loss_per_layer = [
        layer_loss_func(torch.chunk(o, 2, dim=1)[0], torch.chunk(t, 2, dim=1)[0])
        for o, t in zip(full_outputs, full_targets)
        # loss_func(o, t) for o, t in zip(full_outs, model.inverse(labels)[1])
    ]
    logger.logkv(f"Input z(1)) Loss", round(loss_per_layer[0].item(), 3))
    for t, layer_loss in enumerate(loss_per_layer[1:]):
        logger.logkv(f"Layer {t+1} z(1) Loss", round(layer_loss.item(), 3))

    # Loss for component (2) of layer output
    loss_per_layer = [
        layer_loss_func(torch.chunk(o, 2, dim=1)[1], torch.chunk(t, 2, dim=1)[1])
        for o, t in zip(full_outputs, full_targets)
        # loss_func(o, t) for o, t in zip(full_outs, model.inverse(labels)[1])
    ]
    logger.logkv(f"Input z(2) Loss", round(loss_per_layer[0].item(), 3))
    for t, layer_loss in enumerate(loss_per_layer[1:]):
        logger.logkv(f"Layer {t+1} z(2)) Loss", round(layer_loss.item(), 3))

    rec_images = model.reverse(labels)
    reconstruction_loss = layer_loss_func(rec_images, images.view(images.shape[0], -1))
    logger.logkv(f"rec_loss", round(reconstruction_loss.item(), 3))
