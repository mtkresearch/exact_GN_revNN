from ray import train as ray_train
import time
from tqdm import tqdm
import torch
import numpy as np
import sys, os
from .layer_losses import log_layer_losses, log_layer_weights
from fastbreak import MSELoss, compute_accuracy


def train(
    model,
    train_loader,
    val_loader,
    loss_func,
    optimizer,
    scheduler,
    device,
    num_epochs,
    sgd_per_layer=False,
    labeller=None,
    classification=False,
    steps_for_printing=1,
    save_checkpoint_every=False,
    logger=None,
    train_ggn=True,
    layer_loss_func=MSELoss(),
    added_random_in_output=False,
    random_proj_output=False,
    rand_proj_matrix=None,
    log_test=True,
    log_full_train=True,
    tuning=False,
    weight_folder="weights",
    exp_name="test",
    opt_name="GGN"
) -> None:
    model.train()
    best_val_loss = float("inf")
    last_saved_checkpoint = None

    # Save the new model checkpoint
    checkpoint_path = os.path.join(weight_folder, f"{opt_name}_{exp_name}_epoch000.pt")
    # if train_ggn:
    #     torch.save(model.state_dict(), checkpoint_path)
    # else:
    #     ckpt = {"model": model.state_dict(),
    #             "optimizer": optimizer.state_dict()}
    #     torch.save(ckpt, checkpoint_path)


    for epoch in range(num_epochs):
        if logger:
            logger.log(f"Epoch {epoch + 1}/{num_epochs}")

        start_time, test_time, start_time_iter, test_time_iter = (
            time.time(),
            0,
            time.time(),
            0,
        )
        running_loss, running_acc = 0, 0
        with tqdm(train_loader, unit="batch") as tepoch:
            for i, (images, labels) in enumerate(tepoch):
                logger.logkv("epoch", epoch + 1)
                logger.logkv("iter", epoch * len(tepoch) + i + 1)
                images, labels = images.to(device), labels.to(device)

                if epoch == 0 and i == 0:
                    # log_layer_losses(
                    #     logger,
                    #     0,
                    #     model,
                    #     labels,
                    #     images,
                    #     layer_loss_func,
                    # )
                    log_layer_weights(logger, model)

                optimizer.zero_grad()

                if labeller:
                    labels = labeller(images)

                if train_ggn:
                    with torch.no_grad():
                        output = model(images)
                else:
                    output = model(images)

                loss = loss_func(output, labels, model.u_proj)

                if train_ggn:
                    with torch.no_grad():
                        model.backward(output, labels, logger)
                else:
                    loss.backward()

                optimizer.step()

                test_time_tmp = time.time()

                if scheduler:
                    scheduler.step()

                if classification:
                    if random_proj_output:
                        distances = torch.cdist(output, rand_proj_matrix)
                        _, pred_idx = torch.min(distances, dim=1)

                        distances = torch.cdist(labels, rand_proj_matrix)
                        _, label_idx = torch.min(distances, dim=1)

                        acc = 100 * (pred_idx == label_idx).sum() / pred_idx.shape[0]
                    else:
                        acc = compute_accuracy(output, labels, u_proj=model.u_proj)

                output_after_step = model(images)
                loss_after_step = loss_func(output_after_step, labels, model.u_proj)
                delta_loss = 100 * (loss_after_step - loss) / loss

                if logger:
                    running_loss += loss.item()
                    logger.logkv_mean("train_loss", loss.item())
                    logger.logkv_mean(
                        "Percentage loss change per batch", delta_loss.item()
                    )
                    if classification:
                        running_acc += acc.item()
                        logger.logkv_mean("train_acc", acc.item())

                if (i + 1) % steps_for_printing == 0:
                    if classification:
                        tepoch.set_postfix(loss=loss.item(), acc=acc.item())
                    else:
                        tepoch.set_postfix(loss=loss.item())

                if (
                    logger
                    and (i + 1) % save_checkpoint_every == 0
                    or (i == 0 and epoch == 0)
                ):
                    if not (i == 0 and epoch == 0):
                        # log_layer_losses(
                        #     logger,
                        #     epoch * len(tepoch) + i,
                        #     model,
                        #     labels,
                        #     images,
                        #     layer_loss_func,
                        # )
                        log_layer_weights(logger, model)
                    if log_test:
                        val_loss, val_acc = test(
                            model,
                            val_loader,
                            loss_func,
                            device,
                            labeller=None,
                            output_per_layer=True,
                            classification=classification,
                            logger=logger,
                            added_random_in_output=added_random_in_output,
                            random_proj_output=random_proj_output,
                            rand_proj_matrix=rand_proj_matrix,
                        )
                        # if val_loss < best_val_loss:
                        #     best_val_loss = val_loss
                        #     # Delete the old checkpoint if it exists
                        #     if last_saved_checkpoint is not None:
                        #         os.remove(last_saved_checkpoint)
                        #     # Save the new model checkpoint
                        #     checkpoint_path = f"model_checkpoint_epoch{epoch:03d}.pt"
                        #     torch.save(model.state_dict(), checkpoint_path)
                        #     # Update the last saved checkpoint path
                        #     last_saved_checkpoint = checkpoint_path

                    test_time_iter += time.time() - test_time_tmp
                    iter_time = time.time() - start_time_iter - test_time_iter
                    logger.logkv("time", iter_time)
                    logger.logkv(
                        "max_memory_mb", torch.cuda.max_memory_allocated() / 1024**2
                    )
                    if log_test:
                        logger.logkv("val_loss", val_loss)
                        if classification:
                            logger.logkv("val_acc", val_acc)
                    logger.dumpkvs()
                    start_time_iter, test_time_iter = time.time(), 0
                else:
                    test_time_iter += time.time() - test_time_tmp
                    test_time += time.time() - test_time_tmp

            # Save the new model checkpoint
            checkpoint_path = os.path.join(weight_folder, f"{opt_name}_{exp_name}_epoch{epoch+1:03d}.pt")
            # if train_ggn:
            #     torch.save(model.state_dict(), checkpoint_path)
            # else:
            #     ckpt = {"model": model.state_dict(),
            #             "optimizer": optimizer.state_dict()}
            #     torch.save(ckpt, checkpoint_path)
            #
            with logger.scoped_configure(
                os.path.join(logger.get_dir(), "per_epoch"),
                format_strs=["stdout", "csv"],
            ):
                logger.logkv("epoch", epoch)
                test_start_time = time.time()
                if log_full_train:
                    full_train_loss, full_train_acc = test(
                        model,
                        train_loader,
                        loss_func,
                        device,
                        labeller=None,
                        output_per_layer=True,
                        classification=classification,
                        logger=logger,
                        added_random_in_output=added_random_in_output,
                        random_proj_output=random_proj_output,
                        rand_proj_matrix=rand_proj_matrix,
                    )
                    logger.logkv("train_loss", full_train_loss)
                    if classification:
                        logger.logkv("train_acc", full_train_acc)
                else:
                    logger.logkv("train_loss", running_loss / len(train_loader))
                    if classification:
                        logger.logkv("train_acc", running_acc / len(train_loader))

                if log_test:
                    val_loss, val_acc = test(
                        model,
                        val_loader,
                        loss_func,
                        device,
                        labeller=None,
                        output_per_layer=True,
                        classification=classification,
                        logger=logger,
                        added_random_in_output=added_random_in_output,
                        random_proj_output=random_proj_output,
                        rand_proj_matrix=rand_proj_matrix,
                    )
                    if classification:
                        print(
                            f"Validation Loss: {val_loss:.2f}\nValidation Accuracy {val_acc:.2f} %"
                        )
                    else:
                        print(f"Validation Loss: {val_loss:.2f}")

                test_time += time.time() - test_start_time

                epoch_time = time.time() - start_time - test_time
                logger.logkv("time", epoch_time)
                logger.logkv(
                    "max_memory_mb", torch.cuda.max_memory_allocated() / 1024**2
                )
                if log_test:
                    logger.logkv("val_loss", val_loss)
                    if classification:
                        logger.logkv("val_acc", val_acc)
                logger.dumpkvs()

                if tuning:
                    ray_train.report({"mean_accuracy": val_acc})


def test(
    model,
    val_loader,
    loss_func,
    device,
    labeller=None,
    output_per_layer=True,
    logger=None,
    classification=False,
    added_random_in_output=False,
    random_proj_output=False,
    rand_proj_matrix=None,
) -> None:
    model.eval()

    running_acc, running_loss = 0, 0
    with tqdm(val_loader, unit="batch", disable=True) as tepoch:
        for i, (images, labels) in enumerate(tepoch):
            block_size = labels[0].sum()
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                output = model(images)

            loss = loss_func(output, labels, model.u_proj)

            if classification:
                if random_proj_output:
                    inv_proj = torch.pinverse(rand_proj_matrix)
                    distances = torch.cdist(output, rand_proj_matrix)
                    _, pred_idx = torch.min(distances, dim=1)
                    # print(pred_idx)

                    distances = torch.cdist(labels, rand_proj_matrix)
                    _, label_idx = torch.min(distances, dim=1)
                    # print(label_idx)

                    acc = 100 * (pred_idx == label_idx).sum() / pred_idx.shape[0]
                else:
                    output_proj = (
                        output @ model.u_proj if model.u_proj is not None else output
                    )
                    acc = (
                        100
                        * (labels.argmax(dim=1) == output_proj.argmax(dim=1)).sum()
                        / labels.shape[0]
                    )
                running_acc += acc.item()
            running_loss += loss.item()

    model.train()
    return running_loss / len(val_loader), running_acc / len(val_loader)
