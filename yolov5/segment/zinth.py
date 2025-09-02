# Ultralytics YOLOv5 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Train a YOLOv5 segment model with TensorBoard visualization
"""

import matplotlib
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.optim import lr_scheduler
import yaml
import torch.nn as nn
import torch.distributed as dist
import torch
import numpy as np
from datetime import datetime
from copy import deepcopy
import time
import subprocess
import random
import math
import argparse
import segment.val as validate  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import SegmentationModel
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.downloads import attempt_download, is_url
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    check_amp,
    check_dataset,
    check_file,
    check_git_info,
    check_git_status,
    check_img_size,
    check_requirements,
    check_suffix,
    check_yaml,
    colorstr,
    get_latest_run,
    increment_path,
    init_seeds,
    intersect_dicts,
    labels_to_class_weights,
    labels_to_image_weights,
    one_cycle,
    print_args,
    print_mutation,
    strip_optimizer,
    yaml_save,
)
from utils.loggers import GenericLogger
from utils.plots import plot_evolve, plot_labels
from utils.segment.dataloaders import create_dataloader
from utils.segment.loss import ComputeLoss
from utils.segment.metrics import KEYS, fitness
from utils.segment.plots import plot_images_and_masks, plot_results_with_masks
from utils.torch_utils import (
    EarlyStopping,
    ModelEMA,
    de_parallel,
    select_device,
    smart_DDP,
    smart_optimizer,
    smart_resume,
    torch_distributed_zero_first,
)
import sys
import os
from pathlib import Path

# Add YOLOv5 root directory to Python path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# Now import YOLOv5 modules

# Standard imports


# TensorBoard imports
matplotlib.use('Agg')  # Use non-interactive backend

# Environment variables
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
GIT_INFO = check_git_info()


def plot_precision_recall_curve(precision, recall, ap, save_path=None):
    """Plot precision-recall curve and return figure."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(recall, precision, linewidth=2, label=f'mAP@0.5 = {ap:.3f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True)
    ax.legend()
    ax.set_title('Precision-Recall Curve')

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def train(hyp, opt, device, callbacks):
    """
    Trains the YOLOv5 model on a dataset with TensorBoard logging.
    """
    (
        save_dir,
        epochs,
        batch_size,
        weights,
        single_cls,
        evolve,
        data,
        cfg,
        resume,
        noval,
        nosave,
        workers,
        freeze,
        mask_ratio,
    ) = (
        Path(opt.save_dir),
        opt.epochs,
        opt.batch_size,
        opt.weights,
        opt.single_cls,
        opt.evolve,
        opt.data,
        opt.cfg,
        opt.resume,
        opt.noval,
        opt.nosave,
        opt.workers,
        opt.freeze,
        opt.mask_ratio,
    )

    # Initialize TensorBoard writer
    tb_writer = None
    if RANK in {-1, 0}:
        tb_writer = SummaryWriter(log_dir=save_dir / 'tensorboard')
        LOGGER.info(
            f"TensorBoard: Start with 'tensorboard --logdir {save_dir / 'tensorboard'}', view at http://localhost:6006/")

    # Directories
    w = save_dir / "weights"  # weights dir
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / "last.pt", w / "best.pt"

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr("hyperparameters: ") +
                ", ".join(f"{k}={v}" for k, v in hyp.items()))
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints

    # Save run settings
    if not evolve:
        yaml_save(save_dir / "hyp.yaml", hyp)
        yaml_save(save_dir / "opt.yaml", vars(opt))

    # Loggers
    data_dict = None
    if RANK in {-1, 0}:
        logger = GenericLogger(opt=opt, console_logger=LOGGER)

    # Config
    plots = not evolve and not opt.noplots  # create plots
    overlap = not opt.no_overlap
    cuda = device.type != "cpu"
    init_seeds(opt.seed + 1 + RANK, deterministic=True)

    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)  # check if None
    train_path, val_path = data_dict["train"], data_dict["val"]
    nc = 1 if single_cls else int(data_dict["nc"])  # number of classes
    names = {0: "item"} if single_cls and len(
        data_dict["names"]) != 1 else data_dict["names"]  # class names
    is_coco = isinstance(val_path, str) and val_path.endswith(
        "coco/val2017.txt")  # COCO dataset

    # Model
    check_suffix(weights, ".pt")  # check weights
    pretrained = weights.endswith(".pt")
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            # download if not found locally
            weights = attempt_download(weights)
        # load checkpoint to CPU to avoid CUDA memory leak
        ckpt = torch.load(weights, map_location="cpu")
        model = SegmentationModel(
            cfg or ckpt["model"].yaml, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)
        exclude = ["anchor"] if (cfg or hyp.get(
            "anchors")) and not resume else []  # exclude keys
        # checkpoint state_dict as FP32
        csd = ckpt["model"].float().state_dict()
        csd = intersect_dicts(csd, model.state_dict(),
                              exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        LOGGER.info(
            f"Transferred {len(csd)}/{len(model.state_dict())} items from {weights}")  # report
    else:
        model = SegmentationModel(
            cfg, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # create

    amp = check_amp(model)  # check AMP

    # Freeze layers
    freeze = [f"model.{x}." for x in (freeze if len(
        freeze) > 1 else range(freeze[0]))]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            LOGGER.info(f"freezing {k}")
            v.requires_grad = False

    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    # verify imgsz is gs-multiple
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)

    # Batch size
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(model, imgsz, amp)
        logger.update_params({"batch_size": batch_size})

    # Optimizer
    nbs = 64  # nominal batch size
    # accumulate loss before optimizing
    accumulate = max(round(nbs / batch_size), 1)
    hyp["weight_decay"] *= batch_size * accumulate / nbs  # scale weight_decay
    optimizer = smart_optimizer(
        model, opt.optimizer, hyp["lr0"], hyp["momentum"], hyp["weight_decay"])

    # Scheduler
    if opt.cos_lr:
        lf = one_cycle(1, hyp["lrf"], epochs)  # cosine 1->hyp['lrf']
    else:
        def lf(x):
            """Linear learning rate scheduler decreasing from 1 to hyp['lrf'] over 'epochs'."""
            return (1 - x / epochs) * (1.0 - hyp["lrf"]) + hyp["lrf"]  # linear

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # EMA
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # Resume
    best_fitness, start_epoch = 0.0, 0
    if pretrained:
        if resume:
            best_fitness, start_epoch, epochs = smart_resume(
                ckpt, optimizer, ema, weights, epochs, resume)
        del ckpt, csd

    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning(
            "WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.")
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info("Using SyncBatchNorm()")

    # Trainloader
    train_loader, dataset = create_dataloader(
        train_path,
        imgsz,
        batch_size // WORLD_SIZE,
        gs,
        single_cls,
        hyp=hyp,
        augment=True,
        cache=None if opt.cache == "val" else opt.cache,
        rect=opt.rect,
        rank=LOCAL_RANK,
        workers=workers,
        image_weights=opt.image_weights,
        quad=opt.quad,
        prefix=colorstr("train: "),
        shuffle=True,
        mask_downsample_ratio=mask_ratio,
        overlap_mask=overlap,
    )

    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())  # max label class
    assert mlc < nc, f"Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}"

    # Process 0
    if RANK in {-1, 0}:
        val_loader = create_dataloader(
            val_path,
            imgsz,
            batch_size // WORLD_SIZE * 2,
            gs,
            single_cls,
            hyp=hyp,
            cache=None if noval else opt.cache,
            rect=True,
            rank=-1,
            workers=workers * 2,
            pad=0.5,
            mask_downsample_ratio=mask_ratio,
            overlap_mask=overlap,
            prefix=colorstr("val: "),
        )[0]

        if not resume:
            if not opt.noautoanchor:
                # run AutoAnchor
                check_anchors(dataset, model=model,
                              thr=hyp["anchor_t"], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision

            if plots:
                plot_labels(labels, names, save_dir)

    # DDP mode
    if cuda and RANK != -1:
        model = smart_DDP(model)

    # Model attributes
    # number of detection layers (to scale hyps)
    nl = de_parallel(model).model[-1].nl
    hyp["box"] *= 3 / nl  # scale to layers
    hyp["cls"] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp["obj"] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp["label_smoothing"] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(
        dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nb = len(train_loader)  # number of batches
    # number of warmup iterations
    nw = max(round(hyp["warmup_epochs"] * nb), 100)
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    results = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    stopper, stop = EarlyStopping(patience=opt.patience), False
    compute_loss = ComputeLoss(model, overlap=overlap)  # init loss class

    # TensorBoard: Log model architecture
    if tb_writer and RANK in {-1, 0}:
        try:
            dummy_input = torch.randn(1, 3, imgsz, imgsz).to(device)
            tb_writer.add_graph(model, dummy_input)
        except Exception as e:
            LOGGER.warning(f"TensorBoard model graph logging failed: {e}")

    LOGGER.info(f"Image sizes {imgsz} train, {imgsz} val\n"
                f"Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n"
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f"Starting training for {epochs} epochs...")

    # Training loop
    for epoch in range(start_epoch, epochs):
        model.train()

        # Update image weights (optional, single-GPU only)
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
            iw = labels_to_image_weights(
                dataset.labels, nc=nc, class_weights=cw)  # image weights
            dataset.indices = random.choices(
                range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

        mloss = torch.zeros(4, device=device)  # mean losses
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(("\n" + "%11s" * 8) % ("Epoch", "GPU_mem", "box_loss",
                    "seg_loss", "obj_loss", "cls_loss", "Instances", "Size"))
        if RANK in {-1, 0}:
            # progress bar
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)
        optimizer.zero_grad()

        # Track batch losses for TensorBoard
        batch_box_loss = []
        batch_seg_loss = []
        batch_obj_loss = []
        batch_cls_loss = []

        for i, (imgs, targets, paths, _, masks) in pbar:  # batch
            # number integrated batches (since train start)
            ni = i + nb * epoch
            imgs = imgs.to(device, non_blocking=True).float() / \
                255  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                accumulate = max(1, np.interp(
                    ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    x["lr"] = np.interp(
                        ni, xi, [hyp["warmup_bias_lr"] if j == 0 else 0.0, x["initial_lr"] * lf(epoch)])
                    if "momentum" in x:
                        x["momentum"] = np.interp(
                            ni, xi, [hyp["warmup_momentum"], hyp["momentum"]])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(
                    int(imgsz * 0.5), int(imgsz * 1.5) + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) *
                          gs for x in imgs.shape[2:]]  # new shape
                    imgs = nn.functional.interpolate(
                        imgs, size=ns, mode="bilinear", align_corners=False)

            # Forward
            with torch.cuda.amp.autocast(amp):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(pred, targets.to(
                    device), masks=masks.to(device).float())
                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.0

            # Store individual losses for TensorBoard
            if RANK in {-1, 0}:
                batch_box_loss.append(loss_items[0].item())
                batch_seg_loss.append(loss_items[1].item())
                batch_obj_loss.append(loss_items[2].item())
                batch_cls_loss.append(loss_items[3].item())

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)  # unscale gradients
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=10.0)  # clip gradients
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Log
            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / \
                    (i + 1)  # update mean losses
                # (GB)
                mem = f"{torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0:.3g}G"
                pbar.set_description(("%11s" * 2 + "%11.4g" * 6) % (
                    f"{epoch}/{epochs - 1}", mem, *mloss, targets.shape[0], imgs.shape[-1]))

                # Mosaic plots
                if plots:
                    if ni < 3:
                        plot_images_and_masks(
                            imgs, targets, masks, paths, save_dir / f"train_batch{ni}.jpg")
                    if ni == 10:
                        files = sorted(save_dir.glob("train*.jpg"))
                        logger.log_images(files, "Mosaics", epoch)

        # TensorBoard: Log batch-level metrics
        if tb_writer and RANK in {-1, 0}:
            tb_writer.add_scalar(
                'Loss/Box_Loss', np.mean(batch_box_loss), epoch)
            tb_writer.add_scalar('Loss/Segmentation_Loss',
                                 np.mean(batch_seg_loss), epoch)
            tb_writer.add_scalar('Loss/Objectness_Loss',
                                 np.mean(batch_obj_loss), epoch)
            tb_writer.add_scalar('Loss/Classification_Loss',
                                 np.mean(batch_cls_loss), epoch)
            tb_writer.add_scalar('Loss/Total_Loss', mloss.sum().item(), epoch)

        # Scheduler
        lr = [x["lr"] for x in optimizer.param_groups]  # for loggers
        scheduler.step()

        # TensorBoard: Log learning rate
        if tb_writer and RANK in {-1, 0}:
            tb_writer.add_scalar('Learning_Rate/LR', lr[0], epoch)

        if RANK in {-1, 0}:
            # mAP
            ema.update_attr(
                model, include=["yaml", "nc", "hyp", "names", "stride", "class_weights"])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:  # Calculate mAP
                results, maps, _ = validate.run(
                    data_dict,
                    batch_size=batch_size // WORLD_SIZE * 2,
                    imgsz=imgsz,
                    half=amp,
                    model=ema.ema,
                    single_cls=single_cls,
                    dataloader=val_loader,
                    save_dir=save_dir,
                    plots=False,
                    callbacks=callbacks,
                    compute_loss=compute_loss,
                    mask_downsample_ratio=mask_ratio,
                    overlap=overlap,
                )

            # TensorBoard: Log validation metrics
            if tb_writer and RANK in {-1, 0}:
                if len(results) >= 8:
                    precision, recall, map50, map_avg, val_box_loss, val_seg_loss, val_obj_loss, val_cls_loss = results[
                        :8]

                    # Accuracy metrics
                    tb_writer.add_scalar(
                        'Accuracy/Precision', precision, epoch)
                    tb_writer.add_scalar('Accuracy/Recall', recall, epoch)
                    tb_writer.add_scalar('Accuracy/mAP@0.5', map50, epoch)
                    tb_writer.add_scalar(
                        'Accuracy/mAP@0.5:0.95', map_avg, epoch)

                    # Validation losses
                    tb_writer.add_scalar(
                        'Val_Loss/Box_Loss', val_box_loss, epoch)
                    tb_writer.add_scalar(
                        'Val_Loss/Segmentation_Loss', val_seg_loss, epoch)
                    tb_writer.add_scalar(
                        'Val_Loss/Objectness_Loss', val_obj_loss, epoch)
                    tb_writer.add_scalar(
                        'Val_Loss/Classification_Loss', val_cls_loss, epoch)

                    # Log per-class mAP
                    for i, class_map in enumerate(maps):
                        if i < len(names):
                            tb_writer.add_scalar(
                                f'mAP_per_class/{names[i]}', class_map, epoch)

            # Update best mAP
            # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            fi = fitness(np.array(results).reshape(1, -1))
            stop = stopper(epoch=epoch, fitness=fi)  # early stop check
            if fi > best_fitness:
                best_fitness = fi

            log_vals = list(mloss) + list(results) + lr
            metrics_dict = dict(zip(KEYS, log_vals))
            logger.log_metrics(metrics_dict, epoch)

            # Save model
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {
                    "epoch": epoch,
                    "best_fitness": best_fitness,
                    "model": deepcopy(de_parallel(model)).half(),
                    "ema": deepcopy(ema.ema).half(),
                    "updates": ema.updates,
                    "optimizer": optimizer.state_dict(),
                    "opt": vars(opt),
                    "git": GIT_INFO,
                    "date": datetime.now().isoformat(),
                }

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f"epoch{epoch}.pt")
                del ckpt

        # EarlyStopping
        if RANK != -1:  # if DDP training
            broadcast_list = [stop if RANK == 0 else None]
            # broadcast 'stop' to all ranks
            dist.broadcast_object_list(broadcast_list, 0)
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break  # must break all DDP ranks

    # Close TensorBoard writer
    if tb_writer and RANK in {-1, 0}:
        tb_writer.close()

    if RANK in {-1, 0}:
        LOGGER.info(
            f"\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.")
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    LOGGER.info(f"\nValidating {f}...")
                    results, _, _ = validate.run(
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        model=attempt_load(f, device).half(),
                        iou_thres=0.65 if is_coco else 0.60,
                        single_cls=single_cls,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        save_json=is_coco,
                        verbose=True,
                        plots=plots,
                        callbacks=callbacks,
                        compute_loss=compute_loss,
                        mask_downsample_ratio=mask_ratio,
                        overlap=overlap,
                    )
                    if is_coco:
                        metrics_dict = dict(
                            zip(KEYS, list(mloss) + list(results) + lr))
                        logger.log_metrics(metrics_dict, epoch)

        logger.log_metrics(dict(zip(KEYS[4:16], results)), epochs)
        if not opt.evolve:
            logger.log_model(best, epoch)
        if plots:
            plot_results_with_masks(file=save_dir / "results.csv")
            files = ["results.png", "confusion_matrix.png", *
                     (f"{x}_curve.png" for x in ("F1", "PR", "P", "R"))]
            files = [(save_dir / f) for f in files if (save_dir / f).exists()]
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")
            logger.log_images(files, "Results", epoch + 1)

    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    """Parse command line arguments for training configurations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=ROOT /
                        "yolov5s-seg.pt", help="initial weights path")
    parser.add_argument("--cfg", type=str, default="", help="model.yaml path")
    parser.add_argument("--data", type=str, default=ROOT /
                        "data/coco128-seg.yaml", help="dataset.yaml path")
    parser.add_argument("--hyp", type=str, default=ROOT /
                        "data/hyps/hyp.scratch-low.yaml", help="hyperparameters path")
    parser.add_argument("--epochs", type=int, default=100,
                        help="total training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="total batch size for all GPUs, -1 for autobatch")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int,
                        default=640, help="train, val image size (pixels)")
    parser.add_argument("--rect", action="store_true",
                        help="rectangular training")
    parser.add_argument("--resume", nargs="?", const=True,
                        default=False, help="resume most recent training")
    parser.add_argument("--nosave", action="store_true",
                        help="only save final checkpoint")
    parser.add_argument("--noval", action="store_true",
                        help="only validate final epoch")
    parser.add_argument("--noautoanchor", action="store_true",
                        help="disable AutoAnchor")
    parser.add_argument("--noplots", action="store_true",
                        help="save no plot files")
    parser.add_argument("--evolve", type=int, nargs="?", const=300,
                        help="evolve hyperparameters for x generations")
    parser.add_argument("--bucket", type=str, default="", help="gsutil bucket")
    parser.add_argument("--cache", type=str, nargs="?",
                        const="ram", help="image --cache ram/disk")
    parser.add_argument("--image-weights", action="store_true",
                        help="use weighted image selection for training")
    parser.add_argument("--device", default="",
                        help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--multi-scale", action="store_true",
                        help="vary img-size +/- 50%%")
    parser.add_argument("--single-cls", action="store_true",
                        help="train multi-class data as single-class")
    parser.add_argument("--optimizer", type=str,
                        choices=["SGD", "Adam", "AdamW"], default="SGD", help="optimizer")
    parser.add_argument("--sync-bn", action="store_true",
                        help="use SyncBatchNorm, only available in DDP mode")
    parser.add_argument("--workers", type=int, default=8,
                        help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--project", default=ROOT /
                        "runs/train-seg", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true",
                        help="existing project/name ok, do not increment")
    parser.add_argument("--quad", action="store_true", help="quad dataloader")
    parser.add_argument("--cos-lr", action="store_true",
                        help="cosine LR scheduler")
    parser.add_argument("--label-smoothing", type=float,
                        default=0.0, help="Label smoothing epsilon")
    parser.add_argument("--patience", type=int, default=100,
                        help="EarlyStopping patience (epochs without improvement)")
    parser.add_argument("--freeze", nargs="+", type=int,
                        default=[0], help="Freeze layers: backbone=10, first3=0 1 2")
    parser.add_argument("--save-period", type=int, default=-1,
                        help="Save checkpoint every x epochs (disabled if < 1)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Global training seed")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Automatic DDP Multi-GPU argument, do not modify")

    # Instance Segmentation Args
    parser.add_argument("--mask-ratio", type=int, default=4,
                        help="Downsample the truth masks to saving memory")
    parser.add_argument("--no-overlap", action="store_true",
                        help="Overlap masks train faster at slightly less mAP")

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt, callbacks=Callbacks()):
    """Initialize training with TensorBoard logging."""
    if RANK in {-1, 0}:
        print_args(vars(opt))
        check_git_status()
        check_requirements(ROOT / "requirements.txt")

    # Resume
    if opt.resume and not opt.evolve:
        last = Path(check_file(opt.resume) if isinstance(
            opt.resume, str) else get_latest_run())
        opt_yaml = last.parent.parent / "opt.yaml"
        opt_data = opt.data
        if opt_yaml.is_file():
            with open(opt_yaml, errors="ignore") as f:
                d = yaml.safe_load(f)
        else:
            d = torch.load(last, map_location="cpu")["opt"]
        opt = argparse.Namespace(**d)
        opt.cfg, opt.weights, opt.resume = "", str(last), True
        if is_url(opt_data):
            opt.data = attempt_download(opt_data)
    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = (
            check_file(opt.data),
            check_yaml(opt.cfg),
            check_yaml(opt.hyp),
            str(opt.weights),
            str(opt.project),
        )
        assert len(opt.cfg) or len(
            opt.weights), "either --cfg or --weights must be specified"
        if opt.evolve:
            if opt.project == str(ROOT / "runs/train-seg"):
                opt.project = str(ROOT / "runs/evolve-seg")
            opt.exist_ok, opt.resume = opt.resume, False
        if opt.name == "cfg":
            opt.name = Path(opt.cfg).stem
        opt.save_dir = str(increment_path(
            Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        msg = "is not compatible with YOLOv5 Multi-GPU DDP training"
        assert not opt.image_weights, f"--image-weights {msg}"
        assert not opt.evolve, f"--evolve {msg}"
        assert opt.batch_size != - \
            1, f"AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size"
        assert opt.batch_size % WORLD_SIZE == 0, f"--batch-size {opt.batch_size} must be multiple of WORLD_SIZE"
        assert torch.cuda.device_count() > LOCAL_RANK, "insufficient CUDA devices for DDP command"
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device("cuda", LOCAL_RANK)
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo")

    # Train
    if not opt.evolve:
        train(opt.hyp, opt, device, callbacks)


def run(**kwargs):
    """Execute YOLOv5 training with given parameters."""
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
