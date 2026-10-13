import copy
import math
import os
import statistics
from collections import deque
import torch
from data import train_dataloader
from data import valid_dataloader
import torch.nn.functional as F
import torch.nn as nn
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback
import torch.nn.functional as f

torch.set_float32_matmul_precision('high')

class PeriodicCheckpointCallback(Callback):
    def __init__(self, save_dir, interval=50):
        self.save_dir = save_dir
        self.interval = interval

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.interval == 0:
            ckpt_path = os.path.join(self.save_dir, f"epoch_{epoch:02d}.ckpt")
            trainer.save_checkpoint(ckpt_path)

# [mod] With --valid_freq 0, save last.ckpt at the end of every epoch ourselves.
# ModelCheckpoint(save_last=True) defers saving to the end of validation when check_val_every_n_epoch=None,
# so without validation it never saved (no last.ckpt after the 10-02 crash). Write to a temp file and swap,
# so the previous last.ckpt survives a crash during saving.
class LastCheckpointCallback(Callback):
    def __init__(self, save_dir):
        self.path = os.path.join(save_dir, "last.ckpt")

    def on_train_epoch_end(self, trainer, pl_module):
        tmp = self.path + ".tmp"
        trainer.save_checkpoint(tmp)
        if trainer.is_global_zero:
            os.replace(tmp, self.path)

class HazeRemovalModule(pl.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args
        self.criterion = torch.nn.L1Loss()
        self.psnr = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0)
        self.save_hyperparameters(ignore=['model'])
        self.version = 0
        # [mod] --compile: torch.compile only the training step (validation stays eager: image sizes vary).
        # object.__setattr__ = not registered as a submodule -> checkpoint keys stay 'model.*'.
        object.__setattr__(self, 'train_fwd',
                           torch.compile(model) if getattr(args, 'compile', False) else model)
        # [mod 10-08] guard state (training_step / on_before_optimizer_step / _skip)
        self._loss_hist = deque(maxlen=200)   # recently accepted micro-batch losses
        self._skips = 0
        self._consec_skips = 0
        self._last_mon = -1
        self._last_probe = -1
        self._probe_x = None

    def forward(self, x):
        return self.model(x)
        
    def fft_loss(self, pred, label):
        # cuFFT has no bf16 kernels: under --precision bf16-mixed the prediction
        # arrives in bf16, so compute the spectrum in fp32 (no-op in fp32 runs)
        pred, label = pred.float(), label.float()
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        label_fft = torch.fft.fft2(label, dim=(-2, -1))
        pred_fft = torch.stack((pred_fft.real, pred_fft.imag), -1)
        label_fft = torch.stack((label_fft.real, label_fft.imag), -1)
        return self.criterion(pred_fft, label_fft)
    
    def training_step(self, batch, batch_idx):
        input_img, label_img = batch
        pred_img = self.train_fwd(input_img)

        label_img2 = F.interpolate(label_img, scale_factor=0.5, mode='bilinear')
        label_img4 = F.interpolate(label_img, scale_factor=0.25, mode='bilinear')
        l1 = self.criterion(pred_img[0], label_img4)
        l2 = self.criterion(pred_img[1], label_img2)
        l3 = self.criterion(pred_img[2], label_img)
        loss_content = l1 + l2 + l3
        
        f1 = self.fft_loss(pred_img[0], label_img4)
        f2 = self.fft_loss(pred_img[1], label_img2)
        f3 = self.fft_loss(pred_img[2], label_img)
        loss_fft = f1 + f2 + f3
        
        loss = loss_content + 0.1 * loss_fft

        # [mod 10-08] guard 1: if the loss is non-finite or > 10x the median of the last 200 micro-batches,
        # skip this batch (returning None -> PL skips backward). Cause of the 10-07 NaN: for some inputs the residual
        # stream grows 2-10x per block -> loss 1e16 -> grad inf -> all weights NaN (see README, 'Stability').
        lv = loss.detach().item()
        med = statistics.median(self._loss_hist) if len(self._loss_hist) == self._loss_hist.maxlen else None
        if not math.isfinite(lv) or (med is not None and lv > 10 * med):
            self._skip(f'loss {lv:.4g} (median of last 200: {med if med is None else round(med, 4)})', batch_idx)
            return None
        self._loss_hist.append(lv)
        self._consec_skips = 0

        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']

        psnr_value = self.psnr(pred_img[2], label_img)
        self.log("train_psnr", psnr_value, prog_bar=True, logger=True)
        self.log('learning_rate', current_lr, prog_bar=True)
        self.log('train_loss_content', loss_content, prog_bar=True)
        self.log('train_loss_fft', loss_fft, prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)

        return loss

    # [mod 10-08] guard 2: if the (accumulated) gradient has NaN/inf, do not update in this optimizer step.
    # Called right before clipping, so clearing the grads makes clipping a no-op and Adam skips every parameter.
    def on_before_optimizer_step(self, optimizer):
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        if grads:
            tn = torch.nn.utils.get_total_norm(grads)
            if not torch.isfinite(tn):
                optimizer.zero_grad(set_to_none=True)
                self._skip(f'non-finite grad norm {tn.item()}', None)

    def _skip(self, reason, batch_idx):
        self._skips += 1
        self._consec_skips += 1
        print(f"[guard] skip: global_step {self.global_step} epoch {self.current_epoch} batch {batch_idx}: "
              f"{reason} | total skips {self._skips}", flush=True)
        self.log('guard_skips', float(self._skips))
        if self._consec_skips >= 50:   # do not run for days while skipping everything
            raise RuntimeError(f"[guard] {self._consec_skips} consecutive skipped batches -> stop")

    # [mod 10-08] monitoring (off by default): --monitor_freq N -> every N optimizer steps, max |value| of the
    # Encoder.2 / Decoder / final outputs on the current batch. --probe_batch F --probe_freq K -> every K steps the same
    # for a saved batch (the 10-08 blow-up batch), per sample. Computed on a copy: no effect on training/BN stats/compile.
    def on_train_batch_end(self, outputs, batch, batch_idx):
        gs = self.global_step
        mf = getattr(self.args, 'monitor_freq', 0)
        if mf > 0 and gs % mf == 0 and gs != self._last_mon:
            self._last_mon = gs
            r = self._activation_max(batch[0])
            print(f"[monitor] step {gs} " + " ".join(f"{k} {v.max().item():.3e}" for k, v in r.items()), flush=True)
            for k, v in r.items():
                self.log(f'monitor/{k}', v.max().item())
        pf, pb = getattr(self.args, 'probe_freq', 0), getattr(self.args, 'probe_batch', '')
        if pf > 0 and pb and gs % pf == 0 and gs != self._last_probe:
            self._last_probe = gs
            if self._probe_x is None:
                self._probe_x = torch.load(pb, map_location='cpu', weights_only=False)['x']
            r = self._activation_max(self._probe_x.to(self.device))
            print(f"[probe] step {gs} " + " | ".join(
                f"{k} " + " ".join(f"{x:.2e}" for x in v.tolist()) for k, v in r.items()), flush=True)
            for k, v in r.items():
                self.log(f'probe/{k}', v.max().item())

    @torch.no_grad()
    def _activation_max(self, x):
        m = copy.deepcopy(self.model).train()   # same train mode as in training (BN uses batch statistics)
        acts = {}
        def keep(name):
            def h(mod, inp, out):
                acts[name] = out
            return h
        m.Encoder[2].register_forward_hook(keep('enc2'))
        for i in range(len(m.Decoder)):
            m.Decoder[i].register_forward_hook(keep(f'dec{i}'))
        for i, o in enumerate(m(x)):
            acts[f'out{i}'] = o
        res = {k: v.flatten(1).abs().amax(1).cpu() for k, v in acts.items()}   # max |value| per sample
        del m, acts
        return res

    def validation_step(self, batch, batch_idx):
        input_img, label_img = batch
        factor = 8
        
        h, w = input_img.shape[2], input_img.shape[3]
        H, W = ((h+factor)//factor)*factor, ((w+factor)//factor*factor)
        padh = H-h if h%factor!=0 else 0
        padw = W-w if w%factor!=0 else 0
        input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')
        
        pred = self(input_img)[2]
        pred = pred[:,:,:h,:w]
        
        pred_clip = torch.clamp(pred, 0, 1)
        
        psnr = self.psnr(pred_clip, label_img)
        
        self.log('val_psnr', psnr, prog_bar=True, sync_dist=True)
        return psnr
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # [mod] Closed form of the per-epoch LR that the original GradualWarmupScheduler(1 epoch) + CosineAnnealingLR(
        # T_max=num_epoch-1, eta_min=1e-6) produces in a fresh run:
        #   epoch 0 -> 0 (first warmup epoch),  epoch 1 -> lr,  epoch e>=2 -> cosine(t = e-2)
        # The original breaks on resume: (1) it rebuilds a bare cosine and overwrites the saved wrapper state so the LR
        # curve shifts, (2) with the wrapper, the whole cosine object (incl. an optimizer copy) is stored in the state,
        # so after restoring only the copy is stepped and the real LR freezes. LambdaLR stores only last_epoch.
        warmup_epochs = 1
        t_max = self.args.num_epoch - warmup_epochs
        eta_min, base = 1e-6, self.args.learning_rate

        def lr_factor(e):
            if e <= warmup_epochs:
                return e / warmup_epochs
            t = e - warmup_epochs - 1
            return (eta_min + (base - eta_min) * (1 + math.cos(math.pi * t / t_max)) / 2) / base

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }

def _train(model, args):
    lightning_module = HazeRemovalModule(model, args)
    
    train_loader = train_dataloader(args.data_dir, args.batch_size, args.num_worker,
                                    ps=getattr(args, 'patch_size', 256))
    # [mod] --valid_freq 0 = no validation during training: no val loader (no sanity check either),
    # and checkpoints are written every epoch as last.ckpt instead of by val_psnr, so resume works.
    no_val = args.valid_freq <= 0
    val_loader = None if no_val else valid_dataloader(args.data_dir, batch_size=1,
                                  num_workers=min(4, args.num_worker),
                                  center_crop=getattr(args, 'val_crop', None),
                                  split=getattr(args, 'val_split', 'test'))
    
    if no_val:
        checkpoint_callback = LastCheckpointCallback(args.model_save_dir)
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.model_save_dir,
            filename='epoch_{epoch:02d}_psnr_{val_psnr:.2f}',
            monitor='val_psnr',
            mode='max',
            save_top_k=3,
            save_last=True
        )
    periodic_ckpt_callback = PeriodicCheckpointCallback(
        save_dir=args.model_save_dir,
        interval=args.save_freq
    )
    
    logger = TensorBoardLogger(
        save_dir='logs',
        default_hp_metric=False,
        name=args.logs
    )
    
    # Pick the accelerator from what is actually present so the same script
    # runs on a CPU-only laptop (debugging), a single GPU, or multi-GPU DDP.
    n_gpu = torch.cuda.device_count()
    if n_gpu == 0:
        accelerator, devices, strategy = 'cpu', 1, 'auto'
    elif n_gpu == 1:
        accelerator, devices, strategy = 'gpu', 1, 'auto'
    else:
        accelerator, devices, strategy = 'gpu', -1, 'ddp_find_unused_parameters_true'
    print(f"[trainer] accelerator={accelerator} devices={devices} strategy={strategy}")

    precision = getattr(args, 'precision', '32-true')
    if accelerator == 'cpu' and precision != '32-true':
        precision = '32-true'
    accum = getattr(args, 'accum', 1)
    print(f"[trainer] precision={precision}  val_split={getattr(args, 'val_split', 'test')}"
          f"  valid_freq={args.valid_freq}  accumulate_grad_batches={accum}"
          f" (effective batch {args.batch_size * accum})  max_steps={getattr(args, 'max_steps', -1)}")

    trainer = pl.Trainer(
        max_epochs=args.num_epoch,
        log_every_n_steps=50,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        callbacks=[checkpoint_callback, periodic_ckpt_callback],
        logger=logger,
        gradient_clip_val=0.01,
        check_val_every_n_epoch=None if no_val else args.valid_freq,
        # [mod 10-08] upstream UDPNet trains with DDP (devices=-1) at batch 6 per GPU -> effective batch 12 on 2x 3090.
        # On one GPU, --accum 2 gives the same effective batch (PL divides the loss by accum -> same gradient as a batch-12 mean)
        accumulate_grad_batches=accum,
        max_steps=getattr(args, 'max_steps', -1),
    )
    
    if args.resume:
        trainer.fit(
            lightning_module,
            train_loader,
            val_loader,
            ckpt_path = args.resume,
            # [mod] the checkpoint stores args (a Namespace), so the default weights_only=True load is
            # refused (PL 2.6 + torch 2.13). These are our own checkpoints -> full load.
            weights_only = False
        )
    else:
        trainer.fit(lightning_module, train_loader, val_loader)
    
    trainer.save_checkpoint(os.path.join(args.model_save_dir, 'Final.ckpt'))
    print(f"训练完成！最终模型保存在 {args.model_save_dir}/Final.ckpt")
