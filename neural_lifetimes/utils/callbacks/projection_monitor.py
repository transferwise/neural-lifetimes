import pytorch_lightning as pl
import torch


class MonitorProjection(pl.Callback):
    def __init__(self, max_batches: int = 100, mod: int = 5):
        super().__init__()
        self.max_batches = max_batches
        self.mod = mod

    def on_train_epoch_end(self, trainer, net):
        if trainer.current_epoch % self.mod == 0:
            loader = trainer.datamodule.val_dataloader()
            logger = trainer.logger

            encoded_data = []
            labels = []
            for i, mini_batch in enumerate(loader):

                batch_offsets = mini_batch["offsets"][1:] - 1
                embedded = net.net.fc_mu(net.net.encoder(mini_batch))
                emb_lastevent = embedded[batch_offsets]
                encoded_data.append(emb_lastevent)
                if "btyd_mode" in mini_batch.keys():
                    mode = torch.tensor(mini_batch["btyd_mode"][batch_offsets.cpu().numpy()].astype(int))
                    labels.append(mode)

                if len(encoded_data) > self.max_batches:
                    break

            data = torch.cat(encoded_data)
            if labels != []:
                labels = torch.cat(labels)
                logger.experiment.add_embedding(
                    data,
                    metadata=labels,
                    global_step=trainer.global_step,
                    tag=f"Embedding Epoch {trainer.current_epoch}",
                )
                return

            logger.experiment.add_embedding(
                data,
                global_step=trainer.global_step,
                tag=f"Embedding Epoch {trainer.current_epoch}",
            )
