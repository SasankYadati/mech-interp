from dataclasses import dataclass
from typing import Optional, Dict
from jaxtyping import Int, Float
from transformer.sample_transformer import SampleTransformer, get_log_probs
from torch import Tensor
from torch.utils.data import DataLoader
import torch as t
import wandb

device = t.device("cuda" if t.cuda.is_available() else "cpu")

@dataclass
class TransformerTrainingArgs():
    batch_size = 16
    epochs = 10
    max_steps_per_epoch = 200
    lr = 1e-3
    weight_decay = 1e-2
    wandb_project: Optional[str] = "my_transformer"
    wandb_name: Optional[str] = None

class TransformerTrainer:
    def __init__(self, args: TransformerTrainingArgs, model: SampleTransformer, dataset_dict):
        super().__init__()
        self.model = model
        self.args = args
        self.optimizer = t.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.step = 0
        self.dataset_dict = dataset_dict


    def training_step(self, batch: Dict[str, Int[Tensor, "batch seq"]]) -> Float[Tensor, ""]:
        '''
        Calculates the loss on the tokens in the batch, performs a gradient update step, and logs the loss.

        Remember that `batch` is a dictionary with the single key 'tokens'.
        '''
        tokens = batch["tokens"]
        tokens.to(device)
        logits = self.model(tokens)
        loss = -get_log_probs(logits, tokens).mean()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.step += 1
        wandb.log({"train_loss": loss}, step=self.step)
        return loss



    def validation_step(self, batch: Dict[str, Int[Tensor, "batch seq"]]):
        '''
        Calculates & returns the accuracy on the tokens in the batch (i.e. how often the model's prediction
        is correct). Logging should happen in the `train` function (after we've computed the accuracy for 
        the whole validation set).
        '''
        self.model.eval()
        tokens = batch["tokens"].to(device)
        logits = self.model(tokens)
        predicted_tokens = t.argmax(logits, dim=-1)
        predicted_tokens = predicted_tokens[:, :-1]
        tokens = tokens[:, 1:]
        is_correct = tokens == predicted_tokens
        self.model.train()
        return is_correct.flatten()


    def train(self):
        '''
        Trains the model, for `self.args.epochs` epochs. Also handles wandb initialisation, and early stopping
        for each epoch at `self.args.max_steps_per_epoch` steps.
        '''
        wandb.init(project=self.args.wandb_project,
                   name=self.args.wandb_name, config=self.args)
        train_dl = self.train_loader()
        test_dl = self.test_loader()
        for epoch in range(self.args.epochs):
            for i, batch in enumerate(train_dl):
                loss = self.training_step(batch)
                if i >= self.args.max_steps_per_epoch:
                    break
            is_correct = t.concat([self.validation_step(batch) for batch in test_dl])
            acc = is_correct.float().mean().item()
            wandb.log({"accuracy": acc}, step=self.step)
        wandb.finish()


    def train_loader(self) -> DataLoader:
        '''Returns train loader (as in code above).'''
        return DataLoader(self.dataset_dict["train"], batch_size=self.args.batch_size, shuffle=True, num_workers=4, pin_memory=True)


    def test_loader(self) -> DataLoader:
        '''Returns test loader (as in code above).'''
        return DataLoader(self.dataset_dict["test"], batch_size=self.args.batch_size, shuffle=False, num_workers=4, pin_memory=True)