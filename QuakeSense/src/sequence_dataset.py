import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):

    def __init__(
        self,
        X,
        baseline,
        y
    ):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.baseline = torch.tensor(
            baseline,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.float32
        )


    def __len__(self):

        return len(self.X)


    def __getitem__(self, idx):

        return (
            self.X[idx],
            self.baseline[idx],
            self.y[idx]
        )


train_loader = DataLoader(
    SequenceDataset(
        X_train,
        baseline_train,
        y_train
    ),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    SequenceDataset(
        X_val,
        baseline_val,
        y_val
    ),
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    SequenceDataset(
        X_test,
        baseline_test,
        y_test
    ),
    batch_size=BATCH_SIZE,
    shuffle=False
)
