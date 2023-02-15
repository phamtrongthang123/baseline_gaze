import torch
from sklearn.metrics import roc_auc_score

__all__ = ["AUC"]


class AUC:
    def __init__(self, nclasses, mode="weighted", multi_class="ovo"):
        self.mode = mode
        self.multi_class = multi_class
        self.labels = list(range(nclasses))
        self.reset()

    def update(self, output, target):
        pred = torch.softmax(output, dim=1)
        self.pred += pred.cpu().tolist()
        self.target += target.cpu().tolist()

    def reset(self):
        self.pred = []
        self.target = []

    def value(self):
        return roc_auc_score(
            self.target,
            self.pred,
            labels=self.labels,
            average=self.mode,
            multi_class=self.multi_class,
        )

    def summary(self):
        print(f"+ AUC:")
        for mode in ["macro", "weighted"]:
            f1 = roc_auc_score(
                self.target,
                self.pred,
                labels=self.labels,
                average=mode,
                multi_class=self.multi_class,
            )
            print(f"{mode}: {f1}")


if __name__ == "__main__":
    auc = AUC(nclasses=4)
    auc.update(
        auc.calculate(
            torch.tensor(
                [[0.1, 0.2, 0.4, 0.3], [0.1, 0.1, 0.8, 0.0], [0.1, 0.5, 0.2, 0.2]]
            ),
            torch.tensor([2, 2, 3]),
        )
    )
    auc.summary()
    auc.update(
        auc.calculate(
            torch.tensor(
                [[0.9, 0.1, 0.0, 0.0], [0.6, 0.2, 0.1, 0.1], [0.7, 0.0, 0.3, 0.0]]
            ),
            torch.tensor([1, 1, 2]),
        )
    )
    auc.summary()
