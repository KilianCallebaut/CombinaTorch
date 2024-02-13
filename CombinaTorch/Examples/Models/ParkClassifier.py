from typing import List

from torch import nn

from Tasks.Task import Task


class ParkClassifier(nn.Module):
    def __init__(self, task_list: List[Task] = None, output_amount=0):
        super().__init__()
        s1=128
        s2 = 128
        self.classifier = nn.Sequential(
            nn.Linear(s1, s1, bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(s1),
            nn.Linear(s1, s1, bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(s1),
            nn.Dropout(p=0.2),
            nn.Linear(128, s2, bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(s2),
            nn.Dropout(p=0.2),
            nn.Linear(s2, s2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(s2),
        )

        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(s2, len(t.output_labels)),
                nn.LogSoftmax(dim=-1) if t.classification_type == 'multi-class' else nn.Sigmoid()
            )
            for t in task_list]
        ) if task_list else nn.ModuleList([
            nn.Sequential(
                nn.Linear(s2, output_amount),
                nn.Sigmoid()
            )]
        )

    def forward(self, x):
        x = x.squeeze()
        x = self.classifier(x)
        return tuple(head(x) for head in self.output_heads)
