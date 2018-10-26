import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from utils.util import Cos_similarity

class Teacher_Net(nn.Module):
    def __init__(self):
        super(Teacher_Net, self).__init__()
        self.linear1 = nn.Linear(in_features=9216, out_features=4096)
        self.linear2 = nn.Linear(in_features=4096, out_features=4096)
        self.linear3 = nn.Linear(in_features=4096, out_features=1000)

    def forward(self, inputs):
        out = F.relu(self.linear1(inputs))
        out = F.relu(self.linear2(out))
        out = F.softmax(self.linear3(out), dim=1)
        return out

    def predict(self, x_reprets, y_reprets):
        batch_size = x_reprets[0]
        embedding_loss = torch.ones(batch_size, batch_size)
        for i in range(0, batch_size):
            for j in range(0, batch_size):
                embedding_loss[i][j] = 1 - Cos_similarity(x_reprets[i], y_reprets[j])

        preds = torch.argmin(embedding_loss, dim=1)  # return the index of minimal of each row
        return preds



class Text_Net(nn.Module):

    def __init__(self):
        super(Text_Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=16, out_channels=300, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=300, out_channels=300, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=300, out_channels=300, kernel_size=3)
        self.fc1 = nn.Linear(in_features=300*73, out_features=9216)

    def forward(self, inputs):
        out = F.relu(self.conv1(inputs))
        out = F.max_pool1d(F.relu(self.conv2(out)), kernel_size=2)
        out = F.max_pool1d(F.relu(self.conv3(out)), kernel_size=2)
        out = self.fc1(out.view(-1, 300*73))
        return out


class Image_Net():

    def __init__(self):
        super(Image_Net, self).__init__()

    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def initialize_model(self, model_name, feature_extract, use_pretrained=True):
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
        model_ft = None
        input_size = 0
        if model_name == "alexnet":
            """ Alexnet
            """
            model_ft = models.alexnet(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, 9216)  # reinitialize the 6th layer
            input_size = 224

        else:
            print("Invalid model name, exiting...")
            exit()

        return model_ft, input_size


class RankingLossFunc(nn.Module):
    def __init__(self, delta):
        super(RankingLossFunc, self).__init__()
        self.delta = delta

    def forward(self, X, Y):
        assert (X.shape[0] == Y.shape[0])
        loss = 0
        num_of_samples = X.shape[0]
        mask = torch.eye(num_of_samples)
        for idx in range(0, num_of_samples):
            negative_sample_ids = [j for j in range(0, num_of_samples) if mask[idx][j] < 1]
            loss += sum([max(0, self.delta
                     - Cos_similarity(X[idx], Y[idx])
                     + Cos_similarity(X[idx], Y[j])) for j in negative_sample_ids])
        return loss
