import torch.optim as optim
import copy
import time
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from modules.DataLoader import CMRDataset
from modules.Shared_Network import *
from utils.util import *

from torch.autograd import Variable

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

######################################################################
# Helper Functions
# ----------------
# The ``train_model`` function handles the training and validation of a
# given model. As input, it takes a PyTorch model, a dictionary of
# dataloaders, a loss function, an optimizer, a specified number of epochs
# to train and validate for.
# The function trains for the specified number of epochs and after each
# epoch runs a full validation step. It also keeps track of the best
# performing model (in terms of validation accuracy), and at the end of
# training returns the best performing model. After each epoch, the
# training and validation accuracies are printed.
#

def train_model(teacher_model, img_model, txt_model, dataloaders,
                criterion, optimizer, num_epochs=50000):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(teacher_model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                teacher_model.train()  # Set model to training mode
                img_model.train()
                txt_model.train()
                print('Train phase')
            else:
                teacher_model.eval()  # Set model to evaluate mode
                img_model.eval()
                txt_model.eval()
                print('Val phase')

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            num_batch = 0
            for sample_batched in dataloaders[phase]:

                num_batch += 1
                img = sample_batched['image'].float().to(device)

                #print('[%s] Computing word embeddings...'  % str(num_batch))
                embeds = char_table_to_embeddings(model_path=embeds_path, char=sample_batched['char'],
                                                  alphabet=alphabet, sen_size=sen_size, emb_size=emb_size,
                                                  batch_size=batch_size, device=device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    #  In train mode we calculate the loss by summing the final output and the auxiliary output
                    #  but in testing we only consider the final output.

                    print('[%s] forwarding image and embeddings...' % str(num_batch))
                    img_reprets = teacher_model(img_model(img).float()).to(device)
                    txt_reprets = teacher_model(txt_model(embeds).float()).to(device)

                    loss = criterion(img_reprets, txt_reprets)

                    preds = teacher_model.predict(img_reprets, txt_reprets)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        print('[%s] backward and optimize' % str(num_batch))
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * batch_size
                running_corrects += sum([(i==preds[i]) + 0 for i in range(0, len(preds))])

                # release memory
                #del img, embeds, img_reprets, txt_reprets

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(teacher_model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    teacher_model.load_state_dict(best_model_wts)
    return teacher_model, val_acc_history


# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
data_dir = "./datasets/CUB_200_2011"
embeds_path = "./models/GoogleNews-vectors-negative300.bin"

# Look-up dictionary
alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "

# Text embeddings parameters
sen_size = 16
emb_size = 300

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "alexnet"

# hyper-parameter delta, need tuned
Delta = 0.002

# Batch size for training (change depending on how much memory you have)
batch_size = 128

# Number of epochs to train for
epoch_num = 50000

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True
input_size = 224

# Detect if we have a GPU available
gpu = 1 if torch.cuda.is_available() else 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    ######################################################################
    # Initialize models for this run
    # ---------
    model = Teacher_Net()
    # Initialize Image Network
    model_f = Image_Net().initialize_model(model_name, feature_extract, use_pretrained=True)
    # Initialize Text Network
    model_t = Text_Net()

    # Print the model we just instantiated
    # print(model)
    # print(model_f)
    # print(model_t)

    ######################################################################
    # Initialize variables
    # ---------
    # img = Variable(torch.Tensor((batch_size, 3, input_size, input_size)), requires_grad=True)
    # embeds = Variable(torch.Tensor((batch_size, 16, 300)), requires_grad=True)
    # img_reprets = Variable(torch.Tensor((batch_size, 1000)), requires_grad=True)
    # txt_reprets = Variable(torch.Tensor((batch_size, 1000)), requires_grad=True)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    print("Params to learn:")
    params_to_update_share = []
    params_to_update_img = model_f.parameters()
    params_to_update_txt = []

    for name, param in model.named_parameters():
        if param.requires_grad is True:
            params_to_update_share.append(param)
            print("\t", name)

    if feature_extract:
        params_to_update_img = []
        for name, param in model_f.named_parameters():
            if param.requires_grad is True:
                params_to_update_img.append(param)
                print("\t", name)
    else:
        for name, param in model_f.named_parameters():
            if param.requires_grad is True:
                print("\t", name)

    for name, param in model_t.named_parameters():
        if param.requires_grad is True:
            params_to_update_txt.append(param)
            print("\t", name)

    # Send model and variables to GPU if use gpu
    if gpu > 0:
        print('transferring to gpu...')
        model = model.to(device)
        model_f = model_f.to(device)
        model_t = model_t.to(device)
        print('done')
    else:
        print('running model on CPU')

    ######################################################################
    # Load Data
    # ---------
    #
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            Rescale(256),
            RandomCrop(input_size),
            ToTensor(),
            # Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            Rescale(256),
            CenterCrop(input_size),
            ToTensor(),
            # Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    cub_dataset = {x: CMRDataset(root_dir=data_dir, caption_dir='cub_icml',
                                 image_dir='images', split='%s.txt' % x,
                                 transform=data_transforms[x]) for x in ['train', 'val']}

    # Create training and validation dataloaders
    dataloaders_dict = {x: DataLoader(cub_dataset[x], batch_size=batch_size, shuffle=True,
                                      num_workers=0,
                                      drop_last=True) for x in ['train', 'val']}

    ######################################################################
    # Create the Optimizer
    # --------------------
    #
    # Observe that all parameters are being optimized
    params_to_update = params_to_update_share + params_to_update_img + params_to_update_txt
    optimizer = optim.Adam(params_to_update, lr=0.0001)

    ######################################################################
    # Run Training and Validation Step
    # --------------------------------
    #

    # Setup the loss fxn
    criterion = RankingLossFunc(delta=Delta)
    criterion.to(device)

    # Train and evaluate
    model, hist = train_model(teacher_model=model, img_model=model_f, txt_model=model_t,
                              dataloaders=dataloaders_dict, criterion=criterion,
                              optimizer=optimizer, num_epochs=epoch_num)
    #
    # ohist = []
    # ohist = [h.cpu().numpy() for h in hist]
    # plt.title("Validation Accuracy vs. Number of Training Epochs")
    # plt.xlabel("Training Epochs")
    # plt.ylabel("Validation Accuracy")
    # plt.plot(range(1, num_epochs + 1), ohist, label="Teacher-Student Model")
    # plt.ylim((0, 1.))
    # plt.xticks(np.arange(1, num_epochs + 1, 1.0))
    # plt.legend()
    # plt.show()