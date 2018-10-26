import torch.optim as optim
import copy
import time
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from modules.DataLoader import CMRDataset
from modules.Shared_Network import *
from utils.util import *
import gensim

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
    # Load word embedding model: using word-to-vec
    print('Loading the pre-trained word-to-vec model: GoogleNews-vectors-negative300.bin...')

    word_embeds_model = gensim.models.KeyedVectors.load_word2vec_format('./models/GoogleNews-vectors-negative300.bin',
                                                                     binary=True)

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
            else:
                teacher_model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for sample_batched in dataloaders[phase]:
                char, img  = sample_batched['char'], sample_batched['image']
                # processing sentence
                sentence = [char_table_to_sentence(alphabet=alphabet, char_table=char[idx])
                            for idx in range(0, batch_size)]

                embeds = torch.zeros([batch_size, sen_size, emb_size], dtype=torch.float64)
                for idx in range(0, batch_size):
                    embeds[idx] = word2vec(sentence[idx], model=word_embeds_model,
                                           sen_size=sen_size, emb_size=emb_size)

                embeds = embeds.to(device)
                img = img.to(device)
                #txt = txt.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    img_feature = img_model(img)
                    txt_feature = txt_model(embeds)

                    img_reprets = teacher_model(img_feature)
                    txt_reprets = teacher_model(txt_feature)

                    loss = criterion(img_reprets, txt_reprets)

                    preds = teacher_model.predict(img_reprets, txt_reprets)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * batch_size
                running_corrects += sum([(i==preds[i])+0 for i in range(0,len(preds))])#torch.sum(preds == labels.data)

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

def main():
    ######################################################################
    # Initialize the Teacher model for this run
    # ---------
    #
    model = Teacher_Net().double()
    # Print the model we just instantiated
    print(model)
    ######################################################################
    # Initialize Image Network
    #
    image_net = Image_Net()
    model_f, input_size = image_net.initialize_model(model_name, feature_extract, use_pretrained=True)
    model_f = model_f.double()
    # Print the model we just instantiated
    print(model_f)
    ######################################################################
    # Initialize Text Network
    # ---------
    #
    model_t = Text_Net().double()
    # Print the model we just instantiated
    print(model_t)
    # Send the model to GPU
    model = model.to(device)
    model_f = model_f.to(device)
    model_t = model_t.to(device)
    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_f.parameters()
    print("Params to learn:")

    if feature_extract:
        params_to_update = []
        for name, param in model_f.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_f.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)

    for name, param in model_t.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)

    ######################################################################
    # Load Data
    # ---------
    #
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            Rescale(256),
            RandomCrop(224),
            ToTensor(),
            #Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    cub_dataset = {x: CMRDataset(root_dir=data_dir, caption_dir='cub_icml',
                                 image_dir='images', split = 'train_val.txt',
                                 transform=data_transforms[x]) for x in ['train', 'val']}

    # Create training and validation dataloaders
    dataloaders_dict = {x: DataLoader(cub_dataset[x], batch_size=batch_size, shuffle=True,
                                     num_workers=4) for x in ['train', 'val']}

    # test batch loader
    # for i_batch, sample_batched in enumerate(dataloaders_dict['train']):
    #     print(i_batch, sample_batched['image'].numpy().shape)
    #
    #     if i_batch == 3:
    #         plt.figure()
    #         images_batch = sample_batched['image']
    #         #batch_size = len(images_batch)
    #         grid = utils.make_grid(images_batch)
    #         plt.imshow(grid.numpy().transpose((1, 2, 0)))
    #         plt.axis('off')
    #         plt.ioff()
    #         plt.show()
    #         break

    ######################################################################
    # Create the Optimizer
    # --------------------
    #
    # Observe that all parameters are being optimized
    optimizer = optim.Adam(params_to_update, lr=0.0001)

    ######################################################################
    # Run Training and Validation Step
    # --------------------------------
    #

    # Setup the loss fxn
    criterion = RankingLossFunc(delta=Delta)

    # Train and evaluate
    model, hist = train_model(teacher_model=model, img_model=model_f, txt_model=model_t,
                              dataloaders=dataloaders_dict, criterion=criterion,
                              optimizer=optimizer, num_epochs=num_epochs)
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

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
data_dir = "./datasets/CUB_200_2011"

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
batch_size = 16

# Number of epochs to train for
num_epochs = 50000

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    print("Entering function main...")
    main()