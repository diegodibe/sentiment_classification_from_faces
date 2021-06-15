import itertools
import numpy as np
from data_process import preprocess_data, load_dataset
from visualization import *
import torch
from torchsummary import summary
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import models
from models import SimpleConvNet, ComplexConvNet, InceptionConvNet
from sklearn.metrics import classification_report, accuracy_score


def reset_weights(m):
    """
    Try resetting model weights to avoid
    weight leakage.
    """
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


def evaluation(model, test):
    test_loader = torch.utils.data.DataLoader(test, batch_size=10)
    # Evaluation for this fold
    model.eval()
    # Iterate over the test data and generate predictions
    predictions = []
    y = []
    for i, data in enumerate(test_loader, 0):
        # Get inputs
        inputs, targets = data
        y.append(targets.cpu().numpy())

        # Generate outputs
        outputs = model(inputs)

        # Set total and correct
        _, predicted = torch.max(outputs.data, 1)
        predictions.append(predicted.cpu().numpy())

    y = np.asarray(list(itertools.chain(*y)))
    predictions = np.asarray(list(itertools.chain(*predictions)))

    matrix_confusion(y, predictions, '')

    # Print accuracy
    print(classification_report(y, predictions))
    print('--------------------------------')
    return accuracy_score(y, predictions)


def training(model_name, train, test, num_epochs, k_folds, device):
    # Load the model
    model = None
    if model_name == 'SimpleConvNet':
        model = SimpleConvNet().to(device)
    elif model_name == 'ComplexConvNet':
        model = ComplexConvNet().to(device)
    elif model_name == 'InceptionConvNet':
        model = InceptionConvNet().to(device)
    elif model_name == 'vgg':
        model = models.extend_vgg()
    summary(model, (1, 48, 48))
    # For fold results
    results = {}
    best_accuarcy = -1
    loss_function = nn.CrossEntropyLoss()
    kfold = KFold(n_splits=k_folds, shuffle=True)

    for fold, (train_ids, val_ids) in enumerate(kfold.split(train)):
        print(f'training fold {fold}')
        # Sample elements randomly from a given list of ids, no replacement
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

        # Define data loaders for training and testing data in this fold
        train_loader = torch.utils.data.DataLoader(train, batch_size=10, sampler=train_subsampler)
        val_loader = torch.utils.data.DataLoader(train, batch_size=10, sampler=val_subsampler)

        # Init the neural model
        model.apply(reset_weights)

        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
        history = {
            'loss': [],
            'val_loss': []
        }
        # Run the training loop for defined number of epochs
        for epoch in range(num_epochs):

            # Print epoch
            print(f'Starting epoch {epoch}')

            # Set current loss value
            current_loss = 0.0
            running_loss = 0.0

            model.train()

            # Iterate over the DataLoader for training data
            for i, data in enumerate(train_loader):

                # Get inputs
                inputs, targets = data

                # Zero the gradients
                optimizer.zero_grad()

                # Perform forward pass
                outputs = model(inputs)

                # Compute loss
                loss = loss_function(outputs, targets)

                # Perform backward pass
                loss.backward()

                # Perform optimization
                optimizer.step()
                # scheduler.step(running_loss)

                # Print statistics
                current_loss += loss.item()
                running_loss += loss.item() * inputs.size(0)
                # running_loss += loss.item() * len(inputs)
                if i % 500 == 499:
                    print('Loss after mini-batch %5d: %.3f' % (i + 1, current_loss / 500))
                    current_loss = 0.0

            epoch_loss = running_loss / len(train_subsampler)
            print(f"Epoch {epoch}, loss: {epoch_loss}")
            history['loss'].append(epoch_loss)

            model.eval()

            val_loss, correct = 0.0, 0.0
            # Iterate over the test data and generate predictions
            for i, data in enumerate(val_loader, 0):
                    # Get inputs
                    inputs, targets = data

                    # Generate outputs
                    outputs = model(inputs)

                    # Compute loss
                    loss = loss_function(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
                    # val_loss += loss.item().len(inputs)

                    # Set total and correct
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == targets).sum().item()

            accuracy = 100.0 * correct / len(val_subsampler)
            val_loss = val_loss / len(val_subsampler)

            print(f"Val Loss for epoch {epoch}: {val_loss}, accuracy: {accuracy}")
            history['val_loss'].append(val_loss)

        visualize_loss(history, model_name, fold)
        # Process is complete.
        print('Training process has finished. Saving trained model.')

        # Print about testing
        print('Starting testing')

        results[fold] = evaluation(model, test)

        if results[fold] > best_accuarcy:
            # Saving the model
            torch.save(model, f'model_{model_name}_{fold}.pth')
            best_accuarcy = results[fold]

    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average: {sum / len(results.items())} %')


def train_model(model_name, device, train, test):
    k_folds = 5
    num_epochs = 50

    training(model_name, train, test, num_epochs, k_folds, device)


def evaluate_model(model_name, test):
    model = torch.load(f'model_{model_name}.pth')
    evaluation(model, test)


def visualize_results(model_name, train):
    model = torch.load(f'model_{model_name}.pth')
    # get image for each class
    images = {0: {'class': 'anger'},
              1: {'class': 'disgust'},
              2: {'class': 'fear'},
              3: {'class': 'happiness'},
              4: {'class': 'sadness'},
              5: {'class': 'surprise'},
              6: {'class': 'neutral'}}
    for type in images.keys():
        images[type]['img'] = train[torch.nonzero(train.tensors[1] == type)[0].cpu().numpy()[0]][0]

    # visualize learnt representation
    visualize_features(model, model_name, images[0]['img'])

    visualize_weights(model, model_name)

    visualize_backprop(model, model_name, images)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = 'ComplexConvNet'
    # preprocess_data()
    train, test = load_dataset(device)
    train_model(model_name, device, train, test)
    evaluate_model(model_name, test)
    # visualize_results(model_name, train)



