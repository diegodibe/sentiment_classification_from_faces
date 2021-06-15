from sklearn.utils.multiclass import unique_labels
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from captum.attr import GuidedBackprop, Occlusion
import cv2


def visualize_class_distribution(df, lookup, title):
    sns.countplot(x=df.emotion)
    plt.xticks(range(7), lookup)
    plt.show()
    plt.savefig(f'images/{title}_class_distribution.jpg')
    # plt.clf()


def visualize_weights(model, title):
    # Visualize conv filter
    if title == 'InceptionConvNet':
        kernels = {0:{'title':'1x1 layer a'},
                   1:{'title':'1x1 layer b'},
                   2:{'title':'3x3 layer b'},
                   3:{'title':'1x1 layer c'},
                   4:{'title':'5x5 layer c'},
                   5:{'title':'1x1 layer d'}}
        kernels[0]['weights'] = model.layer2single[0].weight.detach().cpu()
        kernels[1]['weights'] = model.layer2a[0].weight.detach().cpu()
        kernels[2]['weights'] = model.layer2a[2].weight.detach().cpu()
        kernels[3]['weights'] = model.layer2b[0].weight.detach().cpu()
        kernels[4]['weights'] = model.layer2b[2].weight.detach().cpu()
        kernels[5]['weights'] = model.layer2c[1].weight.detach().cpu()
        fig, ax = plt.subplots(1, len(kernels))
        for k, v in kernels.items():
            ax[k].set_title(v['title'])
            ax[k].imshow(v['weights'][:,:,0,0], cmap='hot')
        plt.show()
        plt.savefig(f'images/{title}_weight_visualization_inception.jpg')
    else:
        kernels = model.layer1[0].weight.detach().cpu()
        fig, axarr = plt.subplots(kernels.size(0))
        for idx in range(kernels.size(0)):
            axarr[idx].imshow(kernels[idx].squeeze())
        plt.show()
        plt.savefig(f'images/{title}_weight_visualization.jpg')
    # plt.clf()


def visualize_features(model, title, img):
    # Visualize feature maps
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    model.layer1[0].register_forward_hook(get_activation('layer1'))
    data = img
    data.unsqueeze_(0)
    output = model(data)

    act = activation['layer1'].squeeze()
    fig, ax = plt.subplots(1, act.size(0))
    for idx in range(act.size(0)):
        ax[idx].set_title(f'activation {idx}')
        ax[idx].imshow(act[idx].cpu())
    plt.show()
    plt.savefig(f'images/{title}_feature_visualization.jpg')
    # plt.clf()


def visualize_backprop(model, title, images):
    # visualize guided backprop
    # gbp = GuidedBackprop(model)
    # fig, ax = plt.subplots(2, 7)
    # for key, values in images.items():
    #     ax[0, key].set_title(images[key]['class'])
    #     ax[0, key].imshow(images[key]['img'][0, :, :].cpu(), cmap='gray')
    #     #ax[1, key].set_title('')
    #     ax[1, key].imshow(gbp.attribute(images[key]['img'][None, :, :, :], target=key).cpu()[0, 0, :, :], cmap='gray')
    # plt.show()
    # plt.savefig(f'images/{title}_guided_back.jpg')
    # #plt.clf()

    # visualize occlusion
    occlusion = Occlusion(model)
    fig, ax = plt.subplots(2, 7)
    for key, values in images.items():
        ax[0, key].set_title(images[key]['class'])
        ax[0, key].imshow(images[key]['img'][0, :, :].cpu(), cmap='gray')
        # ax[1, key].set_title('')
        attributions_occ = occlusion.attribute(images[key]['img'][None, :, :, :],
                                               strides=(1, 4, 4),
                                               target=key,
                                               sliding_window_shapes=(1, 5, 5),
                                               baselines=0)

        img = cv2.addWeighted(attributions_occ[0, 0, :, :].cpu().numpy(), 0.7,
                              images[key]['img'].cpu().numpy()[0, :, :], 0.3, 0)
        ax[1, key].imshow(img)

    plt.show()
    plt.savefig(f'images/{title}_occlusion.jpg')
    # plt.clf()


def visualize_loss(history, model, fold):
    loss = history["loss"]
    val_loss = history["val_loss"]
    epochs = list(range(len(loss)))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(f'loss visualization {model} fold:{fold}')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    # plt.ylim(0, 0.5)
    plt.legend()
    plt.savefig(f'images/loss_{model}_fold{fold}.jpg')
    # plt.show()
    plt.close()


def matrix_confusion(y_test, y_pred, model):
    classes = ['anger','disgust','fear','happiness','sadness','surprise','neutral']

    # create labels for matrix
    labels = unique_labels(y_test, y_pred)
    # create confusion matrix
    matrix = confusion_matrix(y_test, y_pred, labels)
    sns.heatmap(matrix, square=True, annot=True, fmt='d', cbar=False, xticklabels=classes, yticklabels=classes)
    # set title and labels
    plt.title('Confusion matrix ' + model)
    plt.ylabel('true label')
    plt.xlabel('predicted label')
    plt.show()
    plt.savefig('images/confusion_matrix_' + model + '.jpg')

