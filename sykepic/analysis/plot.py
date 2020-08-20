"""Helper functions for creating plots."""

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt


def view_batch(dataloader, h=None, w=None, save=None):
    batch, _ = next(iter(dataloader))
    bs = batch.shape[0]
    if w:
        h = int(bs/w)
    elif h:
        w = int(bs/h)
    else:
        h = int(np.sqrt(bs))
        w = h
    # Create an image matrix
    for y in range(0, h*w, h):
        # Initialize each row with its first image
        row = batch[y, :, :]
        for x in range(1, h):
            img = batch[x+y, :, :]
            # Concatenate each image to the corresponding row
            row = torch.cat((row, img), dim=1)
        if y == 0:
            matrix = row
        else:
            # Concatenate each row to final matrix
            matrix = torch.cat((matrix, row), dim=2)
    # Modify Tensor to work with opencv
    matrix = np.array(matrix).transpose((1, 2, 0))
    if batch.shape[1] == 3:
        matrix = cv2.cvtColor(matrix, cv2.COLOR_BGR2RGB)
    if save:
        matrix = cv2.convertScaleAbs(matrix, alpha=(255.0))
        cv2.imwrite(str(save), matrix)
    else:
        cv2.imshow(f'{h}x{w} collage', matrix)
        cv2.waitKey(0)


def plot_stats(train_accs, train_losses, val_accs, val_losses,
               title=None, outfile=None, first_epoch=1, epoch_step=1):
    """Plots training statistics"""

    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, dpi=100, figsize=(12, 8.4))
    epochs = np.arange(first_epoch, first_epoch + len(train_accs), epoch_step)
    xticks = np.arange(0, len(train_accs), epoch_step)
    plt.xticks(xticks, epochs)
    plt.xlabel('Epoch')
    if title:
        plt.title(title)

    ax1.plot(train_accs, label='Training', c='turquoise', lw=2)
    ax1.plot(val_accs, label='Validation', c='tomato', lw=2)
    ax1.legend(loc='upper left')
    acc_min = min(train_accs + val_accs)
    acc_stepsize = np.linspace(1, acc_min, 14, retstep=True)[1]
    acc_zeros = int(np.ceil(-np.log10(np.abs(acc_stepsize))))
    acc_stepsize = round(acc_stepsize, acc_zeros)
    acc_yticks = np.arange(1, acc_min + acc_stepsize, acc_stepsize)
    acc_yticks = np.flip(acc_yticks)
    ax1.yaxis.set_ticks(acc_yticks)
    ax1.set_ylabel('Accuracy')

    ax2.plot(train_losses, label='Training', c='turquoise', lw=2)
    ax2.plot(val_losses, label='Validation', c='tomato', lw=2)
    ax2.legend(loc='upper left')
    loss_min = max(train_losses + val_losses)
    loss_stepsize = np.linspace(0, loss_min, 14, retstep=True)[1]
    loss_zeros = int(np.ceil(-np.log10(np.abs(loss_stepsize))))
    loss_stepsize = round(loss_stepsize, loss_zeros)
    loss_yticks = np.arange(0, loss_min + loss_stepsize, loss_stepsize)
    ax2.yaxis.set_ticks(loss_yticks)
    ax2.set_ylabel('Loss')

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile)
    else:
        plt.show()
    plt.close()


def dataset_distribution(data, save=None, size=(8.4, 12)):
    values = []
    labels = []
    # Order classes alphabetically
    classes = sorted(data.distribution.items())
    # Order by class size, descending
    classes = sorted(classes, key=lambda x: x[1][0])
    for class_ in classes:
        values.append(class_[1][0])
        labels.append(class_[0])

    plt.style.use('dark_background')
    plt.figure(figsize=size)
    plt.barh(labels, values, color='turquoise')
    for i, v in enumerate(values):
        # , fontweight='regular')
        plt.text(v, i, " "+str(v), va='center', color='tomato')
    plt.grid(False)
    a = plt.gca().axes
    a.get_xaxis().set_visible(False)
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.spines["bottom"].set_visible(False)
    a.spines["left"].set_visible(False)
    if save:
        plt.tight_layout()
        plt.savefig(save, dpi=100)
    else:
        plt.show()


def plot_img(img, title="insert title here"):  # , cmap='BGR'):
    """Plots image"""

    plt.axis('off')
    plt.title(title)
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    elif img.shape[2] == 1:
        img = img.reshape(img.shape[0], img.shape[1])
        plt.imshow(img, cmap='gray')
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
    plt.show()
