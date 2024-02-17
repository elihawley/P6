from preprocess import get_transfer_datasets
from models.transfered_model import TransferedModel
from models.random_model import RandomModel
from config import image_size, categories, transfer_categories
import matplotlib.pyplot as plt
import time
import numpy as np

# Your code should change these values based on your choice of dataset for the transfer task
# -------------
input_shape = (image_size[0], image_size[1], 3)
categories_count = len(transfer_categories)
# -------------

models = {
    'random_model': RandomModel,
    'transfered_model': TransferedModel,
}

def plot_history_diff(initial_hist, transfered_hist):
    val_acc_initial = initial_hist.history['val_accuracy']
    val_acc_tranfered = transfered_hist.history['val_accuracy']

    epochs_initial = range(1, len(val_acc_initial) + 1)
    epochs_transfered = range(1, len(val_acc_initial) + 1)
    assert epochs_initial == epochs_transfered, "The two models have been tried with different epochs"

    plt.figure(figsize = (24, 6))
    plt.suptitle('Far Transfer from Facial Recognition to Classifying Fresh vs Rotten Fruit')
    plt.plot(epochs_initial, val_acc_initial, 'b', label = 'Random Model Accuracy')
    plt.plot(epochs_initial, val_acc_tranfered, 'r', label = 'Transfered Model Accuracy')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')
    plt.show()

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize = (24, 6))
    plt.subplot(1,2,1)
    plt.plot(epochs, acc, 'b', label = 'Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label = 'Validation Accuracy')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')

    plt.subplot(1,2,2)
    plt.plot(epochs, loss, 'b', label = 'Training Loss')
    plt.plot(epochs, val_loss, 'r', label = 'Validation Loss')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')
    plt.show()


if __name__ == "__main__":
    # if you want to load your model later, you can use:
    # model = Model.load_model("results/optimized_network.keras")
    # to load your history and plot it again, you can use:
    # history = np.load('results/optimized_network.npy',allow_pickle='TRUE').item()
    # plot_history(history)
    # histories = [
    #     np.load('results/random_model_15_epochs_timestamp_1708206429.npy',allow_pickle='TRUE').item(),
    #     np.load('results/transfered_model_15_epochs_timestamp_1708206674.npy',allow_pickle='TRUE').item()
    # ]

    # Your code should change the number of epochs
    epochs = 15
    print('* Data preprocessing')
    train_dataset, validation_dataset, test_dataset = get_transfer_datasets()
    histories = []
    for name, model_class in models.items():
        print('* Training {} for {} epochs'.format(name, epochs))
        model = model_class(input_shape, categories_count)
        model.print_summary()
        history = model.train_model(train_dataset, validation_dataset, epochs)
        histories.append(history)
        print('* Evaluating {}'.format(name))
        model.evaluate(test_dataset)
        print('* Confusion Matrix for {}'.format(name))
        print(model.get_confusion_matrix(test_dataset))
        model_name = '{}_{}_epochs_timestamp_{}'.format(name, epochs, int(time.time()))
        filename = 'results/{}.keras'.format(model_name)
        model.save_model(filename)
        np.save('results/{}.npy'.format(model_name), history)
        print('* Model saved as {}'.format(filename))
    assert len(histories) == 2, "The number of trained models is not equal to two"
    plot_history_diff(*histories)

