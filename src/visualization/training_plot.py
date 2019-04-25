import matplotlib
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
import tensorflow.keras
import numpy as np
import seaborn as sns
from time import time

class TrainingPlot(tensorflow.keras.callbacks.Callback):

    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []
        self.fig = plt.figure()
        sns.set()
        sns.set_style("ticks")
        sns.set_context("talk")
        sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
        plt.ion()
        plt.show()

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):

        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))

        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:

            N = np.arange(0, len(self.losses))

            # Plot train loss, train acc, val loss and val acc against epochs passed
            self.fig.clf()
            plt.plot(N, self.losses, label = "train_loss")
            plt.plot(N, self.acc, label = "train_acc")
            plt.plot(N, self.val_losses, label = "val_loss")
            plt.plot(N, self.val_acc, label = "val_acc")
            plt.title("Training Loss and Accuracy")
            plt.ylim(0, 1.4)
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            plt.draw()
            plt.pause(0.001)


class TrainingPlotPlotly(tensorflow.keras.callbacks.Callback):

    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []
        self.graphData = [go.Scatter(x=[0],y=[0])]
        self.figname = 'logs/trainingplot_{}'.format(time())
        self.graphLayout = go.Layout(
            xaxis=dict(title='Epoch', gridwidth=1, zeroline=False, ticklen=1, linecolor = 'black',linewidth = 2, mirror = True),
            yaxis=dict(range=[0,1], linecolor = 'black',linewidth = 2, mirror = True),
        )

        self.fig = py.plot(self.graphData, filename = self.figname)

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):

        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))

        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:

            N = np.arange(0, len(self.losses))

            # Plot train loss, train acc, val loss and val acc against epochs passed
            self.graphData = []
            self.graphData.append(go.Scatter(
                x = N,
                y = self.losses,
                name = "Training loss"
            ))
            self.graphData.append(go.Scatter(
                x = N,
                y = self.acc,
                name = "Training accuracy"
            ))
            self.graphData.append(go.Scatter(
                x = N,
                y = self.val_losses,
                name = "Validation loss"
            ))
            self.graphData.append(go.Scatter(
                x = N,
                y = self.val_acc,
                name = "Validation accuracy"
            ))

            self.fig.data = self.graphData

def plot_history_keras(history):

    from matplotlib import pyplot as plt

    plt.figure()

    # Use seaborn if available
    try:
        import seaborn as sns
        sns.set()
        sns.set_style("ticks")
        sns.set_context("talk")
        sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
    except ImportError:
        pass

    
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()