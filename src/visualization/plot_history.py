
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