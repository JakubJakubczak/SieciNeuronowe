import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


def build_cnn_model_1():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

def build_cnn_model_2():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

def build_cnn_model_3():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

def build_mlp_model():
    model = Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

def compile_and_train_model(model, x_train, y_train, x_test, y_test, epochs=5):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))
    test_loss, test_acc = model.evaluate(x_test, y_test)
    return test_acc, history

def plot_training_history(history, title='Training and Validation Metrics'):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.suptitle(title)
    plt.show()

def load_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize the data
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Reshape the data to include the channel dimension
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))

    # Convert labels to categorical
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


# Corrupt training labels
def create_corrupted_labels(y_train, corruption_ratio=0.5):
    y_train_corrupted = y_train.copy()
    num_samples = y_train.shape[0]
    num_corrupted = int(num_samples * corruption_ratio)

    corrupted_indices = np.random.choice(num_samples, num_corrupted, replace=False)
    correct_indices = np.setdiff1d(np.arange(num_samples), corrupted_indices)

    # Randomly assign incorrect labels to corrupted indices
    for idx in corrupted_indices:
        incorrect_label = np.random.randint(0, 10)
        while incorrect_label == np.argmax(y_train[idx]):
            incorrect_label = np.random.randint(0, 10)
        y_train_corrupted[idx] = to_categorical(incorrect_label, 10)

    return y_train_corrupted


########################## MAIN ##############

if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = load_dataset()
    # # Train and evaluate CNN models
    # cnn_model_1 = build_cnn_model_1()
    # cnn_model_1_acc,cnn_model_1_history = compile_and_train_model(cnn_model_1, x_train, y_train, x_test, y_test)
    #
    # cnn_model_2 = build_cnn_model_2()
    # cnn_model_2_acc, cnn_model_2_history = compile_and_train_model(cnn_model_2, x_train, y_train, x_test, y_test)
    #
    # cnn_model_3 = build_cnn_model_3()
    # cnn_model_3_acc, cnn_model_3_history = compile_and_train_model(cnn_model_3, x_train, y_train, x_test, y_test)
    #
    # # Train and evaluate MLP model
    mlp_model = build_mlp_model()
    mlp_model_acc, mlp_model_history = compile_and_train_model(mlp_model, x_train, y_train, x_test, y_test)
    #
    # print(f'CNN Model 1 Accuracy: {cnn_model_1_acc}')
    # print(f'CNN Model 2 Accuracy: {cnn_model_2_acc}')
    # print(f'CNN Model 3 Accuracy: {cnn_model_3_acc}')
    print(f'MLP Model Accuracy: {mlp_model_acc}')

    # plot_training_history(cnn_model_1_history, title='CNN Model 1')
    # plot_training_history(cnn_model_2_history, title='CNN Model 2')
    # plot_training_history(cnn_model_3_history, title='CNN Model 3')
    plot_training_history(mlp_model_history, title='MLP model')

    # cnn_model_1.save('cnn_model_1.h5')
    # cnn_model_2.save('cnn_model_2.h5')
    # cnn_model_3.save('cnn_model_3.h5')
    # mlp_model.save('mlp_model.h5')





    # predictions = model.predict(x_test)
    # print(np.argmax(predictions[1]))
    # plt.figure()
    # plt.imshow(x_test[1])
    # plt.colorbar()
    # plt.grid(False)
    # plt.show()
    #
    # test_loss, test_acc = model.evaluate(x_test, y_test)

    # Corrupt the training data
    # y_train_corrupted = create_corrupted_labels(y_train, corruption_ratio=0.5)
    # print(y_train_corrupted[0:10])
    # print("\n")
    # print(y_train[0:10])
    # cnn_model_2= build_cnn_model_2()
    # cnn_model_2_acc, cnn_model_2_history = compile_and_train_model(cnn_model_2, x_train, y_train_corrupted, x_test, y_test)
    # print(f'CNN Model 1 Corrupted Accuracy: {cnn_model_2_acc}')
    # plot_training_history(cnn_model_2_history)