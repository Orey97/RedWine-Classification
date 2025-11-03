from src.dataset import Dataset
import numpy as np
import keras
from keras.layers import Dropout, Dense, Normalization
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

class Modello:
    def __init__(self, dataset):
        self.dataset = dataset
        self.model = None
        self.num_classes = None

    def set_modello(self):
        normalizer = Normalization(axis=-1)
        normalizer.adapt(self.X_train)
        self.model = Sequential([
            normalizer,
            Dense(64, activation='relu'), Dropout(0.3),
            Dense(32, activation='relu'), Dropout(0.2),
            Dense(self.num_classes, activation='softmax')  # Output layer for regression
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def drop_colonna(self, colonna):
        if colonna in self.dataset.data.columns:
            self.true_labels = self.dataset.data[colonna]
            self.dataset.data = self.dataset.data.drop(columns=[colonna])
        else:
            print(f"La colonna '{colonna}' non esiste nel dataset." )

    def split_data(self, test_size=0.2, random_state=42):
        X = self.dataset.data.values
        y = self.true_labels.values
        unique_classes = np.unique(y)
        self.num_classes = len(unique_classes)
        label_map = {label: idx for idx, label in enumerate(unique_classes)}
        y_mapped = np.array([label_map[val] for val in y])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_mapped, test_size=test_size, random_state=random_state)
        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)
        
    def train_model(self, epochs=50, batch_size=32):
        if self.model is None:
            raise ValueError("Il modello non è stato impostato. Chiama 'set_modello' prima di addestrare.")
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[self.early_stopping])

    def early_stopping(self, monitor, patience, restore_best_weights, verbose):
        self.early_stopping = EarlyStopping (monitor=monitor, patience=patience, restore_best_weights=restore_best_weights, verbose=verbose)

    def evaluate_model(self):
        if self.model is None:
            raise ValueError("Il modello non è stato impostato. Chiama 'set_modello' prima di valutare.")
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)
        print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")




