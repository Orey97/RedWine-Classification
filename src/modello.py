from dataset import Dataset
import keras
from keras.layers import Dropout, Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

class Modello:
    def __init__(self, dataset):
        self.dataset = dataset
        self.model = None

    def set_modello(self, input_shape, num_classes):
        self.model = Sequential([
            Dense(64, activation='relu', input_shape=(input_shape,)), Dropout(0.3),    
            Dense(32, activation='relu'), Dropout(0.2),
            Dense(num_classes, activation='softmax')  # Output layer for regression
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
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
        
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




