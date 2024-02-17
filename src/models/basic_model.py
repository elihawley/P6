from models.model import Model
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import RMSprop, Adam

class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Your code goes here
        # you have to initialize self.model to a keras model
        self.model = Sequential()
        # VGG-style architecture: Conv2D blocks, each with small 3x3 filters followed by a max pooling layer
        self.model.add(layers.Conv2D(8, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=input_shape))
        self.model.add(layers.MaxPooling2D((2,2)))
        self.model.add(layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        self.model.add(layers.MaxPooling2D((2,2)))
        self.model.add(layers.Dropout(.25))
        self.model.add(layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        self.model.add(layers.MaxPooling2D((2,2)))
        self.model.add(layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        self.model.add(layers.MaxPooling2D((2,2)))
        self.model.add(layers.Dropout(.15))
        self.model.add(layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        self.model.add(layers.MaxPooling2D((2,2)))
        # Extra layers
        self.model.add(layers.Flatten())
        # Fully connected layer with softmax
        self.model.add(layers.Dense(categories_count, activation='softmax'))
    
    def _compile_model(self):
        # Your code goes here
        # you have to compile the keras model, similar to the example in the writeup
        self.model.compile(
            optimizer=RMSprop(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )