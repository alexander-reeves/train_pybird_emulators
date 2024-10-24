import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, Input, Dense, Activation, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    LearningRateScheduler,
    LambdaCallback,
)
from sklearn.preprocessing import StandardScaler  # for scaling input and output data
from sklearn.preprocessing import RobustScaler  # for scaling input and output data
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tqdm import tqdm
import pickle

# # Sample model with the custom activation
def create_model(input_dim, hidden_layers=[64, 32], output_dim=1):
    inputs = Input(shape=(input_dim,))
    x = inputs

    # Add hidden layers with custom activation
    for units in hidden_layers:
        x = Dense(units)(x)
        x = CustomActivation()(x)

    outputs = Dense(output_dim)(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer="adam", loss="mse"
    )  # Using Mean Squared Error for simplicity
    return model


# #Custom activation function to match https://arxiv.org/abs/2106.03846
class CustomActivation(Layer):
    def build(self, input_shape):
        # Trainable weight variables for alpha and beta initialized with random normal distribution
        self.alpha = self.add_weight(
            name="alpha",
            shape=input_shape[1:],
            initializer="random_normal",
            trainable=True,
        )
        self.beta = self.add_weight(
            name="beta",
            shape=input_shape[1:],
            initializer="random_normal",
            trainable=True,
        )
        super(CustomActivation, self).build(input_shape)

    def call(self, inputs):
        return (self.beta + tf.sigmoid(self.alpha * inputs) * (1 - self.beta)) * inputs


# Custom loss function
class CustomLoss(Layer):
    def __init__(self, element_weights=None, **kwargs):
        super(CustomLoss, self).__init__(**kwargs)
        self.element_weights = (
            tf.convert_to_tensor(element_weights, dtype=tf.float32)
            if element_weights is not None
            else tf.constant(1.0, dtype=tf.float32)
        )

    def call(self, y_true, y_pred):
        # Compute element-wise squared errors
        squared_errors = tf.square(y_true - y_pred)

        if self.element_weights is not None:
            # Apply element-wise weights
            squared_errors = squared_errors / self.element_weights

        # Compute the mean of squared errors
        return tf.reduce_mean(squared_errors)

    def get_config(self):
        config = super(CustomLoss, self).get_config()
        config.update({"element_weights": self.element_weights.numpy()})
        return config


class IntegratedModel:
    def __init__(
        self,
        keras_model,
        input_scaler,
        output_scaler,
        temp_file=None,
        offset=None,
        log_preprocess=False,
        zero_columns=None,
        rescaling_factor=None,
        pca=None,
        pca_scaler=None,
    ):
        self.model = keras_model
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler
        self.offset = offset
        self.zero_columns = zero_columns
        self.rescaling_factor = rescaling_factor
        self.temp_file = temp_file
        self.train_losses = []
        self.val_losses = []
        self.log_preprocess = log_preprocess
        self.pca = pca
        self.pca_scaler = pca_scaler

    def predict(self, data):
        scaled_data = (data - self.scaler_mean_in) / self.scaler_scale_in
        prediction = self.model(tf.convert_to_tensor(scaled_data, dtype=tf.float32))

        # Apply PCA transform if PCA is used
        if self.pca is not None:
            print("yes PCA")
            prediction = prediction * self.scaler_scale_out + self.scaler_mean_out
            prediction = self.pca.inverse_transform(prediction)
            prediction = self.pca_scaler.inverse_transform(prediction)
        else:
            prediction = prediction * self.scaler_scale_out + self.scaler_mean_out

        # Inverse offset and log
        if self.log_preprocess:
            prediction = np.exp(prediction) + 2 * self.offset

        if self.zero_columns is not None:
            # re-inset columns of zeros using the indices of zeros from the original array size

            output_shape = (
                prediction.shape[0],
                prediction.shape[1] + len(self.zero_columns),
            )

            # Create a boolean mask for the columns
            all_cols = np.arange(output_shape[1])
            mask = ~np.isin(all_cols, self.zero_columns)

            # Create an array of zeros with the original shape
            final_prediction = np.zeros(output_shape)

            # Fill in the data from the reduced array using boolean indexing
            final_prediction[:, mask] = prediction

        else:
            final_prediction = prediction

        return final_prediction

    def train(self, X, y, learning_rates=[1e-3, 5e-4, 1e-4, 5e-5], *args, **kwargs):
        # Scale the training data
        if self.input_scaler is not None:
            X_scaled = self.input_scaler.transform(X)
            y_scaled = self.output_scaler.transform(y)
        else:
            X_scaled = X
            y_scaled = y

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=200, verbose=1, restore_best_weights=True
        )

        for lr in learning_rates:
            print(f"\nTraining with learning rate: {lr}")

            # Compile the model with the current learning rate
            custom_loss = CustomLoss(element_weights=self.rescaling_factor)
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=lr), loss=custom_loss
            )

            total_epochs = kwargs.get("epochs", 200)

            # Wrap the training in a tqdm progress bar
            with tqdm(total=total_epochs, unit="epoch", postfix={}) as pbar:

                def on_epoch_end(epoch, logs):
                    train_loss = logs.get(
                        "loss", 0.0
                    )  # Get the training loss from logs
                    val_loss = logs.get(
                        "val_loss", 0.0
                    )  # Get the validation loss from logs
                    pbar.set_postfix(
                        {
                            "train_loss": f"{train_loss:.4e}",
                            "val_loss": f"{val_loss:.4e}",
                        }
                    )
                    self.train_losses.append(train_loss)
                    self.val_losses.append(val_loss)
                    pbar.update(1)

                self.model.fit(
                    X_scaled,
                    y_scaled,
                    callbacks=[
                        early_stopping,
                        LambdaCallback(on_epoch_end=on_epoch_end),
                    ],
                    verbose=0,
                    *args,
                    **kwargs,
                )

            if self.temp_file is not None:
                self.save(self.temp_file)

    def save(self, filename):
        """
        Save IntegratedModel attributes

        Parameters:
            filename (str): filename tag (without suffix) where model will be saved
        """
        # Save the Keras model in the .h5 format to handle custom objects like CustomActivation
        self.model.save(filename + "_model.h5")

        # attributes, excluding the model which we've saved separately
        attributes = [
            self.input_scaler,
            self.output_scaler,
            self.offset,
            self.log_preprocess,
            self.pca,
            self.pca_scaler,
            self.zero_columns,
            self.rescaling_factor,
        ]

        # Save the remaining attributes to file
        with open(filename + ".pkl", "wb") as f:
            pickle.dump(attributes, f)

    def restore(self, filename):
        """
        Load pre-saved IntegratedModel attributes

        Parameters:
            filename (str): filename tag (without suffix) where model was saved
        """
        # Load the Keras model from the .h5 format
        # self.model = load_model(filename + "_model.h5", custom_objects={'CustomActivation': CustomActivation,'CustomLoss': CustomLoss}, compile=False)
        self.model = load_model(
            filename + "_model.h5",
            custom_objects={
                "CustomLoss": CustomLoss,
                "CustomActivation": CustomActivation,
            },
            compile=False,
        )

        # Load the remaining attributes
        with open(filename + ".pkl", "rb") as f:
            attributes = pickle.load(f)

            try:
                (
                    self.input_scaler,
                    self.output_scaler,
                    self.offset,
                    self.log_preprocess,
                    self.pca,
                    self.pca_scaler,
                    self.zero_columns,
                    self.rescaling_factor,
                ) = attributes

            except:
                print("old model")
                (
                    self.input_scaler,
                    self.output_scaler,
                    self.offset,
                    self.log_preprocess,
                    self.pca,
                    self.zero_columns,
                    self.rescaling_factor,
                ) = attributes

        self.scaler_mean_in = self.input_scaler.mean_  # Mean of the features
        self.scaler_scale_in = (
            self.input_scaler.scale_
        )  # Standard deviation of the features

        self.scaler_scale_out = (
            self.output_scaler.scale_
        )  # Standard deviation of the features
        self.scaler_mean_out = (
            self.output_scaler.mean_
        )  # Standard deviation of the features

        self.scaler_mean_in_tf = tf.constant(self.scaler_mean_in)
        self.scaler_scale_in_tf = tf.constant(self.scaler_scale_in)

        self.scaler_mean_out_tf = tf.constant(self.scaler_mean_out)
        self.scaler_scale_out_tf = tf.constant(self.scaler_scale_out)

        try:
            self.pca_scaler_mean = self.pca_scaler.mean_
            self.pca_scaler_scale = self.pca_scaler.scale_

            self.pca_scaler_mean_tf = tf.constant(self.pca_scaler_mean)
            self.pca_scaler_scale_tf = tf.constant(self.pca_scaler_scale)

        except:
            print("no pca")
            self.pca_scaler_mean = None
            self.pca_scaler_scale = None

            self.pca_scaler_mean_tf = None
            self.pca_scaler_scale_tf = None

            self.pca_tensor = None
            self.mean_tensor = None

        print("restore successful")
        # print("rescaling factor", self.rescaling_factor)
