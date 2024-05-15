from jax.numpy import array, ones, dot, multiply, subtract, exp, log, add, float32, zeros, delete, arange
import tensorflow as tf
from tensorflow.keras.layers import Layer
from sklearn.preprocessing import StandardScaler #for scaling input and output data
from tensorflow.keras.models import load_model
from tqdm import tqdm
import pickle
from flax import linen as nn
from jax.nn import sigmoid
from flax import traverse_util
import jax.random as random
import jax 


class MyNetwork(nn.Module):
    weights: list  # List of (weight, bias) tuples for each layer
    hyper_params: list  # List of (alpha, beta) for each custom activation layer

    @nn.compact
    def __call__(self, x):
        # Loop over layers except the last one
        for (w, b), (a, b_hyper) in zip(self.weights[:-1], self.hyper_params):
            x = CustomActivation_jax(a=a, b=b_hyper)(dot(x, w) + b)

        # Final layer (no activation)
        final_w, final_b = self.weights[-1]
        x = dot(x, final_w) + final_b
        return x


class CustomActivation_jax(nn.Module):
    a: float  # alpha hyperparameter
    b: float  # beta hyperparameter

    @nn.compact
    def __call__(self, x):
        return multiply(add(self.b, multiply(sigmoid(multiply(self.a, x)), subtract(1., self.b))), x)

#Custom activation function to match https://arxiv.org/abs/2106.03846 for original tf model 
class CustomActivation(Layer):
    def build(self, input_shape):
        # Trainable weight variables for alpha and beta initialized with random normal distribution
        self.alpha = self.add_weight(name='alpha', shape=input_shape[1:], initializer='random_normal', trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[1:], initializer='random_normal', trainable=True)
        super(CustomActivation, self).build(input_shape)

    def call(self, inputs):
        return (self.beta + tf.sigmoid(self.alpha * inputs) * (1 - self.beta)) * inputs

# Custom loss function for original tf model
class CustomLoss(Layer):
    def __init__(self, element_weights=None, **kwargs):
        super(CustomLoss, self).__init__(**kwargs)
        self.element_weights = tf.convert_to_tensor(element_weights, dtype=tf.float32) if element_weights is not None else tf.constant(1.0, dtype=tf.float32)

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


def insert_zero_columns(prediction, zero_columns_indices):
    # Total number of columns in the final output
    num_columns = prediction.shape[1] + len(zero_columns_indices)

    # Initialize the final_prediction array
    final_prediction = zeros((prediction.shape[0], num_columns))

    # Indices in final_prediction where values from prediction will be inserted
    non_zero_indices = delete(arange(num_columns), zero_columns_indices)

    # Insert values from prediction into the correct positions in final_prediction
    final_prediction = final_prediction.at[:, non_zero_indices].set(prediction)

    return final_prediction

class IntegratedModel:
    def __init__(self, keras_model, input_scaler, output_scaler, temp_file=None, offset=None, log_preprocess=False, zero_columns=None, rescaling_factor=None, pca=None, pca_scaler=None, verbose=False):
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
        self.verbose = verbose


    def predict(self, data):
        scaled_data = (data - self.scaler_mean_in)/self.scaler_scale_in
        # prediction = self.model(tf.convert_to_tensor(scaled_data, dtype=tf.float32))

        # Convert to TensorFlow tensor
        scaled_data = array(scaled_data, dtype=float32)
        prediction = self.jax_model.apply(self.jax_params, scaled_data)
        # Apply PCA transform if PCA is used
        if self.pca is not None:
            prediction = prediction*self.scaler_scale_out + self.scaler_mean_out
            prediction = self.pca.inverse_transform(prediction)
            prediction = self.pca_scaler.inverse_transform(prediction)
        else:
            prediction = prediction*self.scaler_scale_out + self.scaler_mean_out

        #Inverse offset and log
        if self.log_preprocess:
            prediction = exp(prediction) + 2*self.offset

        if self.zero_columns is not None:
            #re-inset columns of zeros using the indices of zeros from the original array size
            final_prediction = insert_zero_columns(prediction, self.zero_columns)

            # output_shape = (prediction.shape[0], prediction.shape[1] + len(self.zero_columns))

            # # Create a boolean mask for the columns
            # all_cols = arange(output_shape[1])
            # mask = ~isin(all_cols, self.zero_columns)

            # # Create an array of zeros with the original shape
            # final_prediction = zeros(output_shape)

            # # Fill in the data from the reduced array using boolean indexing
            # final_prediction.at[:, mask].set(prediction)

        else:
            final_prediction = prediction


        return final_prediction

    def restore(self, filename):
        """
        Load pre-saved IntegratedModel attributes'

        Transform tf weights to jax weights dynamically and initialize the jax model

        Parameters:
            filename (str): filename tag (without suffix) where model was saved
        """
        # Load the Keras model from the .h5 format
        tf_model = load_model(filename + "_model.h5", custom_objects={'CustomActivation': CustomActivation, 'CustomLoss': CustomLoss}, compile=False)

        tf_params = []
        for layer in tf_model.layers:
            weights = layer.get_weights()
            if weights:  # Check if the layer has parameters
                tf_params.append(weights)

        #load in the weights and hyperparameters
        tf_weights = []
        tf_hyperparams = []
        for i, layer in enumerate(tf_params):
            if i%2 == 0:
                tf_weights.append(tf_params[i])

            else:
                tf_hyperparams.append(tf_params[i])

        # Load the remaining attributes
        with open(filename + ".pkl", 'rb') as f:
            attributes = pickle.load(f)

            try:
                self.input_scaler, self.output_scaler, self.offset, \
                self.log_preprocess, self.pca, self.pca_scaler, self.zero_columns, self.rescaling_factor = attributes

            except:
                if self.verbose:
                    print("old model")
                self.input_scaler, self.output_scaler, self.offset, \
                self.log_preprocess, self.pca, self.zero_columns, self.rescaling_factor = attributes

        self.scaler_mean_in = self.input_scaler.mean_   # Mean of the features
        self.scaler_scale_in = self.input_scaler.scale_ # Standard deviation of the features


        self.num_zero_columns = len(self.zero_columns) if self.zero_columns is not None else 0


        self.scaler_scale_out = self.output_scaler.scale_ # Standard deviation of the features
        self.scaler_mean_out = self.output_scaler.mean_ # Standard deviation of the features


        try:
            self.pca_scaler_mean = self.pca_scaler.mean_
            self.pca_scaler_scale = self.pca_scaler.scale_

        except:
            if self.verbose:
                print("no pca")

        self.jax_model = MyNetwork(hyper_params=tf_hyperparams, weights=tf_weights)

        # Initialize the model with dummy data
        input_shape = tf_weights[0][0].shape[0] #the input shape of the first layer
        rng = random.PRNGKey(0)
        dummy_input = ones((1, input_shape)) 
        params = self.jax_model.init(rng, dummy_input)

        # Flatten the parameters dictionary
        flattened_params = traverse_util.flatten_dict(params)

        # Replace the initialized parameters with your pre-trained weights and hyperparameters
        for i, layer_path in enumerate(flattened_params.keys()):
            if "CustomActivation" in layer_path:  # Assuming this is the name of your custom activation layers
                # Replace hyperparameters
                flattened_params[layer_path] = tf_hyperparams[i]
            else:
                # Replace weights and biases
                flattened_params[layer_path] = tf_weights[i]

        # Unflatten the parameters dictionary
        self.jax_params = traverse_util.unflatten_dict(flattened_params)

        if self.verbose:
            print("restore successful")
