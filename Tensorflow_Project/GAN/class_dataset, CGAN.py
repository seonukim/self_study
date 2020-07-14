import os
import time
import tqdm
import warnings ; warnings.filterwarnings('ignore')
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.misc import imresize
from random import shuffle
from distutils.version import LooseVersion


## Define class Dataset
class Dataset(object):
    def __init__(self, data, labels = None, width = 28, height = 28, max_value = 255, channels = 3):
        # Record image specification.
        self.IMAGE_WITH = width
        self.IMAGE_HEIGHT = height
        self.IMAGE_MAX_VALUE = float(max_value)
        self.CHANNELS = channels
        self.shape = len(data), self.IMAGE_WITH, self.IMAGE_HEIGHT, self.CHANNELS
        if self.CHANNELS == 3:
            self.image_mode = 'RGB'
            self.cmap = None
        elif self.CHANNELS == 1:
            self.image_mode = 'L'
            self.cmap = 'gray'

        # If the image is different in size, resize it.
        if data.shape[1] != self.IMAGE_HEIGHT or data.shape[2] != self.IMAGE_WITH:
            data = self.image_resize(data, self.IMAGE_HEIGHT, self.IMAGE_WITH)
        # Store mixed data separately.
        index = list(range(len(data)))
        shuffle(index)
        self.data = data[index]

        if len(labels) > 0:
            # Save mixed labels separately.
            self.labels = labels[index]
            # List unique category values
            self.classes = np.unique(labels)
            # Create a one-hot encoding for each category based on the location in self.classes.
            one_hot = dict()
            no_classes = len(self.classes)
            for j, i in enumerate(self.classes):
                one_hot[i] = np.zeros(no_classes)
                one_hot[j] = 1.0
            self.one_hot = one_hot
        else:
            # Keep label variables as placeholders.
            self.labels = None
            self.classes = None
            self.one_hot = None
    
    def image_resize(self, dataset, newHeight, newWidth):
        # If necessary, resize image.
        channels = dataset.shape[3]
        images_resized = np.zeros([0, newHeight, newWidth, channels], dtype = np.uint8)
        for image in range(dataset.shape[0]):
            if channels == 1:
                temp = imresize(dataset[image][:, :, 0], [newHeight, newWidth], 'nearest')
                temp = np.expand_dims(temp, axis = 2)
            else:
                temp = imresize(dataset[image], [newHeight, newWidth], 'nearest')
            images_resized = np.append(images_resized, np.expand_dims(temp, axis = 0), axis = 0)
        return images_resized

    def get_batches(self, batch_size):
        # Importing batch of images and labels.
        current_index = 0
        # Verify that batch remains to be imported.
        while current_index < self.shape[0]:
            if current_index + batch_size > self.shape[0]:
                batch_size = self.shape[0] - current_index
            data_batch = self.data[current_index:current_index + batch_size]

            if len(self.labels) > 0:
                y_batch = np.array([self.one_hot[k] for k in \
                                    self.labels[current_index:current_index + batch_size]])
            else:
                y_batch = np.array([])
            current_index += batch_size
            yield (data_batch / self.IMAGE_MAX_VALUE) - 0.5, y_batch

# Define class CGAN
class CGAN(object):
    def __init__(self, dataset, epochs = 1, batch_size = 32,
                 z_dim = 96, generator_name = 'generator', alpha = 0.2,
                 smooth = 0.1, learning_rate = 0.001, beta1 = 0.35):
        # As a first step, ensure that the system can perform a GAN.
        self.check_system()
        # Set Key Parameters.
        self.generator_name = generator_name
        self.dataset = dataset
        self.cmap = self.dataset.cmap
        self.image_mode = self.dataset.image_mode
        self.epochs = epochs
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.alpha = alpha
        self.smooth = smooth
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.g_vars = list()
        self.trained = False
    
    def check_system(self):
        # Ensure that the system is adequate to carry out the project.
        # Ensure TensorFlo version is higher than 1.2.
        version = tf.__version__
        print(f'Tensorflow Version : {version}')

        assert LooseVersion(version) >= LooseVersion('1.2'), ('You are using %s, please use TensorFlow version 1.2 or newer.' % version)

        # Ensure GPU is present.
        if not tf.test.gpu_device_name():
            warnings.warn('No GPU found installed on the system. \
                          It is advised to train your GAN using a GPU or on AWS.')
        else:
            print('Default GPU Device: %s' % tf.test.gpu_device_name())
    
    def instantiate_inputs(self, image_width, image_height, image_channels, z_dim, classes):
        '''
        Instantiate Input and Parameter Placeholder:
        actual input for image creation (inputs_real), z input (inputs_z)
        actual input labels (labels), learning_rate
        '''
        inputs_real = tf.compat.v1.placeholder(tf.float32,
                                               (None, image_width, image_height,
                                                image_channels), name = 'input_real')
        inputs_z = tf.compat.v1.placeholder(tf.float32,
                                            (None, z_dim + classes), name = 'input_z')
        labels = tf.compat.v1.placeholder(tf.float32,
                                          (None, image_width, image_height, classes), name = 'labels')
        learning_rate = tf.compat.v1.placeholder(tf.float32, None)
        return inputs_real, inputs_z, labels, learning_rate
    
    def leaky_ReLU_activation(self, x, alpha = 0.2):
        return tf.compat.v1.maximum(alpha * x, x)
    
    def dropout(self, x, keep_prob = 0.9):
        return tf.nn.dropout(x, keep_prob)
    
    def d_conv(self, x, filters, kernel_size, strides,
               padding = 'same', alpha = 0.2, keep_prob = 0.5, train = True):
        '''
        Discriminant Layer Architecture
        Applies create Convolution, batch_normalization, leaky_relu function, dropout
        '''
        x = tf.compat.v1.layers.conv2d(x, filters, kernel_size, strides, padding,
                                       kernel_initializer = tf.contrib.layers.xavier_initializer())
        x = tf.compat.v1.layers.batch_normalization(x, training = train)
        x = self.leaky_ReLU_activation(x, alpha)
        x = self.dropout(x, keep_prob)
        return x
    
    def g_reshaping(self, x, shape, alpha = 0.2, keep_prob = 0.5, train = True):
        '''
        Generator layer architecture
        Applies feature change layer, batch_normalization, leaky_relu activation function, dropout
        '''
        x = tf.compat.v1.reshape(x, shape)
        x = tf.compat.v1.layers.batch_normalization(x, training = train)
        x = self.leaky_ReLU_activation(x, alpha)
        x = self.dropout(x, keep_prob)
        return x

    def g_conv_transpose(self, x, filters, kernel_size, strides,
                         padding = 'same', alpha = 0.2, keep_prob = 0.5, train = True):
        '''
        generator layer architecture
        transpose the Convolution to a new size
        Applies feature change layer, batch_normalization, leaky_relu activation function, dropout
        '''
        x = tf.compat.v1.layers.conv2d_transpose(x, filters, kernel_size, strides, padding)
        x = tf.compat.v1.layers.batch_normalization(x, training = train)
        x = self.leaky_ReLU_activation(x, alpha)
        x = self.dropout(x, keep_prob)
        return x

    def discriminator(self, images, labels, reuse = False):
        with tf.compat.v1.variable_scope('discriminator', reuse = reuse):
            # Input layer is 28*28*3 --> Connect input
            x = tf.compat.v1.concat([images, labels], 3)

            # d_conv --> Result Size is 14*14*32
            x = self.d_conv(x, filters = 32, kernel_size = 5,
                            strides = 2, padding = 'same',
                            alpha = 0.2, keep_prob = 0.5)
            
            # d_conv --> Result Size is 7*7*64
            x = self.d_conv(x, filters = 64, kernel_size = 5,
                            strides = 2, padding = 'same',
                            alpha = 0.2, keep_prob = 0.5)

            # d_conv --> Result Size is 7*7*128
            x = self.d_conv(x, filters = 128, kernel_size = 5,
                            strides = 1, padding = 'same',
                            alpha = 0.2, keep_prob = 0.5)
            
            # Flatten into one layer --> Estimated size is 4096
            x = tf.compat.v1.reshape(x, (-1, 7 * 7 * 128))

            # Calculate logit and sigmoid
            logits = tf.compat.v1.layers.dense(x, 1)
            sigmoids = tf.compat.v1.sigmoid(logits)

            return sigmoids, logits
    
    def generator(self, z, out_channel_dim, is_train = True):
        with tf.compat.v1.variable_scope('generator', reuse = (not is_train)):
            # First, configure a full connected layer
            x = tf.compat.v1.layers.dense(z, 7 * 7 * 512)

            # Change the shape of the full connected layer to start the Convolution stack.
            x = self.g_reshaping(x, shape = (-1, 7, 7, 512),
                                 alpha = 0.2, keep_prob = 0.5,
                                 train = is_train)
            
            # g_conv_transpose --> 7 * 7 * 128
            x = self.g_conv_transpose(x, filters = 256, kernel_size = 5,
                                      strides = 2, padding = 'same',
                                      alpah = 0.2, keep_prob = 0.5,
                                      train = is_train)
            
            # g_conv_transpose --> 14 * 14 * 64
            x = self.g_conv_transpose(x, filters = 128, kernel_size = 5,
                                      strides = 2, padding = 'same',
                                      alpha = 0.2, keep_prob = 0.5,
                                      train = is_train)
            
            # Calculate logit and output layer --> 28 * 28 * 5
            logits = tf.compat.v1.layers.conv2d_transpose(x, filters = out_channel_dim,
                                                          kernel_size = 5, strides = 1,
                                                          padding = 'same')
            output = tf.compat.v1.tanh(logits)
            return output

    def loss(self, input_real, input_z, labels, out_channel_dim):
        # Generate output
        g_output = self.generator(input_z, out_channel_dim)
        # Classification of actual inputs
        d_output_real, d_logits_real = self.discriminator(input_real, labels, reuse = False)
        # Classify generated output
        d_output_fake, d_logits_fake = self.discriminator(g_output, labels, reuse = True)
        # Calculate the loss of actual input classification results
        real_input_labels = tf.compat.v1.ones_like(d_output_real) * (1 - self.smooth)   # smoothed value
        d_loss_real = tf.compat.v1.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits = d_logits_real, lables = real_input_labels))
        # Calculate the loss of generated output classification results
        fake_input_labels = tf.compat.v1.zeros_like(d_output_fake)  # just zeros
        d_loss_fake = tf.compat.v1.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits = d_logits_fake, labels = fake_input_labels))
        # Combined actual input classification loss with output classification loss generated
        d_loss = d_logits_real + d_logits_fake      # Total Losses on the Classifier
        # Calculate the loss of the generator: All generated images must be classified as true by the classifier
        target_fake_input_labels = tf.compat.v1.ones_like(d_logits_fake)        # all
        g_loss = tf.compat.v1.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits = d_logits_fake, lables = target_fake_input_labels))
        return d_loss, g_loss

    def rescale_images(self, image_array):
        # Fit image scale to 0-255 range
        new_array = image_array.copy().astype(float)
        min_value = new_array.min()
        range_value = new_array.max() - min_value
        new_array = ((new_array - min_value) / range_value) * 255
        return new_array.astype(np.uint8)
    
    def images_grid(self, images, n_cols):
        # Align images to a grid suitable for plotting
        # Gets the image size and defines the shape of the grid
        n_images, height, width, depth = images.shape
        n_rows = n_images // n_cols
        projected_images = n_rows * n_cols
        # Fit image scale to 0-255 range
        images = self.rescale_images(images)
        # Adjust if there are fewer projected images
        if projected_images < n_images:
            images = images[:projected_images]
        # Arrange images in square form
        square_grid = images.reshape(n_rows, n_cols, height, width, depth)
        square_grid = square_grid.swapaxes(1, 2)
        # Return Grid Image
        if depth >= 3:
            return square_grid.reshape(height * n_rows, width * n_cols, depth)
        else:
            return square_grid.reshape(height * n_rows, width * n_cols)
        
    def plotting_images_grid(self, n_images, samples):
        # Representing images in the grid
        n_cols = tf.compat.v1.math.floor(tf.compat.v1.math.sqrt(n_images))
        images_grid = self.images_grid(samples, n_cols)
        plt.imshow(images_grid, cmap = self.cmap)
        plt.show()

    def show_generator_output(self, sess, n_images, input_z, labels, out_channel_dim, image_mode):
        # Show samples made by the actual generator.
        # Generate input_z for examples
        z_dim = input_z.get_shape().as_list()[-1]
        example_z = np.random.uniform(-1, 1, size = [n_images, z_dim - labels.shape[1]])
        example_z = np.concatenate((example_z, labels), axis = 1)
        # Run the generator
        sample = sess.run(self.generator(input_z, out_channel_dim, False),
                          feed_dict = {input_z: example_z})
        # Draw a sample
        self.plotting_images_grid(n_images, sample)

    def show_original_images(self, n_images):
        # Show original image sample
        # Sampling from Available Images
        index = np.random.randint(self.dataset.shape[0], size = (n_images))
        sample = self.dataset.data[index]
        # Draw a sample
        self.plotting_images_grid(n_images, sample)
    
    def optimization(self):
        # GAN Optimization Procedure
        # initialize input, placeholder of parameter
        cases, image_width, image_height, out_channel_dim = self.dataset.shape
        input_real, input_z, labels, learn_rate = \
            self.instantiate_inputs(image_width, image_height,
                                    out_channel_dim, self.z_dim,
                                    len(self.dataset.classes))

        # Define network and calculate loss
        d_loss, g_loss = self.loss(input_real, input_z, labels, out_channel_dim)

        # Enumerate trainable_variables, Divide into G and D parts
        d_vars = [v for v in tf.compat.v1.trainable_variables() \
                    if v.name.startswith('discriminator')]
        g_vars = [v for v in tf.compat.v1.trainable_variables() \
                    if v.name.startswith('generator')]
        self.g_vars = g_vars

        # First optimize the classifier, then optimize the generator.
        with tf.compat.v1.control_dependencies(
            tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.UPDATE_OPS)):
                d_train_opt = tf.compat.v1.train.AdamOptimizer(
                    self.learning_rate, self.beta1).minimize(d_loss, var_list = d_vars)
                g_train_opt = tf.compat.v1.train.AdamOptimizer(
                    self.learning_rate, self.beta1).minimize(g_loss, var_list = g_vars)
                return input_real, input_z, labels, learn_rate, d_loss, g_loss, d_train_opt, g_train_opt

    # Step of train
    def train(self, save_every_n = 1000):
        losses = []
        step = 0
        epoch_count = self.epochs
        batch_size = self.batch_size
        z_dim = self.z_dim
        learning_rate = self.learning_rate
        get_batches = self.dataset.get_baches
        classes = len(self.dataset.classes)
        data_image_mode = self.dataset.image_mode

        cases, image_width, image_height, out_channel_dim = self.dataset.shape
        input_real, input_z, labels, learn_rate, d_loss, g_loss, d_train_opt, g_train_opt = \
            self.optimization()
        
        # Save the trained GAN
        saver = tf.compat.v1.train.Saver(var_list = self.g_vars)

        # Prepare the mask for plotting procedure
        rows, cols = min(5, classes), 5
        target = np.array([self.dataset.one_hot[i] \
                    for j in range(cols) for i in range(rows)])
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            for epoch_i in range(epoch_count):
                for batch_images, batch_labels in get_batches(batch_size):
                    # Count the step
                    step += 1
                    # Define Z
                    batch_z = np.random.uniform(-1, 1, size = (len(batch_images), z_dim))
                    batch_z = np.concatenate((batch_z, batch_labels), axis = 1)
                    # Adjust shape of label for generator
                    batch_labels = batch_labels.reshape(batch_size, 1, 1, classes)
                    batch_labels = batch_labels * np.ones((batch_size, image_width, image_height, classes))
                    # Sampling random noise for G
                    batch_images = batch_images * 2
                    # Run the optimized model
                    _ = sess.run(d_train_opt, feed_dict = {input_real: batch_images, input_z: batch_z, \
                                                            labels: batch_labels, learn_rate: learning_rate})
                    _ = sess.run(g_train_opt, feed_dict = {input_z: batch_z, input_real: batch_images, \
                                                            labels: batch_labels, learn_rate: learning_rate})
                    # Report the fitted results and generator output to the intersection
                    if step % (save_every_n / 10) == 0:
                        train_loss_d = sess.run(d_loss, {input_z: batch_z,
                                                         input_real: batch_images,
                                                         labels: batch_labels})
                        train_loss_g = g_loss.eval({input_z: batch_z, labels: batch_labels})
                        print("Epoch %i/%i step %i..." % (epoch_i + 1, epoch_count, step),
                              "Discriminator Loss: %0.3f..." % train_loss_d,
                              "Generator Loss: %0.3f" % train_loss_g)
                    if step % save_every_n == 0:
                        rows = min(5, classes)
                        cols = 5
                        target = np.array([self.dataset.one_hot[i] for j in range(cols) \
                                            for i in range(rows)])
                        self.show_generator_output(sess, rows * cols, input_z, target,
                                                   out_channel_dim, data_image_mode)
                        saver.save(sess, './' + self.generator_name + '/generator.ckpt')
                    
                # Calculate and output losses at the end of each generation
                try:
                    train_loss_d = sess.run(d_loss, {input_z: batch_z,
                                                     input_real: batch_images,
                                                     labels: batch_labels})
                    train_loss_g = g_loss.eval({input_z: batch_z, labels: batch_labels})
                    print("Epoch %i/%i step %i..." % (epoch_i + 1, epoch_count, step),
                          "Discriminator Loss: %0.3f..." % train_loss_d,
                          "Generator Loss: %0.3f" % train_loss_g)
                except:
                    train_loss_d, train_loss_g = -1, -1
                
                # Save loss values to be reported after the training is over
                losses.append([train_loss_d, train_loss_g])
            
            # Result of final generator
            self.show_generator_output(sess, rows * cols, input_z, target, out_channel_dim, data_image_mode)
            saver.save(sess, './' + self.generator_name + '/generator.ckpt')
        
        return np.array(losses)

    def generator_new(self, target_class = -1, rows = 5, cols = 5, plot = True):
        # Generating a new sample
        # Fixing minimum rows and cols values
        rows, cols = max(1, rows), max(1, cols)
        n_images = rows * cols
        # Checking if we already have a Tensorflow graph
        if not self.trained:
            # Complete restoration of tensorflow graphs
            tr.reset_default_graph()
            self._session = tf.compat.v1.Session()
            self._classes = len(self.dataset.classes)
            self._input_z = tf.compat.v1.placeholder(tf.float32, shape = (None, self.z_dim + \
                                                     self._classes), name = 'input_z')
            out_channel_dim = self.dataset.shape[3]
            # Restoring the generator graph
            self._generator = self.generator(self._input_z, out_channel_dim)
            g_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]
            saver = tf.train.Saver(var_list=g_vars)
            print('Restoring generator graph')
            saver.restore(self._session, tf.train.latest_checkpoint(self.generator_name))
            # Setting trained flag as True
            self.trained = True
        
        # Continuing the session
        sess = self._session
        # Building an array of examples examples
        target = np.zeros((n_images, self._classes))
        for j in range(cols):
            for i in range(rows):
                if target_class == -1:
                    target[j * cols + i, j] = 1.0
                else:
                    target[j * cols + i] = self.dataset.one_hot[target_class].tolist()
        # Generating the random input
        z_dim = self._input_z.get_shape().as_list()[-1]
        example_z = np.random.uniform(-1, 1, size=[n_images, z_dim - target.shape[1]])
        example_z = np.concatenate((example_z, target), axis=1)
        # Generating the images
        sample = sess.run(
            self._generator,
            feed_dict={self._input_z: example_z})
        # Plotting
        if plot:
            if rows * cols==1:
                if sample.shape[3] <= 1:
                    images_grid = sample[0,:,:,0]
                else:
                    images_grid = sample[0]
            else:
                images_grid = self.images_grid(sample, cols)
            plt.imshow(images_grid, cmap=self.cmap)
            plt.show()
        # Returning the sample for later usage (and not closing the session)
        return sample
    
    def fit(self, learning_rate = 0.0002, beta1 = 0.35):
        # Fit procedure, starting training and result storage
        # Setting training parameters
        self.learning_rate = learning_rate
        self.beta1 = beta1
        # Training generator and discriminator
        with tf.compat.v1.Graph().as_default():
            train_loss = self.train()
        # Plotting training fitting
        plt.plot(train_loss[:, 0], label = 'Discriminator')
        plt.plot(train_loss[:, 1], label = 'Generator')
        plt.title('Training fitting')
        plt.legend()
