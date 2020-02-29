"""
-------------------------------------------------------------------------------------------------
    Tel Aviv University's - Deep Learning Course - 0510-7255 - Final Project
                        By: Nir Aharon, Stanislav Bromberg

    This code is our implementation for the paper:
    "Multimodal matching using a Hybrid Convolutional Neural", by Elad Ben Baruch, Yosi Keller.

    Go to the 'main' for choosing the configurations for the running.
-------------------------------------------------------------------------------------------------
"""

# Imports
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import sys
import time
import pandas


# Network class
class MultimodalHybridCNN:
    def __init__(self):
        self.patchSize = 64
        self.inputShape = (self.patchSize, self.patchSize, 1)
        self.weightDecay = tf.keras.regularizers.l2(0.0005)
        self.lossType = 'L2'

        self.siameseModel = []
        self.asymmetricModelX = []
        self.asymmetricModelY = []
        self.hybridModel = []

        self.HybridCNN()
        self.Rotation90Matrix = cv2.getRotationMatrix2D((int(self.patchSize / 2), int(self.patchSize / 2)), 90, 1)
        self.Rotation_90Matrix = cv2.getRotationMatrix2D((int(self.patchSize / 2), int(self.patchSize / 2)), -90, 1)
        self.identityWarp = np.float32([[1, 0, 0], [0, 1, 0]])

        self.lossHistory = []
        self.LsHistory = []
        self.LaHistory = []
        self.LhHistory = []

    def baseCNN(self, backtrack=False):

        featureMap = []
        inputs = tf.keras.Input(shape=self.inputShape, name='input_patch')

        # Conv0
        x = tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu',
                                   kernel_regularizer=self.weightDecay, bias_regularizer=self.weightDecay)(inputs)
        # Pooling0
        x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

        if backtrack:
            x_c = tf.keras.layers.UpSampling2D((2, 2), interpolation='bilinear')(x)
            x_c = tf.math.reduce_max(x_c, 3)
            x_c = tf.math.divide(x_c, tf.math.reduce_max(x_c))

            featureMap = x_c

        # Conv1
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu',
                                   kernel_regularizer=self.weightDecay, bias_regularizer=self.weightDecay)(x)

        # Pooling1
        x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

        if backtrack:
            x_c = tf.keras.layers.UpSampling2D((4, 4), interpolation='bilinear')(x)
            x_c = tf.math.reduce_max(x_c, 3)
            x_c = tf.math.divide(x_c, tf.math.reduce_max(x_c))
            featureMap = tf.keras.layers.add([featureMap, x_c])

        # Conv2
        x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                                   kernel_regularizer=self.weightDecay, bias_regularizer=self.weightDecay)(x)
        # Pooling2
        x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

        if backtrack:
            x_c = tf.keras.layers.UpSampling2D((8, 8), interpolation='bilinear')(x)
            x_c = tf.math.reduce_max(x_c, 3)
            x_c = tf.math.divide(x_c, tf.math.reduce_max(x_c))
            featureMap = tf.keras.layers.add([featureMap, x_c])

        # Conv3
        x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu',
                                   kernel_regularizer=self.weightDecay, bias_regularizer=self.weightDecay)(x)

        # Conv4
        x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu',
                                   kernel_regularizer=self.weightDecay, bias_regularizer=self.weightDecay)(x)

        if backtrack:
            x_c = tf.keras.layers.UpSampling2D((16, 16), interpolation='bilinear')(x)
            x_c = tf.math.reduce_max(x_c, 3)
            x_c = tf.math.divide(x_c, tf.math.reduce_max(x_c))
            featureMap = tf.keras.layers.add([featureMap, x_c])

        if self.lossType == 'L2':
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(128, kernel_regularizer=self.weightDecay, bias_regularizer=self.weightDecay)(x)
            outputs = tf.keras.layers.LayerNormalization(axis=1)(x)

        if backtrack:
            return tf.keras.Model(inputs, [outputs, featureMap])

        return tf.keras.Model(inputs, outputs)

    def SiameseCNN(self, input_x, input_y):

        self.siameseModel = self.baseCNN(backtrack=True)
        # self.siameseModel.summary()

        return self.siameseModel(input_x), self.siameseModel(input_y)

    def AsymmetricCNN(self, input_x, input_y):

        self.asymmetricModelX = self.baseCNN(backtrack=True)
        # self.asymmetricModelX.summary()

        self.asymmetricModelY = self.baseCNN(backtrack=True)
        # self.asymmetricModelY.summary()

        return self.asymmetricModelX(input_x), self.asymmetricModelY(input_y)

    def HybridCNN(self):

        # Inputs
        input_x = tf.keras.Input(shape=self.inputShape, name='input_x')
        input_y = tf.keras.Input(shape=self.inputShape, name='input_y')

        # Siamese Branch
        siamese_x, siamese_y = self.SiameseCNN(input_x, input_y)
        siamese_x_map = siamese_x[1]
        siamese_y_map = siamese_y[1]

        siamese_x = siamese_x[0]
        siamese_y = siamese_y[0]

        # Asymmetric branch
        asymmetric_x, asymmetric_y = self.AsymmetricCNN(input_x, input_y)
        asymmetric_x_map = asymmetric_x[1]
        asymmetric_y_map = asymmetric_y[1]

        asymmetric_x = asymmetric_x[0]
        asymmetric_y = asymmetric_y[0]

        # Fuse branches
        Hx = tf.keras.layers.concatenate([siamese_x, asymmetric_x])
        Hy = tf.keras.layers.concatenate([siamese_y, asymmetric_y])

        x_representation = tf.keras.layers.Dense(128, kernel_regularizer=self.weightDecay,
                                                 bias_regularizer=self.weightDecay)(Hx)
        y_representation = tf.keras.layers.Dense(128, kernel_regularizer=self.weightDecay,
                                                 bias_regularizer=self.weightDecay)(Hy)

        model = tf.keras.Model(inputs=[input_x,
                                       input_y],
                               outputs=[x_representation,
                                        y_representation,
                                        siamese_x,
                                        siamese_y,
                                        asymmetric_x,
                                        asymmetric_y,
                                        siamese_x_map,
                                        siamese_y_map,
                                        asymmetric_x_map,
                                        asymmetric_y_map],
                               name="HybridCNN")

        self.hybridModel = model
        # model.summary()

    def L2HingeLoss(self, x, y, labels):

        labels = tf.convert_to_tensor(tf.squeeze(labels))
        distance = tf.keras.backend.square(x - y)

        distance = tf.keras.backend.sum(distance, axis=1)
        distance = tf.keras.backend.sqrt(distance)

        p_loss = labels * distance
        n_loss = (tf.keras.backend.ones_like(labels) - labels) * tf.keras.backend.relu(tf.keras.backend.ones_like(distance) - distance)
        loss = p_loss + n_loss

        return tf.keras.backend.mean(loss)

    def augmantation(self, image):

        r = np.random.randint(0, 6)

        if r == 0:  # Flip left right
            image = cv2.flip(image, 0)

        elif r == 1:  # Flip up down
            image = cv2.flip(image, 1)

        elif r == 2:  # Rotate 90
            image = cv2.warpAffine(image, self.Rotation90Matrix, (self.patchSize, self.patchSize))

        elif r == 3:  # Rotate -90
            image = cv2.warpAffine(image, self.Rotation_90Matrix, (self.patchSize, self.patchSize))

        return image

    def augmantations(self, image1, image2):

        r = np.random.randint(0, 5)

        if r == 0:  # Flip left right
            image1 = cv2.flip(image1, 0)
            image2 = cv2.flip(image2, 0)

        elif r == 1:  # Flip up down
            image1 = cv2.flip(image1, 1)
            image2 = cv2.flip(image2, 1)

        elif r == 2:  # Rotate 90
            image1 = cv2.warpAffine(image1, self.Rotation90Matrix, (self.patchSize, self.patchSize))
            image2 = cv2.warpAffine(image2, self.Rotation90Matrix, (self.patchSize, self.patchSize))

        elif r == 3:  # Rotate -90
            image1 = cv2.warpAffine(image1, self.Rotation_90Matrix, (self.patchSize, self.patchSize))
            image2 = cv2.warpAffine(image2, self.Rotation_90Matrix, (self.patchSize, self.patchSize))

        return image1, image2

    def getBatchDataWithLoad(self, datasetName, dataDir, df, idx, augmantations=1, hard_negative_mining=True):

        images_rgb = np.empty((len(idx), self.patchSize, self.patchSize))
        images_ir = np.empty((len(idx), self.patchSize, self.patchSize))
        labels = np.empty((len(idx),))

        # for i in idx:
        for i in range(len(idx)):

            if datasetName == 'VEDAI':
                imageRgbPath = dataDir + df['rgb'][idx[i]][:-3] + 'png'
                imageIrPath = dataDir + df['nir'][idx[i]][:-3] + 'png'

            else:
                sys.exit("ERROR: Unknown dataset")

            imageRgbX = int(df['rgb_x'][idx[i]] - self.patchSize / 2)
            imageRgbY = int(df['rgb_y'][idx[i]] - self.patchSize / 2)

            imageIrX = int(df['nir_x'][idx[i]] - self.patchSize / 2)
            imageIrY = int(df['nir_y'][idx[i]] - self.patchSize / 2)

            label_str = df['type'][idx[i]]
            label = int(label_str == 'positive')

            img_rgb = cv2.imread(imageRgbPath)
            if img_rgb.size == 0:
                print("ERROR: Unable to load: ", imageRgbPath)
                sys.exit("ERROR: Empty image")

            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
            rgb_h, rgb_w = np.shape(img_rgb)

            img_ir = cv2.imread(imageIrPath)
            if img_ir.size == 0:
                print("ERROR: Unable to load: ", imageIrPath)
                sys.exit("ERROR: Empty image")

            img_ir = cv2.cvtColor(img_ir, cv2.COLOR_RGB2GRAY)
            ir_h, ir_w = np.shape(img_ir)

            if imageRgbY + self.patchSize > rgb_h:
                imageRgbY += rgb_h - (imageRgbY + self.patchSize)

            if imageRgbX + self.patchSize > rgb_w:
                imageRgbX += rgb_w - (imageRgbX + self.patchSize)

            if imageIrY + self.patchSize > ir_h:
                imageIrY += ir_h - (imageIrY + self.patchSize)

            if imageIrX + self.patchSize > ir_w:
                imageIrX += ir_w - (imageIrX + self.patchSize)

            patch_rgb = img_rgb[imageRgbY:imageRgbY + self.patchSize, imageRgbX:imageRgbX + self.patchSize]
            patch_ir = img_ir[imageIrY:imageIrY + self.patchSize, imageIrX:imageIrX + self.patchSize]

            if augmantations == 1:
                patch_rgb, patch_ir = self.augmantations(patch_rgb, patch_ir)
            elif augmantations == 2:
                patch_rgb = self.augmantation(patch_rgb)
                patch_ir = self.augmantation(patch_ir)

            if np.size(patch_rgb) != self.patchSize * self.patchSize or np.size(
                    patch_ir) != self.patchSize * self.patchSize:
                sys.exit("ERROR: Wrong image division")

            images_rgb[i, :, :] = patch_rgb - np.mean(patch_rgb)
            images_ir[i, :, :] = patch_ir - np.mean(patch_ir)
            labels[i] = label

        images_rgb = np.expand_dims(images_rgb, 3).astype(np.float32)
        images_ir = np.expand_dims(images_ir, 3).astype(np.float32)
        labels = np.expand_dims(labels, 1).astype(np.float32)

        if hard_negative_mining:

            negatives_samples, _ = np.where(labels == 0)
            x_representation, y_representation, Ws_x, Ws_y, Wx_x, Wy_y, _, _, _, _ = self.hybridModel(
                [images_rgb, images_ir], training=False)

            hard_neg_idx = []
            for n in negatives_samples:
                closest = 1000000
                closest_idx = n

                for h in negatives_samples:
                    dist = np.linalg.norm(Ws_x[n] - Ws_y[h]) + np.linalg.norm(Wx_x[n] - Wy_y[h])

                    if dist < closest and h not in hard_neg_idx:
                        closest = dist
                        closest_idx = h

                hard_neg_idx.append(closest_idx)

            images_ir[negatives_samples, :, :, :] = images_ir[hard_neg_idx, :, :, :]

        return images_rgb.astype(np.float32), images_ir.astype(np.float32), labels.astype(np.float32)

    @tf.function
    def train_step(self, x_image, y_image, labels, opt):

        with tf.GradientTape() as gradTape:
            x_representation, y_representation, Ws_x, Ws_y, Wx_x, Wy_y, _, _, _, _ = self.hybridModel([x_image, y_image], training=True)

            Ls = self.L2HingeLoss(Ws_x, Ws_y, labels)
            La = self.L2HingeLoss(Wx_x, Wy_y, labels)
            Lh = self.L2HingeLoss(x_representation, y_representation, labels)

            Loss = Lh + Ls + La

        # Calculate gradients
        gradients = gradTape.gradient(Loss, self.hybridModel.trainable_variables)

        # Apply gradient step
        opt.apply_gradients(zip(gradients, self.hybridModel.trainable_variables))

        return Loss, Lh, Ls, La

    def train(self, datasetPath, csvTrainPath, dataSetName, epochs=40, batchSize=64, lr=0.01, momentum=0.9, augmantationsType=1, HNM=True):

        # Load csv file
        df = pandas.read_csv(csvTrainPath)
        numberOfPatches = df.shape[0]
        patchesIdx = np.arange(0, numberOfPatches)

        iterationPerEpoch = numberOfPatches // batchSize

        # initialize the optimizer
        opt = tf.keras.optimizers.SGD(lr=lr, momentum=momentum)

        # Set checkpoints
        checkpoint_dir = './training_checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint()

        for epoch in range(epochs):
            start = time.time()

            np.random.shuffle(patchesIdx)

            overallLoss = 0
            overallLs = 0
            overallLa = 0
            overallLh = 0

            batch_per_epoch = 0

            for iteration in range(iterationPerEpoch):

                rgbBatch, irBatch, labelsBatch = self.getBatchDataWithLoad(dataSetName, datasetPath, df, patchesIdx[iteration * batchSize:iteration * batchSize + batchSize], augmantations=augmantationsType, hard_negative_mining=HNM)

                Loss, Lh, Ls, La = self.train_step(rgbBatch, irBatch, labelsBatch, opt)

                if np.isnan(Loss):
                    sys.exit("ERROR: Loss is NaN\nBreak from code.")

                overallLoss += Loss
                overallLs += Ls
                overallLa += La
                overallLh += Lh
                batch_per_epoch += 1

            # Mean loss of epoch
            overallLoss /= batch_per_epoch
            overallLs /= batch_per_epoch
            overallLa /= batch_per_epoch
            overallLh /= batch_per_epoch

            print("Epoch %d: Overall Loss: %f, Lh: %f, Ls: %f, La: %f, Time for epoch: %d sec" % (epoch + 1, overallLoss, overallLh, overallLs, overallLa, time.time() - start))

            self.lossHistory.append(overallLoss)
            self.LsHistory.append(overallLs)
            self.LaHistory.append(overallLa)
            self.LhHistory.append(overallLh)

            # Save the model every 15 epochs
            if epoch % 1 == 0 or epoch == epochs:
                checkpoint.save(file_prefix=checkpoint_prefix)
                fileName = "hybridModel_Epoch" + str(epoch + 1) + ".h5"
                self.hybridModel.save(fileName)

        plt.figure()
        ax = plt.subplot(111)
        ax.plot(self.lossHistory, label='L')
        ax.plot(self.LsHistory, label='Ls')
        ax.plot(self.LaHistory, label='La')
        ax.plot(self.LhHistory, label='Lh')
        plt.grid()
        plt.title('Loss per epoch')
        ax.legend()
        plt.show()

    def load_Weights(self, weightsPath):
        print("Loading: ", weightsPath)
        self.hybridModel.load_weights(weightsPath)

    def evaluate_fpr(self, datasetName, dirPath, csvTest, batchSize=64):
        df = pandas.read_csv(csvTest)

        hybrid_scores = np.empty((len(df), 1))
        labels = np.array(df['type'] == 'positive', dtype=int)

        for b in range(int(np.ceil(len(df) / batchSize))):
            print("%d / %d" % (b, int(np.ceil(len(df) / batchSize))))

            # Extract batch
            idx = np.arange(b * batchSize, min((b + 1) * batchSize, len(df)))
            rgb_batch, ir_batch, labels_batch = self.getBatchDataWithLoad(datasetName, dirPath, df, idx, augmantations=False, hard_negative_mining=False)

            # Feed network
            x_representation, y_representation, Ws_x, Ws_y, Wx_x, Wy_y, siamese_features_x, siamese_features_y, asymetric_features_x, asymetric_features_y = self.hybridModel([rgb_batch, ir_batch], training=False)

            # Calculate and store distances
            hybrid_scores[idx, 0] = np.linalg.norm(x_representation - y_representation, axis=1)

        # Calculate FPR at 95% Recall
        fpr = self.fpr95recall(hybrid_scores, labels, 0.95)
        print("FPR at 95 Recall: ", fpr)

    def fpr95recall(self, scores, labels, recall_goal):
        scores = np.squeeze(scores)
        sorted_index = np.argsort(scores)

        if np.sum(sorted_index) == 0:
            sys.exit("ERROR: No argsort")

        number_of_true_matches = np.sum(labels == 1)

        print("number_of_true_matches = ", number_of_true_matches)

        threshold_number = recall_goal * number_of_true_matches
        tp = 0
        count = 0

        for si in sorted_index:

            count = count + 1.0

            print("count: %d, score: %f, label: %d" % (count, scores[si], labels[si]))

            if labels[si] == 1:
                tp = tp + 1.0

            if tp >= threshold_number:
                break

        far = ((count - tp) / count) * 100.0
        return far

    def evaluate_detections(self, datasetName, dirPath, csvTest, Ns=500):

        df = pandas.read_csv(csvTest)
        NUMBER_OF_IMAGES = len(df)

        # Calculate cumulative detection probability
        R = 10
        siamese_cum_det_prob = np.zeros((R, 1))
        asymmetric_cum_det_prob = np.zeros((R, 1))

        for i in range(NUMBER_OF_IMAGES):

            print("%d / %d" % (i, NUMBER_OF_IMAGES))

            fileName = df['rgb'][i]

            if datasetName == "VEDAI":
                img_rgb = cv2.imread(dirPath + fileName[:-6] + "co.png")
                img_ir = cv2.imread(dirPath + fileName[:-6] + "ir.png")

            else:
                sys.exit("ERROR: Unknown dataset")

            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            img_ir = cv2.cvtColor(img_ir, cv2.COLOR_RGB2GRAY)

            img_rgb = img_rgb - np.mean(img_rgb)
            img_ir = img_ir - np.mean(img_ir)

            H, W = np.shape(img_rgb)
            grid_x = W // self.patchSize
            grid_y = H // self.patchSize
            grid_size = grid_x * grid_y

            img_rgb_input = np.empty((grid_size, self.patchSize, self.patchSize, 1))
            img_ir_input = np.empty((grid_size, self.patchSize, self.patchSize, 1))

            # Split to patches
            for g in range(grid_size):
                x = (g % grid_x) * self.patchSize
                y = (g // grid_x) * self.patchSize
                img_rgb_input[g, :, :, 0] = img_rgb[y:y + self.patchSize, x:x + self.patchSize]
                img_ir_input[g, :, :, 0] = img_ir[y:y + self.patchSize, x:x + self.patchSize]

            # Feed network
            _, _, _, _, _, _, siamese_features_x, siamese_features_y, asymmetric_features_x, asymmetric_features_y = self.hybridModel([img_rgb_input, img_ir_input], training=False)

            # Bring back to image size

            rgb_features_map_siamese = np.empty((H, W))
            ir_features_map_siamese = np.empty((H, W))

            rgb_features_map_asymmetric = np.empty((H, W))
            ir_features_map_asymmetric = np.empty((H, W))

            for g in range(grid_size):
                x = (g % grid_x) * self.patchSize
                y = (g // grid_x) * self.patchSize

                rgb_features_map_siamese[y:y + self.patchSize, x:x + self.patchSize] = siamese_features_x[g, :, :]
                ir_features_map_siamese[y:y + self.patchSize, x:x + self.patchSize] = siamese_features_y[g, :, :]

                rgb_features_map_asymmetric[y:y + self.patchSize, x:x + self.patchSize] = asymmetric_features_x[g, :, :]
                ir_features_map_asymmetric[y:y + self.patchSize, x:x + self.patchSize] = asymmetric_features_y[g, :, :]

            locs = np.unravel_index(np.argsort(-1 * rgb_features_map_siamese, axis=None),
                                    rgb_features_map_siamese.shape)
            rgb_siamese_features_locs = np.empty((Ns, 2))
            rgb_siamese_features_locs[:, 1] = locs[0][:Ns]
            rgb_siamese_features_locs[:, 0] = locs[1][:Ns]

            locs = np.unravel_index(np.argsort(-1 * ir_features_map_siamese, axis=None),
                                    ir_features_map_siamese.shape)
            ir_siamese_features_locs = np.empty((Ns, 2))
            ir_siamese_features_locs[:, 1] = locs[0][:Ns]
            ir_siamese_features_locs[:, 0] = locs[1][:Ns]

            locs = np.unravel_index(np.argsort(-1 * rgb_features_map_asymmetric, axis=None),
                                    rgb_features_map_asymmetric.shape)
            rgb_asymmetric_features_locs = np.empty((Ns, 2))
            rgb_asymmetric_features_locs[:, 1] = locs[0][:Ns]
            rgb_asymmetric_features_locs[:, 0] = locs[1][:Ns]

            locs = np.unravel_index(np.argsort(-1 * ir_features_map_asymmetric, axis=None),
                                    ir_features_map_asymmetric.shape)
            ir_asymmetric_features_locs = np.empty((Ns, 2))
            ir_asymmetric_features_locs[:, 1] = locs[0][:Ns]
            ir_asymmetric_features_locs[:, 0] = locs[1][:Ns]

            for r in range(R):

                siamese_rgb_ir = 0
                siamese_ir_rgb = 0
                asymmetric_rgb_ir = 0
                asymmetric_ir_rgb = 0

                for p1 in range(Ns):

                    detect_siamese_rgb_ir = 0
                    detect_siamese_ir_rgb = 0
                    detect_asymmetric_rgb_ir = 0
                    detect_asymmetric_ir_rgb = 0

                    for p2 in range(Ns):

                        if detect_siamese_rgb_ir == 0:
                            detect_siamese_rgb_ir = int(np.sqrt(
                                (rgb_siamese_features_locs[p1, 1] - ir_siamese_features_locs[p2, 1]) ** 2 + (
                                        rgb_siamese_features_locs[p1, 0] - ir_siamese_features_locs[p2, 0]) ** 2) <= r)
                            siamese_rgb_ir += detect_siamese_rgb_ir

                        if detect_siamese_ir_rgb == 0:
                            detect_siamese_ir_rgb = int(np.sqrt(
                                (ir_siamese_features_locs[p1, 1] - rgb_siamese_features_locs[p2, 1]) ** 2 + (
                                        ir_siamese_features_locs[p1, 0] - rgb_siamese_features_locs[p2, 0]) ** 2) <= r)
                            siamese_ir_rgb += detect_siamese_ir_rgb

                        if detect_asymmetric_rgb_ir == 0:
                            detect_asymmetric_rgb_ir = int(np.sqrt(
                                (rgb_asymmetric_features_locs[p1, 1] - ir_asymmetric_features_locs[p2, 1]) ** 2 + (
                                        rgb_asymmetric_features_locs[p1, 0] - ir_asymmetric_features_locs[
                                    p2, 0]) ** 2) <= r)
                            asymmetric_rgb_ir += detect_asymmetric_rgb_ir

                        if detect_asymmetric_ir_rgb == 0:
                            detect_asymmetric_ir_rgb = int(np.sqrt(
                                (ir_asymmetric_features_locs[p1, 1] - rgb_asymmetric_features_locs[p2, 1]) ** 2 + (
                                        ir_asymmetric_features_locs[p1, 0] - rgb_asymmetric_features_locs[
                                    p2, 0]) ** 2) <= r)
                            asymmetric_ir_rgb += detect_asymmetric_ir_rgb

                siamese_cum_det_prob[r, 0] += (siamese_rgb_ir + siamese_ir_rgb) / 2 / Ns
                asymmetric_cum_det_prob[r, 0] += (asymmetric_rgb_ir + asymmetric_ir_rgb) / 2 / Ns

        siamese_cum_det_prob /= NUMBER_OF_IMAGES
        asymmetric_cum_det_prob /= NUMBER_OF_IMAGES

        plt.figure()
        ax = plt.subplot(111)
        ax.plot(np.arange(1, R + 1), siamese_cum_det_prob, label='Hybrid-Siamese')
        ax.plot(np.arange(1, R + 1), asymmetric_cum_det_prob, label='Hybrid-Asymmetric')
        title_str = datasetName + " dataset detection results"
        plt.title(title_str)
        plt.grid()
        plt.xlabel("Radius [pixels]")
        plt.ylabel("Cumulative Detection Probability")
        ax.legend()
        plt.xlim([1, R])
        plt.ylim([0, 1])
        plt.xticks(np.arange(1, R + 1), np.arange(1, R + 1))
        plt.show()

    def descriptors_matching(self, datasetName, dirPath, csvTest, Ns=500, good=0.3):

        df = pandas.read_csv(csvTest)
        IMAGE = np.random.randint(0, len(df))

        fileName = df['rgb'][IMAGE]

        if datasetName == "VEDAI":
            imageRgbPath = dirPath + fileName[:-6] + "co.png"
            imageIrPath = dirPath + fileName[:-6] + "ir.png"

        else:
            sys.exit("ERROR: Unknown dataset")

        print("Loading: ", imageRgbPath)
        img_rgb = cv2.imread(imageRgbPath)
        print("[image RGB]: ", np.shape(img_rgb))

        print("Loading: ", imageIrPath)
        img_ir = cv2.imread(imageIrPath)
        print("[image IR]: ", np.shape(img_ir))

        if np.size(img_rgb) != np.size(img_ir):
            sys.exit("ERROR: Different images resolution")

        if np.size(img_rgb, 0) % self.patchSize != 0:
            img_rgb = img_rgb[0:np.size(img_rgb, 0)//self.patchSize*self.patchSize, :]
            img_ir = img_ir[0:np.size(img_rgb, 0) // self.patchSize * self.patchSize, :]
            print("[image RGB]: ", np.shape(img_rgb))
            print("[image IR]: ", np.shape(img_ir))

        if np.size(img_rgb, 1) % self.patchSize != 0:
            img_rgb = img_rgb[:, 0:np.size(img_rgb, 0)//self.patchSize*self.patchSize]
            img_ir = img_ir[:, 0:np.size(img_rgb, 0) // self.patchSize * self.patchSize]
            print("[image RGB]: ", np.shape(img_rgb))
            print("[image IR]: ", np.shape(img_ir))


        img_rgb_raw = img_rgb
        img_ir_raw = img_ir

        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_ir = cv2.cvtColor(img_ir, cv2.COLOR_RGB2GRAY)

        img_rgb_gray = img_rgb
        img_ir_gray = img_ir

        # Extract keypoints using GFTT
        mask = np.zeros(np.shape(img_rgb), np.uint8)
        mask[self.patchSize:np.size(img_rgb, 0) - self.patchSize - 1, self.patchSize:np.size(img_rgb, 1) - self.patchSize - 1] = 255
        rgb_corners = cv2.goodFeaturesToTrack(img_rgb, Ns, 0.01, 3, mask=mask)
        ir_corners = cv2.goodFeaturesToTrack(img_ir, Ns, 0.01, 3, mask=mask)

        if np.size(rgb_corners, 0) != Ns and np.size(ir_corners, 0) != Ns:
            sys.exit("Error: Not enough features detected")

        # Extract patches around keypoints
        img_rgb = img_rgb - np.mean(img_rgb)
        img_ir = img_ir - np.mean(img_ir)

        img_rgb_input = np.empty((np.size(rgb_corners, 0), self.patchSize, self.patchSize, 1))
        img_ir_input = np.empty((np.size(ir_corners, 0), self.patchSize, self.patchSize, 1))

        for c in range(np.size(rgb_corners, 0)):
            x = int(rgb_corners[c, 0, 0])
            y = int(rgb_corners[c, 0, 1])
            img_rgb_input[c, :, :, 0] = img_rgb[y:y + self.patchSize, x:x + self.patchSize]

        for c in range(np.size(ir_corners, 0)):
            x = int(ir_corners[c, 0, 0])
            y = int(ir_corners[c, 0, 1])
            img_ir_input[c, :, :, 0] = img_ir[y:y + self.patchSize, x:x + self.patchSize]

        # Feed network
        x_representation, y_representation, Ws_x, Ws_y, Wx_x, Wy_y, siamese_features_x, siamese_features_y, asymmetric_features_x, asymmetric_features_y = self.hybridModel([img_rgb_input, img_ir_input], training=False)

        # Brute Force Matcher solution, pick 2-norm distance minimum as a pair
        dis_list = []
        min_min_dis = float("inf")
        for current in range(Ns):

            min_dis = float("inf")
            min_idx = 0

            for candidate in range(Ns):

                dis = np.linalg.norm(x_representation[current, :] - y_representation[candidate, :])

                if dis < min_dis:
                    min_idx = candidate
                    min_dis = dis

            dis_list.append([current, min_idx, min_dis])

            if min_dis < min_min_dis:
                min_min_dis = min_dis

        dis_list = np.array(dis_list)
        sorted_idx = np.argsort(dis_list[:, 2])
        sorted_idx = sorted_idx[:int(good*len(sorted_idx))]
        matchesImage = np.hstack([img_rgb_raw, img_ir_raw])
        h, w = np.shape(img_rgb)

        for f in range(len(sorted_idx)):

            rgb_x = int(rgb_corners[int(dis_list[f, 0]), 0, 0])
            rgb_y = int(rgb_corners[int(dis_list[f, 0]), 0, 1])

            ir_x = int(ir_corners[int(dis_list[f, 1]), 0, 0]) + w
            ir_y = int(ir_corners[int(dis_list[f, 1]), 0, 1])

            R = np.random.randint(0, 255)
            G = np.random.randint(0, 255)
            B = np.random.randint(0, 255)

            cv2.line(matchesImage, (rgb_x, rgb_y), (ir_x, ir_y), (B, G, R), 1)

        matchesImage = cv2.cvtColor(matchesImage, cv2.COLOR_BGR2RGB)

        plt.figure()
        plt.imshow(matchesImage)
        plt.title("Siamese Matching")
        plt.show()

        # GFTT + ORB Feature matching
        gftt = cv2.GFTTDetector_create(Ns, 0.01, 3)
        rgb_kp = gftt.detect(img_rgb_gray, mask)
        ir_kp = gftt.detect(img_ir_gray, mask)

        orb = cv2.ORB_create()
        _, rgb_des = orb.compute(img_rgb_gray, rgb_kp)
        _, ir_des = orb.compute(img_ir_gray, ir_kp)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, False)

        # Match descriptors.
        matches = bf.match(rgb_des, ir_des)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        # Draw first matches.
        img_rgb_raw = cv2.cvtColor(img_rgb_raw, cv2.COLOR_BGR2RGB)
        orb_img = cv2.drawMatches(img_rgb_raw, rgb_kp, img_ir_raw, ir_kp, matches[:len(sorted_idx)], None)

        plt.figure()
        plt.imshow(orb_img)
        plt.title("ORB Matching")
        plt.show()


def main():
    """
    --------------------------------------------------------------------------------------
                                       Configuration
    --------------------------------------------------------------------------------------
    """
    LOAD_WEIGHTS = True  # True | False
    TRAIN = False  # True | False
    EVALUATE = 0  # 0 = Visualize matching | 1 = FPR95 | 2 = Cumulative Detection Probability
    dataset = 0  # 0 = VEDAI
    basePath = 'Datasets'  # Set the path for the dataset directory

    ''' Training parameters '''
    AUGMENTATION_TYPE = 1  # 0 - None | 1 - Same for both patches | 2 - different for each rgb-ir patch
    HARD_NEGATIVE_MINING = True
    EPOCHS = 40
    BATCH_SIZE = 64
    LEARNING_RATE = 0.01
    MOMENTUM = 0.01

    ''' Evaluate FPR95% Recall parameters '''
    EVALUATE_BATCH = 32

    ''' Evaluate Cumulative Detection Probability parameters '''
    Ns = 500

    """--------------------------------------------------------------------------------------"""

    dataDir = []
    DATASET = []
    csvTrainPath = []
    csvTestPath = []
    weightsPath = []

    if dataset == 0:
        print("Loading VEDAI files")
        DATASET = "VEDAI"
        dataDir = basePath + '/VEDAI/Vehicules512/'
        csvTrainPath = basePath + '/VEDAI/vedaiTrain.csv'
        csvTestPath = basePath + '/VEDAI/vedaiTest.csv'
        csvDetectionsTestPath = basePath + '/VEDAI/vedaiTestDetections.csv'
        weightsPath = 'hybridModel_Epoch40_vedai.h5'

    else:
        sys.exit("ERROR: Unknown dataset")

    model = MultimodalHybridCNN()

    if LOAD_WEIGHTS:
        model.load_Weights(weightsPath)

    if TRAIN:
        print("Training")
        model.train(dataDir, csvTrainPath, DATASET, epochs=EPOCHS, batchSize=BATCH_SIZE, lr=LEARNING_RATE, momentum=MOMENTUM, augmantationsType=AUGMENTATION_TYPE, HNM=HARD_NEGATIVE_MINING)

    elif EVALUATE == 1:
        print("Evaluate FPR95 Recall")
        model.evaluate_fpr(DATASET, dataDir, csvTestPath, batchSize=EVALUATE_BATCH)

    elif EVALUATE == 2:
        print("Evaluate Cumulative Detection Probability")
        model.evaluate_detections(DATASET, dataDir, csvDetectionsTestPath, Ns=Ns)

    else:
        while 1:
            model.descriptors_matching(DATASET, dataDir, csvDetectionsTestPath, Ns=100, good=0.3)

    return 0


if __name__ == '__main__':
    main()
