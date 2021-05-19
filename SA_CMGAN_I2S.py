from __future__ import division
#coding=utf-8

import os
import time
import tensorflow as tf
import numpy as np
from glob import glob

from ops import *
from utils import *

class I2S(object):
    def __init__(self, sess, epoch, batch_size, z_dim, dataset_name, checkpoint_dir, result_dir, log_dir, test_dir):
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.test_dir = test_dir
        self.epoch = epoch
        self.batch_size = batch_size
        self.model_name = "I2S"     # name for checkpoint
        self.n_critic = 2
        self.pretrian_step = 4

        if dataset_name == 'Sub-URMP':
            # parameters
            self.input_height = 108
            self.input_width = 130
            self.output_height = 64
            self.output_width = 64

            self.z_dim = z_dim         # dimension of noise-vector
            self.y_dim = 13         # dimension of code-vector (label)
            self.c_dim = 3

            # train
            self.learning_rate = 0.0001
            self.beta1 = 0.5
            self.beta2 = 0.999

            # test
            self.sample_num = 64  # number of generated images to be saved

            #train iter
            self.train_iter = 0

            # load mnist
            self.data_X, self.data_S, self.data_MIS_X, self.data_MIS_S, self.data_y = load_Sub(self.dataset_name, self.y_dim)

            # get number of batches for a single epoch
            self.num_batches = 2000

            self.dataset_num = len(self.data_X)

        else:
            raise NotImplementedError

    def encoder_Img(self, x, is_training=True, reuse=False):
        with tf.variable_scope("encoder_Img", reuse=reuse):
            net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='enI_conv1'))
            net = self.attention_Img(net, scope="attention", reuse=reuse)
            net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='enI_conv2'), is_training=is_training, scope='enI_bn2'))
            net = lrelu(bn(conv2d(net, 256, 4, 4, 2, 2, name='enI_conv3'), is_training=is_training, scope='enI_bn3'))
            net = lrelu(bn(conv2d(net, 256, 4, 4, 1, 1, name='enI_conv4'), is_training=is_training, scope='enI_bn4'))
            net = lrelu(bn(conv2d(net, 512, 4, 4, 2, 2, name='enI_conv5'), is_training=is_training, scope='enI_bn5'))
            net = tf.reshape(net, [self.batch_size, -1])
            net = lrelu(bn(linear(net, 1024, scope='enI_fc6'), is_training=is_training, scope='enI_bn6'))
            out_classifier = lrelu(bn(linear(net, 128, scope='enI_fc7'), is_training=is_training, scope='enI_bn7'))
            out = lrelu(bn(linear(out_classifier, 64, scope='enI_fc8'), is_training=is_training, scope='enI_bn8'))
        return out, out_classifier

    def classifier_Img(self, x, reuse=False):
        with tf.variable_scope("classifier_Img", reuse=reuse):
            net = linear(x, 13, scope='cI_fc1')
            prediction = tf.nn.softmax(net)
        return prediction

    def encoder_Sud(self, x, is_training=True, reuse=False):
        with tf.variable_scope("encoder_Sud", reuse=reuse):
            net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='enS_conv1'))
            net = self.attention_Sud(net, scope="attention", reuse=reuse)
            net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='enS_conv2'), is_training=is_training, scope='enS_bn2'))
            net = lrelu(bn(conv2d(net, 256, 4, 4, 2, 2, name='enS_conv3'), is_training=is_training, scope='enS_bn3'))
            net = lrelu(bn(conv2d(net, 256, 4, 4, 1, 1, name='enS_conv4'), is_training=is_training, scope='enS_bn4'))
            net = lrelu(bn(conv2d(net, 512, 4, 4, 2, 2, name='enS_conv5'), is_training=is_training, scope='enS_bn5'))
            net = tf.reshape(net, [self.batch_size, -1])
            net = lrelu(bn(linear(net, 1024, scope='enS_fc6'), is_training=is_training, scope='enS_bn6'))
            out_classifier = lrelu(bn(linear(net, 128, scope='enS_fc7'), is_training=is_training, scope='enS_bn7'))
            out = lrelu(bn(linear(out_classifier, 64, scope='enS_fc8'), is_training=is_training, scope='enS_bn8'))
        return out, out_classifier

    def classifier_Sud(self, x, reuse=False):
        with tf.variable_scope("classifier_Sud", reuse=reuse):
            net = linear(x, 13, scope='cS_fc1')
            prediction = tf.nn.softmax(net)
        return prediction

    def generator(self, z, x, is_training=True, reuse=False):
        with tf.variable_scope("generator", reuse=reuse):
            z = concat([x, z], 1)
            net = tf.nn.relu(bn(linear(z, 512 * 4 * 4, scope='g_fc1'), is_training=is_training, scope='g_bn1'))
            net = tf.reshape(net, [self.batch_size, 4, 4, 512])
            net = tf.nn.relu(bn(deconv2d(net, [self.batch_size, 8, 8, 256], 4, 4, 2, 2, name='g_dc2'),
                                is_training=is_training, scope='g_bn2'))
            net = tf.nn.relu(bn(deconv2d(net, [self.batch_size, 8, 8, 256], 4, 4, 1, 1, name='g_dc3'),
                                is_training=is_training, scope='g_bn3'))
            net = tf.nn.sigmoid(deconv2d(net, [self.batch_size, 16, 16, 128], 4, 4, 2, 2, name='g_dc4'))
            net = tf.nn.sigmoid(deconv2d(net, [self.batch_size, 32, 32, 64], 4, 4, 2, 2, name='g_dc5'))
            net = self.attention_Sud(net, scope="attention", reuse=reuse)
            out = tf.nn.tanh(deconv2d(net, [self.batch_size, 64, 64, 3], 4, 4, 2, 2, name='g_dc6'))
        return out

    def discriminator(self, x_sound, x_img, is_training=True, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            for i in range(4):
                x_sound = concat([x_sound, x_sound], 1)
            x_sound = tf.reshape(x_sound, [self.batch_size, 4, 4, 64])
            net = lrelu(conv2d(x_img, 64, 4, 4, 2, 2, name='d_conv1', sn=True))
            net = self.attention_Sud(net, scope="attention", reuse=reuse, sn=True)
            net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='d_conv2', sn=True), is_training=is_training, scope='d_bn2'))
            net = lrelu(bn(conv2d(net, 256, 4, 4, 2, 2, name='d_conv3', sn=True), is_training=is_training, scope='d_bn3'))
            net = lrelu(bn(conv2d(net, 256, 4, 4, 1, 1, name='d_conv4', sn=True), is_training=is_training, scope='d_bn4'))
            net = lrelu(bn(conv2d(net, 512, 4, 4, 2, 2, name='d_conv5', sn=True), is_training=is_training, scope='d_bn5'))
            net = conv_cond_concat(net, x_sound)
            net = tf.reshape(net, [self.batch_size, -1])
            #net = MinibatchLayer(32, 32, net, 'd_fc6')
            net = lrelu(bn(linear(net, 1024, scope='d_fc7', sn=True), is_training=is_training, scope='d_bn7'))
            out_logit = linear(net, 1, scope='d_fc8', sn=True)
            out = tf.nn.sigmoid(out_logit)
        return out, out_logit

    def attention_Img(self, x, is_training=True, scope='attention_Img', reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            f = tf.nn.relu(bn(conv2d(x, 8, 4, 4, 1, 1, name='fI_conv'), is_training=is_training, scope='fI_bn'))
            g = tf.nn.relu(bn(conv2d(x, 8, 4, 4, 1, 1, name='gI_conv'), is_training=is_training, scope='gI_bn'))
            h = tf.nn.relu(bn(conv2d(x, 64, 4, 4, 1, 1, name='hI_conv'), is_training=is_training, scope='hI_bn'))

            s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

            beta = tf.nn.softmax(s, axis=-1)  # attention map

            o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
            gamma = tf.get_variable("gammaI", [1], initializer=tf.constant_initializer(0.0))

            o = tf.reshape(o, shape=x.shape)  # [bs, h, w, C]
            x = gamma * o + x

        return x

    def attention_Sud(self, x, is_training=True, scope='attention_Sud', reuse=False, sn=False):
        with tf.variable_scope(scope, reuse=reuse):
            f = tf.nn.relu(bn(conv2d(x, 8, 4, 4, 1, 1, name='fS_conv', sn=sn), is_training=is_training, scope='fS_bn'))
            g = tf.nn.relu(bn(conv2d(x, 8, 4, 4, 1, 1, name='gS_conv', sn=sn), is_training=is_training, scope='gS_bn'))
            h = tf.nn.relu(bn(conv2d(x, 64, 4, 4, 1, 1, name='hS_conv', sn=sn), is_training=is_training, scope='hS_bn'))

            s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

            beta = tf.nn.softmax(s, axis=-1)  # attention map

            o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
            gamma = tf.get_variable("gammaS", [1], initializer=tf.constant_initializer(0.0))

            o = tf.reshape(o, shape=x.shape)  # [bs, h, w, C]
            x = gamma * o + x

        return x

    def build_model(self):
        # some parameters
        image_dims = [self.output_height, self.output_width, self.c_dim]
        bs = self.batch_size

        """ Graph Input """
        # images
        self.inputs_img = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')

        self.inputs_sound = tf.placeholder(tf.float32, [bs] + image_dims, name='real_sounds')

        self.inputs_sound_mis = tf.placeholder(tf.float32, [bs] + image_dims, name='mis_sounds')

        # labels
        self.y = tf.placeholder(tf.float32, [bs, self.y_dim], name='y')

        # noises
        self.z = tf.placeholder(tf.float32, [bs, self.z_dim], name='z')

        """ Loss Function """

        En_Img, Cla_Img = self.encoder_Img(self.inputs_img, is_training=True, reuse=False)

        En_Sud, Cla_Sud = self.encoder_Sud(self.inputs_sound, is_training=True, reuse=False)

        G_sound = self.generator(self.z, En_Img, is_training=True, reuse=False)

        D_real, D_real_logits = self.discriminator(En_Img, self.inputs_sound, is_training=True, reuse=False)

        D_fake, D_fake_logits = self.discriminator(En_Img, G_sound, is_training=True, reuse=True)

        D_mis, D_mis_logits = self.discriminator(En_Img, self.inputs_sound_mis, is_training=True, reuse=True)

        prediction_Img = self.classifier_Img(Cla_Img, reuse=False)

        prediction_Sud = self.classifier_Sud(Cla_Sud, reuse=False)

        # get loss for discriminator
        self.d_loss = discriminator_loss('hinge', real=D_real_logits, fake=D_fake_logits, fake_mis=D_mis_logits)

        # get loss for generator
        self.g_loss = generator_loss('hinge', fake=D_fake_logits)

        # Loss Classifier Image
        self.classifier_loss_Img = classifier_loss(predition=prediction_Img, label=self.y)

        # Loss Classifier Sound
        self.classifier_loss_Sud = classifier_loss(predition=prediction_Sud, label=self.y)

        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        g_vars = [var for var in t_vars if 'generator' in var.name]
        cI_vars = [var for var in t_vars if ('encoder_Img' in var.name) or ('classifier_Img' in var.name)]
        cS_vars = [var for var in t_vars if ('encoder_Sud' in var.name) or ('classifier_Sud' in var.name)]

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1, beta2=self.beta2) \
                .minimize(self.d_loss, var_list=d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate * 5, beta1=self.beta1, beta2=self.beta2) \
                .minimize(self.g_loss, var_list=g_vars)
            self.cI_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1, beta2=self.beta2) \
                .minimize(self.classifier_loss_Img, var_list=cI_vars)
            self.cS_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1, beta2=self.beta2) \
                .minimize(self.classifier_loss_Sud, var_list=cS_vars)

        """" Testing """
        self.img, self.cla_img = self.encoder_Img(self.inputs_img, is_training=False, reuse=True)
        self.prediction_Img = self.classifier_Img(self.cla_img, reuse=True)
        self.sound, self.cla_sud = self.encoder_Sud(self.inputs_sound, is_training=False, reuse=True)
        self.prediction_Sud = self.classifier_Sud(self.cla_sud, reuse=True)
        self.fake_sounds = self.generator(self.z, self.img, is_training=False, reuse=True)

        """ Summary """
        self.d_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.g_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.cI_sum = tf.summary.scalar("cI_loss", self.classifier_loss_Img)
        self.cS_sum = tf.summary.scalar("cS_loss", self.classifier_loss_Sud)

    def train(self):

        # initialize all variables
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        self.sample_z = np.random.normal(0, 1, size=(self.batch_size, self.z_dim))

        # saver to save model
        self.saver = tf.train.Saver()

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = int(checkpoint_counter / self.num_batches) - self.pretrian_step
            start_batch_id = checkpoint_counter - (start_epoch+self.pretrian_step) * self.num_batches
            counter = checkpoint_counter - self.pretrian_step*self.num_batches
            print(" [*] Load SUCCESS")
            if start_epoch == self.epoch:
                print('testing............')
                self.visualize_paper(start_epoch)
                self.visualize_results(start_epoch, 2000)

        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        if start_epoch != self.epoch:
            # summary writer
            self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

        # loop for epoch
        start_time = time.time()

        if not could_load:
            print("pre_training Classifier")
            for epoch in range(0, self.pretrian_step):
                for idc in range(start_batch_id, self.num_batches):
                    random_index = np.random.choice(self.dataset_num, size=self.batch_size, replace=False)
                    labels = np.array(self.data_y)[random_index]
                    batch_images = get_pix_image(self.data_X, random_index)
                    batch_sounds = get_sound(self.data_S, random_index)
                    train_feed_dict = {
                        self.y: labels,
                        self.inputs_img: batch_images,
                        self.inputs_sound:batch_sounds
                    }

                    # update Classifer network
                    _, summary_str, c_loss_Img = self.sess.run([self.cI_optim, self.cI_sum, self.classifier_loss_Img],
                                                               feed_dict=train_feed_dict)
                    self.writer.add_summary(summary_str, counter)

                    _, summary_str, c_loss_Sud = self.sess.run([self.cS_optim, self.cS_sum, self.classifier_loss_Sud],
                                                               feed_dict=train_feed_dict)
                    self.writer.add_summary(summary_str, counter)

                    # display training status
                    counter += 1

                    if np.mod(counter, 30) == 0:
                        print("Epoch: [pre_training-%2d] [%4d/%4d] time: %4.4f, cI_loss: %.8f, cS_loss: %.8f"\
                              % (epoch, idc, self.num_batches, time.time() - start_time, c_loss_Img, c_loss_Sud))

            start_batch_id = 0
            # save model for final step
            self.save(self.checkpoint_dir, counter)
            counter = counter - self.pretrian_step*self.num_batches

        past_d_loss = -1.
        idx = 0
        for epoch in range(start_epoch, self.epoch):
            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                batch_z = np.random.normal(0, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                random_index = np.random.choice(self.dataset_num, size=self.batch_size, replace=False)
                batch_images = get_pix_image(self.data_X, random_index)
                batch_sounds = get_sound(self.data_S, random_index)
                batch_mis_sounds = get_sound(self.data_MIS_S, random_index)
                train_feed_dict_d = {
                    self.z: batch_z,
                    self.inputs_img: batch_images,
                    self.inputs_sound: batch_sounds,
                    self.inputs_sound_mis: batch_mis_sounds
                }
                train_feed_dict_g = {
                    self.z: batch_z,
                    self.inputs_img: batch_images
                }

                # update D network
                d_loss = None
                if (counter - 1) % self.n_critic == 0:
                    _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss],
                                                           feed_dict=train_feed_dict_d)
                    self.writer.add_summary(summary_str, counter)
                    past_d_loss = d_loss

                # update G network
                _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss],
                                                       feed_dict=train_feed_dict_g)
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                if d_loss is None:
                    d_loss = past_d_loss

                if np.mod(idx+1, 5) == 0:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"\
                          % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss))

                # save training results for every 300 steps
                if np.mod(idx+1, 300) == 0:
                    self.visualize_results(epoch, idx)

                if np.mod(idx+1, 1000) == 0:
                    start_batch_id = 0
                    samples = self.sess.run(self.fake_sounds,
                                            feed_dict=train_feed_dict_g)
                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                './' + check_folder(
                                    self.result_dir + '/' + self.model_dir) + '/' + self.model_name +
                                '_train_{:02d}_{:04d}.png'.format(epoch, idx))

            # save model
            self.save(self.checkpoint_dir, counter + self.pretrian_step*self.num_batches)

            # show temporal results
            self.visualize_results_test(epoch)
        # save model for final step
        self.save(self.checkpoint_dir, counter + self.pretrian_step*self.num_batches)

    def visualize_results(self, epoch, idx):
        z_sample = np.random.normal(0, 1, [self.batch_size, self.z_dim]).astype(np.float32)
        random_index = np.random.choice(self.dataset_num, size=self.batch_size, replace=False)
        batch_labels = np.array(self.data_y)[random_index]
        batch_images = get_pix_image(self.data_X, random_index)
        batch_sounds = get_sound(self.data_S, random_index)

        len_discrete_code_ply = 10

        samples = self.sess.run(self.fake_sounds, feed_dict={self.z: z_sample, self.inputs_img: batch_images})
        prediction_Img = self.sess.run(self.prediction_Img, feed_dict={self.inputs_img:batch_images})
        correct_prediction_Img = np.equal(np.argmax(prediction_Img, 1), np.argmax(batch_labels, 1))
        accuracy_Img = tf.reduce_mean(tf.cast(correct_prediction_Img, tf.float32))
        result_Img = self.sess.run(accuracy_Img, feed_dict={self.inputs_img:batch_images, self.y:batch_labels})
        print(epoch, " accuracy_Img:",  result_Img)

        prediction_Sud = self.sess.run(self.prediction_Sud, feed_dict={self.inputs_sound: samples})
        correct_prediction_Sud = np.equal(np.argmax(prediction_Sud, 1), np.argmax(batch_labels, 1))
        accuracy_Sud = tf.reduce_mean(tf.cast(correct_prediction_Sud, tf.float32))
        result_Sud = self.sess.run(accuracy_Sud, feed_dict={self.inputs_sound:samples, self.y:batch_labels})
        print(epoch, " accuracy_Sound:", result_Sud)

        f = open(check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + "accuracy_train.txt",
                 'a')
        f.write("Epoch:" + str(epoch) + "-" + str(idx) + " accuracy_Img:" + str(result_Img) + "\n")
        f.write("Epoch:" + str(epoch) + "-" + str(idx) + " accuracy_Sud:" + str(result_Sud) + "\n")
        f.close()

        np.random.seed()
        si = np.random.choice(self.batch_size, len_discrete_code_ply)

        samples = samples[si, :, :, :]
        batch_images = batch_images[si, :, :, :]
        batch_sounds = batch_sounds[si, :, :, :]

        all_samples = np.concatenate((batch_images, batch_sounds), axis=0)
        all_samples = np.concatenate((all_samples, samples), axis=0)

        """ save merged images to check style-consistency """

        canvas = np.zeros_like(all_samples)
        for s in range(3):
            for c in range(len_discrete_code_ply):
                canvas[:, :, :, :] = all_samples[:, :, :, :]
                # canvas[s * self.len_discrete_code + c, :, :, :] = all_samples[c * n_styles + s, :, :, :]

        save_images(canvas, [3, len_discrete_code_ply],
                    check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name +
                    '_{:02d}_{:04d}'.format(epoch, idx) + '_test_all_classes_style_by_style.png')

    def visualize_results_test(self, epoch):
        test_x, test_sounds, test_img_mis, test_sounds_mis, test_y = load_Sub_test(self.dataset_name, self.y_dim)
        dataset_num = len(test_x)
        z_sample = np.random.normal(0, 1, [self.batch_size, self.z_dim]).astype(np.float32)
        random_index = np.random.choice(dataset_num, size=self.batch_size, replace=False)
        batch_labels = np.array(test_y)[random_index]
        batch_images = get_pix_image(test_x, random_index)
        batch_sounds = get_sound(test_sounds, random_index)

        len_discrete_code_ply = 10

        samples = self.sess.run(self.fake_sounds, feed_dict={self.z: z_sample, self.inputs_img: batch_images})
        prediction_Img = self.sess.run(self.prediction_Img, feed_dict={self.inputs_img: batch_images})
        correct_prediction_Img = np.equal(np.argmax(prediction_Img, 1), np.argmax(batch_labels, 1))
        accuracy_Img = tf.reduce_mean(tf.cast(correct_prediction_Img, tf.float32))
        result_Img = self.sess.run(accuracy_Img,
                                   feed_dict={self.z: z_sample, self.inputs_img: batch_images, self.y: batch_labels})
        print(epoch, " test_accuracy_Img:", result_Img)

        prediction_Sud = self.sess.run(self.prediction_Sud, feed_dict={self.inputs_sound: samples})
        correct_prediction_Sud = np.equal(np.argmax(prediction_Sud, 1), np.argmax(batch_labels, 1))
        accuracy_Sud = tf.reduce_mean(tf.cast(correct_prediction_Sud, tf.float32))
        result_Sud = self.sess.run(accuracy_Sud,
                                   feed_dict={self.inputs_sound: samples, self.y: batch_labels})
        print(epoch, " test_accuracy_Sound:", result_Sud)

        f = open(check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + "accuracy_test.txt",
                 'a')
        f.write("Epoch:" + str(epoch) + " test_accuracy_Img:" + str(result_Img) + "\n")
        f.write("Epoch:" + str(epoch) + " test_accuracy_Sud:" + str(result_Sud) + "\n")
        f.close()

        np.random.seed()
        si = np.random.choice(self.batch_size, len_discrete_code_ply)

        samples = samples[si, :, :, :]
        batch_images = batch_images[si, :, :, :]
        batch_sounds = batch_sounds[si, :, :, :]

        all_samples = np.concatenate((batch_images, batch_sounds), axis=0)
        all_samples = np.concatenate((all_samples, samples), axis=0)

        """ save merged images to check style-consistency """

        canvas = np.zeros_like(all_samples)
        for s in range(3):
            for c in range(len_discrete_code_ply):
                canvas[:, :, :, :] = all_samples[:, :, :, :]
                # canvas[s * self.len_discrete_code + c, :, :, :] = all_samples[c * n_styles + s, :, :, :]

        save_images(canvas, [3, len_discrete_code_ply],
                    check_folder(self.test_dir + '/' + self.model_dir) + '/' + self.model_name +
                    '_epoch%03d' % epoch + '_test_all_classes_style_by_style.png')

    def visualize_paper(self, epoch):
        z_sample = np.random.normal(0, 1, [self.batch_size, self.z_dim]).astype(np.float32)
        random_index = np.random.choice(self.dataset_num, size=self.batch_size, replace=False)
        batch_labels = np.array(self.data_y)[random_index]
        batch_images = get_pix_image(self.data_X, random_index)
        batch_sounds = get_sound(self.data_S, random_index)

        len_discrete_code_ply = 13

        samples = self.sess.run(self.fake_sounds, feed_dict={self.z: z_sample, self.inputs_img: batch_images})
        si = [40, 31, 5, 53, 19, 27, 13, 51, 7, 0, 20, 2, 36]

        samples = samples[si, :, :, :]
        batch_images = batch_images[si, :, :, :]
        batch_sounds = batch_sounds[si, :, :, :]

        all_samples = np.concatenate((batch_images, batch_sounds), axis=0)
        all_samples = np.concatenate((all_samples, samples), axis=0)

        """ save merged images to check style-consistency """

        canvas = np.zeros_like(all_samples)
        for s in range(3):
            for c in range(len_discrete_code_ply):
                canvas[:, :, :, :] = all_samples[:, :, :, :]
                # canvas[s * self.len_discrete_code + c, :, :, :] = all_samples[c * n_styles + s, :, :, :]

        save_images(canvas, [3, len_discrete_code_ply],
                    check_folder(self.test_dir + '/' + self.model_dir) + '/' + self.model_name + '_simple.png')

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.z_dim)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def train_check(self):
        import re
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir, self.model_name)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            start_epoch = (int)(counter / self.num_batches)
        if start_epoch == self.epoch:
            print(" [*] Training already finished! Begin to test your model")
