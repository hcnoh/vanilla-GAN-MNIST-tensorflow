import os
import pickle

import numpy as np
import tensorflow as tf

import config as conf

from data.mnist import Mnist
from models.vanilla_gan import VanillaGAN


def feature_normalize(features):
    return (features - 0.5) / 0.5


def feature_denormalize(features):
    return (features + 1) / 2


def main():
    model_ckpt_name = "%s-model.ckpt" % conf.MODEL_NAME
    model_spec_name = "%s-model-spec.json" % conf.MODEL_NAME
    model_rslt_name = "%s-results.pickle" % conf.MODEL_NAME

    model_save_path = os.path.join(conf.MODEL_SAVE_DIR, conf.MODEL_NAME)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    model_ckpt_path = os.path.join(model_save_path, model_ckpt_name)
    model_spec_path = os.path.join(model_save_path, model_spec_name)
    model_rslt_path = os.path.join(model_save_path, model_rslt_name)

    loader = Mnist()

    features = np.vstack([loader.train_features, loader.test_features])
    features = feature_normalize(features)

    num_sets = loader.num_train_sets + loader.num_test_sets
    
    feature_depth = loader.feature_depth
    feature_shape = loader.feature_shape
    
    latent_depth = conf.LATENT_DEPTH

    batch_size = conf.BATCH_SIZE
    num_epochs = conf.NUM_EPOCHS
    
    x = tf.placeholder(dtype=tf.float32, shape=[None, feature_depth])
    z = tf.placeholder(dtype=tf.float32, shape=[None, latent_depth])

    gan = VanillaGAN(x, z, feature_depth)

    loss_g, loss_d, vars_g, vars_d = gan.get_minimax_losses()
    opt_g = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(loss=loss_g, var_list=vars_g)
    opt_d = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(loss=loss_d, var_list=vars_d)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(max_to_keep=1000)

    steps_per_epoch = num_sets // batch_size
    train_steps = steps_per_epoch * num_epochs

    train_losses_g = []
    train_losses_d = []
    train_losses_epoch_g = []
    train_losses_epoch_d = []
    gs = []
    for i in range(1, train_steps+1):
        epoch = i // steps_per_epoch

        idxes = np.random.choice(num_sets, batch_size, replace=False)
        features_i = features[idxes]
        z_i = np.random.randn(batch_size, latent_depth)

        loss_d_i, _ = sess.run([loss_d, opt_d], feed_dict={x: features_i, z: z_i})
        loss_g_i, _ = sess.run([loss_g, opt_g], feed_dict={x: features_i, z: z_i})

        train_losses_g.append(loss_g_i)
        train_losses_d.append(loss_d_i)

        if i % steps_per_epoch == 0:
            view_range = 10
            g = sess.run(gan.g[:view_range], feed_dict={z: z_i})
            g = feature_denormalize(g)

            train_loss_epoch_g = np.mean(train_losses_g[-steps_per_epoch:])
            train_loss_epoch_d = np.mean(train_losses_d[-steps_per_epoch:])

            print("Epoch: %i,  Training G Loss: %f,  Training D Loss: %f" % \
                (epoch, train_loss_epoch_g, train_loss_epoch_d)
            )

            train_losses_epoch_g.append(train_loss_epoch_g)
            train_losses_epoch_d.append(train_loss_epoch_d)

            gs.append(g)

            saver.save(sess, model_ckpt_path, global_step=epoch)

            with open(model_rslt_path, "wb") as f:
                pickle.dump((train_losses_epoch_g, train_losses_epoch_d, gs), f)


if __name__ == "__main__":
    main()