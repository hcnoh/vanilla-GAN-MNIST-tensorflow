import tensorflow as tf


def generator(z, feature_depth, hidden_sizes=[128, 256, 256]):
    h = z
    for i, hidden_size in enumerate(hidden_sizes):
        h = tf.layers.dense(
            inputs=h,
            units=hidden_size,
            activation=tf.nn.leaky_relu,
            name="dense_%i" % i
        )
    g = tf.layers.dense(
        inputs=h,
        units=feature_depth,
        activation=tf.nn.tanh,
        name="dense_output"
    )

    return g


def discriminator(x, hidden_sizes=[256, 256, 128]):
    h = x
    for i, hidden_size in enumerate(hidden_sizes):
        h = tf.layers.dense(
            inputs=h,
            units=hidden_size,
            activation=tf.nn.relu,
            name="dense_%i" % i
        )
    score_logits = tf.layers.dense(
        inputs=h,
        units=1,
        name="dense_output"
    )

    return score_logits


class VanillaGAN(object):
    def __init__(self, x, z, feature_depth):
        self.x = x
        self.z = z
        self.feature_depth = feature_depth

        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
            self.g = generator(self.z, self.feature_depth)
        
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            self.x_score_logits = discriminator(self.x)
            self.g_score_logits = discriminator(self.g)
    
    def get_minimax_losses(self):
        loss_g = \
            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(self.g_score_logits),
                logits=self.g_score_logits
            ))
        loss_d = \
            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(self.x_score_logits),
                logits=self.x_score_logits
            )) + \
            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(self.g_score_logits),
                logits=self.g_score_logits
            ))
        vars_g = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
        vars_d = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
        
        return loss_g, loss_d, vars_g, vars_d