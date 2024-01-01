import tensorflow as tf
import os.path as op
import tensorflow.keras.backend as K
import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())


def siamese_backbone(input_shape):
    """
    It will return EfficientNetB2 model as backbone
    Parameters
    ----------
    input_shape : int
        image size for architecture

    Returns
    -------
    keras.engine.functional.Functional
        customized EfficientNetB2 model as backbone
    """
    efficient_model = tf.keras.applications.EfficientNetB2(include_top=False, input_shape=(input_shape, input_shape, 3))
    efficient_model.trainable = True

    global_avg_layer = efficient_model.get_layer('block7b_se_squeeze').output
    embedding_layer = tf.keras.layers.Dense(64, name='embedding_layer')(global_avg_layer)
    embedding_layer_regularized = tf.keras.layers.Lambda(lambda x: K.l2_normalize(x, axis=1),
                                                         name='embedding_layer_regularized')(embedding_layer)

    backbone_model = tf.keras.models.Model(efficient_model.input, outputs=embedding_layer_regularized)
    return backbone_model


def siamese_network(backbone_model, input_shape):
    """
    Create a triple model
    Parameters
    ----------
    backbone_model : keras.engine.functional.Functional
        customized EfficientNetB2 model as backbone
    input_shape : int
        image size for architecture
    Returns
    -------
    keras.engine.functional.Functional
        triplet architectre
    """
    anchor_input = tf.keras.layers.Input(shape=(input_shape, input_shape, 3), name='anchor_input')
    positive_input = tf.keras.layers.Input(shape=(input_shape, input_shape, 3), name='positive_input')
    negative_input = tf.keras.layers.Input(shape=(input_shape, input_shape, 3), name='negative_input')

    anchor_embedding = backbone_model(anchor_input)
    positive_embedding = backbone_model(positive_input)
    negative_embedding = backbone_model(negative_input)

    training_model = tf.keras.models.Model(inputs=[anchor_input, positive_input, negative_input],
                                           outputs=[anchor_embedding, positive_embedding, negative_embedding])

    # backbone_model=tf.keras.models.load_model('model/15back/')
    return training_model


class TripletLossModel(tf.keras.models.Model):
    """
    Parameters
    ---------------
    margin = margin for model

    Returns
    ---------
    loss calculation

    """

    def __init__(self, training_model, margin=0.4):
        super().__init__()
        self.margin = margin
        self.training_model = training_model
        self.loss_tracker = tf.keras.metrics.Mean('loss')

    def call(self, inputs):
        return self.training_model(inputs)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)
        gradients = tape.gradient(loss, self.training_model.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.training_model.trainable_weights))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        anchor_embed, positive_embed, negative_embed = self.training_model(data[0])

        anchor_positive_distance = tf.reduce_sum(tf.square(tf.subtract(anchor_embed, positive_embed)), -1)

        anchor_negative_distance = tf.reduce_sum(tf.square(tf.subtract(anchor_embed, negative_embed)), -1)

        triplet_loss = tf.maximum(anchor_positive_distance - anchor_negative_distance + self.margin, 0)
        neg_dis = tf.multiply(anchor_embed, negative_embed)
        d = anchor_embed.shape[1]
        gor_loss = tf.pow(tf.reduce_mean(neg_dis), 2) + tf.maximum(tf.reduce_mean(tf.pow(neg_dis, 2), -1), 0)

        alpha = 1.1
        total_loss = triplet_loss + alpha * gor_loss
        return total_loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]
