
import sys, os

import tensorflow as tf
from NumTForm import Transformer



class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

tform = Transformer(
            num_layers = 6,
            d_model = 128,
            num_heads = 8,
            dff = 512)

checkpoint_path = os.path.join('..', 'models', 'tform_train')
ckpt = tf.train.Checkpoint(transformer=tform, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# Recovers the latest checkpoing
# I could see how I might not want this
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)

EPOCHS = 20


loss_function = tf.keras.losses.MeanSquaredError(reduction='none')
train_loss = tf.keras.metrics.Mean(name='train_loss')

# This is probably not my signature, I will be using 
# floats that are already composed into batches
train_step_signature = [
    tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
    tf.TensorSpec(shape=(None), dtype=tf.float32),
        ]

@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    with tf.GradientTape() as tape:
        predictions, _ = tform(inp, training=True)
        loss = loss_funtion(tar, predictions)
    gradients = tape.gradient(loss, tform.trainable_variables)
    optimizer.apply_gradients(zip(gradients, tform.trainable_variables))

    train_loss.update_state(loss)


for epoch in range(EPOCHS):
    pass

    









