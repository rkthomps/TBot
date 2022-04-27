
import sys, os

import tensorflow as tf
import numpy as np
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


num_layers = 6
d_model = 128
num_heads = 8
dff = 512

tform = Transformer(
            num_layers = num_layers,
            d_model = d_model,
            num_heads = num_heads,
            dff = dff)


examples = 10
seq_len = 4 
d_model_prev= 4
in_size = examples * seq_len * d_model_prev
inp = tf.constant(np.arange(in_size).reshape(examples, seq_len, d_model_prev), dtype=tf.float32)
out = tf.constant(np.arange(examples)[:, None], dtype=tf.float32) 

learning_rate = CustomSchedule(d_model)
print(tform(inp))

tform.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, 
                                            beta_2=0.98, epsilon=1e-9),
        loss=tf.keras.losses.MeanSquaredError())
tform.fit(inp, out, batch_size=3, epochs=2)







