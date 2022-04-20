import tensorflow as tf
import numpy as np

MAX_TOKENS = 400
MAX_DIM = 500


def point_wise_feed_forward_network(d_model, dff):
    '''
    It seems that dff can be whatever I want it to be
    '''
    return tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model) # No activation on last layer?
        ])

def scaled_dot_product_attention(q, k, v):
    '''
        q: ... * seq_len * d_model 
        k: ... * seq_len * d_model 
        v: ... * seq_len * d_model 
    '''
    similarity = tf.matmul(q, k, transpose_b=True) # ... * seq_len * seq_len
    dk = tf.cast(tf.shape(k)[1], tf.float32)

    # Scaled Attention
    scaled_attention = similarity / tf.math.sqrt(dk)

    # Similarities are on the last axis, so we apply softmax on the last axis
    weights = tf.nn.softmax(scaled_attention, axis=-1)

    # Find a weighted sum of the values based on weights
    output = tf.matmul(weights, v) # ... * seq_len * depth

    return output, weights


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, None],
                            np.arange(d_model)[None, :],
                            d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_enc = angle_rads[None, ...]
    return tf.cast(pos_enc, dtype=tf.float32)



class MultiHeadAttention(tf.keras.layers.Layer):
    '''
    In this function we define the linear layers
    that compose W_q, W_k, W_v
    '''
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        assert d_model % self.num_heads == 0
        self.d_model = d_model
        self.depth = d_model // self.num_heads
        
        self.wqs = [tf.keras.layers.Dense(self.depth)] * self.num_heads
        self.wks = [tf.keras.layers.Dense(self.depth)] * self.num_heads
        self.wvs = [tf.keras.layers.Dense(self.depth)] * self.num_heads
        self.comb = tf.keras.layers.Dense(d_model)

    '''
    Call method has
        - q: tensor of queries
        - k: tensor of keys
        - v: tensor of values
        - m: mask
    '''
    def call(self, q, k, v):
        batch_size = tf.shape(q)[0]

        q = tf.concat([ql(q)[:, None, :, :] for ql in self.wqs], axis=1)
        k = tf.concat([kl(k)[:, None, :, :] for kl in self.wks], axis=1)
        v = tf.concat([vl(v)[:, None, :, :] for vl in self.wvs], axis=1)

        scaled_attention, attention_weights = scaled_dot_product_attention(
                q, k, v)

        # When you reshape, you want the dimensions you are merging as the
        # last dimensions in the tensor
        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3]) 
        concat_attention = tf.reshape(scaled_attention,
                                        (batch_size, -1, self.d_model))
        output = self.comb(concat_attention)
        return output, attention_weights



class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        # Layer normalization done on the last axis. 
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Dropout (0.1 in the Google paper)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training):
        attn_output, attn_weights = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training) # need to be separate?
        out2 = self.layernorm2(out1 + ffn_output)
        return out2, attn_weights


class Encoder(tf.keras.layers.Layer):
    '''
    Takes stock market data from the past however many weeks 
    '''
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        #self.pos_encoding = positional_encoding(MAX_TOKENS, MAX_DIM)
        self.pos_encoding = point_wise_feed_forward_network(d_model, dff)

        self.enc_layers = [
                EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, rate=rate)
                for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training):
        seq_len = tf.shape(x)[1]
        dim_len = tf.shape(x)[2]
        attn_weights = {}

        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32)) # Don't see why necessary
        x_pos = self.pos_encoding(x)

        x_pos = self.dropout(x_pos, training=training) # Dropout after positional encoding

        for i in range(self.num_layers):
            x_pos, block_attn_weights = self.enc_layers[i](x_pos, training)
            attn_weights[f'encoder_layer{i+1}'] = block_attn_weights

        return x_pos, attn_weights


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
        super().__init__() # Don't need to specify transformer here
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model, 
                                num_heads=num_heads, dff=dff, rate=rate)

        self.final_layer = tf.keras.layers.Dense(1)

    def call(self, inp, training):
        enc_output, attention_weights = self.encoder(inp, training)
        final_output = self.final_layer(enc_output)
        return final_output, attention_weights


