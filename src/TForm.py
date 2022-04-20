import tensorflow as tf


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    '''
    The in band function takes in a matrix, num_lower and num_upper
    num_lower is valid if m - n <= num_lower
    num_upper is valid if n - m <= num_upper (in our case this mask
        will be the everything above the central digonal)
    '''
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


'''
It seems that dff can be whatever I want it to be
'''
def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu')
            tf.keras.layers.Dense(d_model) # No activation on last layer?
        ])

'''
    q: ... * seq_len * d_model 
    k: ... * seq_len * d_model 
    v: ... * seq_len * d_model 
'''
def scaled_dot_product_attention(q, k, v, mask):
    similarity = tf.matmul(q, k, transpose_b=True) # ... * seq_len * seq_len
    dk = tf.cast(tf.shape(k)[1], tf.float32)

    # Scaled Attention
    scaled_attention = similarity / tf.math.sqrt(dk)

    # Add Mask
    if mask is not None:
        scaled_attention += (mask * -1e9)

    # Similarities are on the last axis, so we apply softmax on the last axis
    weights = tf.nn.softmax(scaled_attention, axis=-1)

    # Find a weighted sum of the values based on weights
    output = tf.matmul(weights, v) # ... * seq_len * depth

    return output, weights


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
    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]

        q = tf.concat([ql(q)[:, None, :, :] for ql in self.wqs], axis=1)
        k = tf.concat([kl(k)[:, None, :, :] for kl in self.wks], axis=1)
        v = tf.concat([vl(v)[:, None, :, :] for vl in self.wvs], axis=1)

        scaled_attention, attention_weights = scaled_dot_product_attention(
                q, k, v, mask)

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

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training) # need to be separate?
        out2 = self.layernorm2(out1 + ffn_output)
        return out2


# I haven't used the decoder yet so I'm not sure how well it will work
# Example mha: v, k, q
# Mind: q, k, v
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.mha2 = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Droupout(rate)
        self.dropout2 = tf.keras.layers.Droupout(rate)
        self.dropout3 = tf.keras.layers.Droupout(rate)

    def call(self, x, enc_output, training, 
            look_agead_mask, padding_mask):

        # I thought this should be related to the outputs. Unless x is
        # part of the outputs
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
                out1, enc_output, enc_output, padding=mask)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2


## The encoder in the tutorial uses some sort of embedding function to turn vocbulary
## indices into a dense vector. I don't think that is necessary here
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(MAX_TOKENS, self.d_model)

        self.enc_layers = [
                EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, rate=rate)
                for _ in num_layers]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        x = self.embedding(x) # don't know if I'll need this
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32)) # Don't see why necessary
        x += self.pos_encoding[:, :seq_len, :] # What even is the positional encoding

        x = self.dropout(x, training=training) # Dropout after positional encoding

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_heads, dff, target_vocab_size, rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(MAX_TOKENS, d_model)

        self.dec_layers = [
                DecoderLayer(d_model=d_modl, num_heads=num_heads, dff=dff, rate=rate)
                for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {} 

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, 
                                                    look_ahead_mask, padding_mask)
            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2

        return x, attention_weights


class Transformer(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff, input_vocab_size,
            target_vocab_size, rate=0.1):
        super().__init__() # Don't need to specify transformer here
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model, 
                                num_heads=num_heads, dff=dff,
                                input_vocab_size=input_vocab_size, rate=rate)
        
        self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                                num_heads=num_heads, dff=dff,
                                target_vocab_size=target_vocab_size, rate=rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, training):
        inp, tar = inputs
        padding_mask, look_ahead_mask = self.create_masks(inp, tar)

        enc_output = self.encoder(inp, training, padding_mask)

        dec_output, attention_weights = self.decoder(
                tar, enc_output, training, look_ahead_mask, padding_mask)

        final_output = self.final_layer(dec_output)

        return final_output, attention_weights

    def create_masks(self, inp, tar):
        create_padding_mask(inp) # ?????k I have no idea whats going on ehre
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return padding_mask, look_ahead_mask
                    


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


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask) # ends up being an average

def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))
    pass ## AS OF NOW I"M just copying. I should start the application of this
         # stuff
    





