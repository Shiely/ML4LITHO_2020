import tensorflow as tf 
import numpy as np

def log_images(stack_name, image_stack):
    image_unstack = tf.unstack(image_stack, axis=3)
    for i, image in enumerate(image_unstack):
        tf.summary.image(stack_name+'_'+str(i), tf.expand_dims(image, axis=3))
        
class bayes_CVAE(tf.keras.Model):
    def __init__(self,hparams):
        super(bayes_CVAE, self).__init__()
        print('hparams'.format(hparams))
        image_dim=hparams['image_dim']
        self.hparams=hparams
        print('image_dim {}'.format(image_dim))

        self.design_inference_net = tf.keras.Sequential(
            [
            tf.keras.layers.InputLayer(input_shape=(self.hparams['image_dim'], self.hparams['image_dim'],2)),
            tf.keras.layers.Conv2D(filters=16, kernel_size=4, strides=2, padding='same', activation=tf.nn.leaky_relu),
            tf.keras.layers.Dropout(rate=self.hparams['dropout_rate']),
            tf.keras.layers.Conv2D(filters=16, kernel_size=4, strides=2, padding='same', activation=tf.nn.leaky_relu),
            tf.keras.layers.Dropout(rate=self.hparams['dropout_rate']),
            tf.keras.layers.Conv2D(filters=32, kernel_size=4, strides=1, padding='same', activation=tf.nn.leaky_relu),
            tf.keras.layers.Dropout(rate=self.hparams['dropout_rate']),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=2 * self.hparams['n_latent'])
            ]
        )
        
        self.design_generative_net = tf.keras.Sequential(  
            [       
            tf.keras.layers.InputLayer(input_shape=(self.hparams['n_latent'])),
            tf.keras.layers.Dense(units=7*7, activation=tf.nn.leaky_relu),
            tf.keras.layers.Reshape(target_shape=(7, 7, 1)),
            tf.keras.layers.Conv2DTranspose(
                filters=32,
                kernel_size=4,
                strides=(1, 1),
                padding="SAME",
                activation=tf.nn.leaky_relu),
            tf.keras.layers.Dropout( rate=self.hparams['dropout_rate'] ),
            tf.keras.layers.Conv2DTranspose(
                filters=16,
                kernel_size=4,
                strides=(2, 2),
                padding="SAME",
                activation=tf.nn.leaky_relu),
            tf.keras.layers.Dropout( rate=self.hparams['dropout_rate'] ),
            tf.keras.layers.Conv2DTranspose(
                filters=16,
                kernel_size=4,
                strides=(2, 2),
                padding="SAME",
                activation=tf.nn.leaky_relu),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=self.hparams['image_dim']*self.hparams['image_dim']*2),
            tf.keras.layers.Reshape((self.hparams['image_dim'],self.hparams['image_dim'],2)),
            ]
        )
        self.mask_inference_net = tf.keras.Sequential(
            [
            tf.keras.layers.InputLayer(input_shape=(self.hparams['image_dim'] ,self.hparams['image_dim'])),
            tf.keras.layers.Reshape(target_shape=(self.hparams['image_dim'] ,self.hparams['image_dim'],1)),
            tf.keras.layers.Conv2D(filters=16, kernel_size=4, strides=2, padding='same', activation=tf.nn.leaky_relu),
            tf.keras.layers.Dropout(rate=self.hparams['dropout_rate']),
            tf.keras.layers.Conv2D(filters=16, kernel_size=4, strides=2, padding='same', activation=tf.nn.leaky_relu),
            tf.keras.layers.Dropout(rate=self.hparams['dropout_rate']),
            tf.keras.layers.Conv2D(filters=32, kernel_size=4, strides=1, padding='same', activation=tf.nn.leaky_relu),
            tf.keras.layers.Dropout(rate=self.hparams['dropout_rate']),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=2 * self.hparams['n_latent']),

            ]
        )
        
        self.mask_generative_net = tf.keras.Sequential(
            [
            tf.keras.layers.InputLayer(input_shape=(self.hparams['n_latent'],)),
            tf.keras.layers.Dense(units=7*7, activation=tf.nn.leaky_relu),
            tf.keras.layers.Reshape(target_shape=(7, 7, 1)),
            tf.keras.layers.Conv2DTranspose(
                filters=32,
                kernel_size=4,
                strides=(1, 1),
                padding="SAME",
                activation=tf.nn.leaky_relu),
            tf.keras.layers.Dropout( rate=self.hparams['dropout_rate'] ),
            tf.keras.layers.Conv2DTranspose(
                filters=16,
                kernel_size=4,
                strides=(2, 2),
                padding="SAME",
                activation=tf.nn.leaky_relu),
            tf.keras.layers.Dropout( rate=self.hparams['dropout_rate'] ),
            tf.keras.layers.Conv2DTranspose(
                filters=16,
                kernel_size=4,
                strides=(2, 2),
                padding="SAME",
                activation=tf.nn.leaky_relu),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=self.hparams['image_dim']*self.hparams['image_dim']),
            tf.keras.layers.Reshape((self.hparams['image_dim'],self.hparams['image_dim'],1)), 
            ]
        )
    @tf.function
    def sample_designs(self, eps=None):
        if eps is None:
          eps = tf.random.normal(shape=(100, self.hparams['n_latent']))
        return self.decode_designs(eps, apply_sigmoid=True)

    def sample_masks(self, eps=None):
        if eps is None:
          eps = tf.random.normal(shape=(100, self.hparams['n_latent']))
        return self.decode_masks(eps, apply_sigmoid=True)


    def encode_designs(self, x):
        mean, logvar = tf.split(self.design_inference_net(x,), num_or_size_splits=2, axis=-1)
        return mean, logvar
    
    def encode_masks(self, x):
        mean, logvar = tf.split(self.mask_inference_net((x,)), num_or_size_splits=2, axis=-1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=[self.hparams['batch_size'],self.hparams['n_latent']])
        return eps * tf.exp(logvar * .5) + mean

    def decode_designs(self, z, apply_sigmoid=False):
        logits_design, logits_context=tf.split(self.design_generative_net(z),num_or_size_splits=2, axis=-1)
        if apply_sigmoid:
            probs_design = tf.sigmoid(logits_design)
            probs_context = tf.sigmoid(logits_context)
            return probs_design, probs_context
        return logits_design, logits_context
                                               
    def decode_masks(self, z, apply_sigmoid=False):
        logits = self.mask_generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits  
def get_bayes_estimator(run_config, hparams):
    def _model_fn(features, labels, mode, params):

        masksin   =tf.cast(tf.reshape(features["masks"]   ,[-1,hparams['image_dim'],hparams['image_dim'],1]), tf.float32)
        designsin =tf.cast(tf.reshape(features["designs"] ,[-1,hparams['image_dim'],hparams['image_dim'],1]), tf.float32)
        contextsin=tf.cast(tf.reshape(features["contexts"],[-1,hparams['image_dim'],hparams['image_dim'],1]), tf.float32)

        if not hparams['context_on']:
            contextsin=contextsin*0.0
        
        log_images('input_masks'   ,masksin)
        log_images('input_designs' ,designsin)
        log_images('input_contexts',contextsin)    
        
        encoder_hidden_units = params["encoder_hidden_units"]
        decoder_hidden_units = encoder_hidden_units
        output_layer_size = hparams['image_dim']*hparams['image_dim']

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        
        model = bayes_CVAE(hparams)
        z_mask, logvar_mask= model.encode_masks(masksin)
#        z_design_and_context, logvar_design_and_context= model.encode_designs(tf.stack([tf.squeeze(designsin),tf.squeeze(contextsin)],axis=-1))
        z_design_and_context, logvar_design_and_context=model.encode_designs(
            tf.concat([designsin,contextsin],axis=-1))

        if is_training:
            z_mask = model.reparameterize(z_mask,logvar_mask)
            z_design_and_context = model.reparameterize(z_design_and_context, logvar_design_and_context)#for loss?# imge_dim x image_dim output pixel vector
        output_mask = model.decode_masks(z_mask, apply_sigmoid=True)
        reconstruction_output_mask=tf.squeeze(output_mask)
        output_design, output_context = model.decode_designs(z_design_and_context ,apply_sigmoid=True)
        reconstruction_output_design=tf.squeeze(output_design)
        reconstruction_output_context=tf.squeeze(output_context) 
        log_images('reconstructed_masks'   ,output_mask)
        log_images('reconstructed_designs'   ,output_design)   
        log_images('reconstructed_contexts'   ,output_context)    
        # Provide an estimator spec for `ModeKeys.PREDICT`.
        if mode == tf.estimator.ModeKeys.PREDICT:
            # Convert predicted_indices back into strings.
            predictions = {
                'mask_encoding': z_mask,
                'mask_reconstruction': output_mask,
                'design_and_context_encoding': z_design_and_context,
                'design_reconstruction': output_design,
                'context_reconstruction': output_context,
            }
            export_outputs = {
                'predict': tf.estimator.export.PredictOutput(predictions)
            }
            # Provide an estimator spec for `ModeKeys.PREDICT` modes.
            return tf.estimator.EstimatorSpec(
            mode, predictions=predictions, export_outputs=export_outputs)


        # Define loss based on reconstruction and regularization.

        # Create optimizer
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=params["learning_rate"], epsilon=1e-3)

        def log_normal_pdf(sample, mean, logvar, raxis=1):
            log2pi = tf.math.log(2. * np.pi)
            return tf.reduce_sum(
                -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
                axis=raxis)

        @tf.function
        def compute_loss(model, masks, designs, contexts):
            mask_target=masks
            mask_mean, logvar = model.encode_masks(masks)
            print('mean {} logvar {}'.format(mask_mean,logvar))
            z = model.reparameterize(mask_mean, logvar)
            mask_logit=model.decode_masks(z)
            mask_cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=tf.squeeze(mask_logit), labels=tf.squeeze(mask_target))
            logpx_z = -tf.reduce_sum(mask_cross_ent, axis=[1, 2])
            logpz = log_normal_pdf(z, 0., 0.)
            logqz_x = log_normal_pdf(z, mask_mean, logvar)
            mask_loss= -tf.reduce_mean(logpx_z + logpz - logqz_x)            

            design_target=designs
            context_target=contexts
            design_mean, logvar = model.encode_designs(
                tf.concat([designs,contexts],axis=-1))


            print('mean {} logvar {}'.format(design_mean,logvar))
            z = model.reparameterize(design_mean, logvar)
            design_logit, context_logit = model.decode_designs(z)
            print(design_logit.shape, context_logit.shape)
            design_cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=tf.squeeze(design_logit), labels=tf.squeeze(design_target))
            logpx_z = -tf.reduce_sum(design_cross_ent, axis=[1, 2])
            logpz = log_normal_pdf(z, 0., 0.)
            logqz_x = log_normal_pdf(z, design_mean, logvar)
            design_loss= -tf.reduce_mean(logpx_z + logpz - logqz_x)            

            context_cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=tf.squeeze(context_logit), labels=tf.squeeze(context_target))
            logpx_z = -tf.reduce_sum(context_cross_ent, axis=[1, 2])
            logpz = log_normal_pdf(z, 0., 0.)
            logqz_x = log_normal_pdf(z, design_mean, logvar)
            context_loss= -tf.reduce_mean(logpx_z + logpz - logqz_x)   

            embedding_binding_loss = tf.reduce_sum(tf.keras.losses.cosine_similarity(mask_mean,design_mean))

            return mask_loss+design_loss+context_loss+hparams['beta']*embedding_binding_loss


        loss = compute_loss(model, masksin, designsin, contextsin)
            
        train_op = optimizer.minimize(loss=loss,
                                          global_step=tf.compat.v1.train.get_global_step())
        eval_metric_ops = {
            'rmse_mask': tf.compat.v1.metrics.root_mean_squared_error(masksin,
                                                     reconstruction_output_mask),
            'rmse_design': tf.compat.v1.metrics.root_mean_squared_error(designsin,
                                                     reconstruction_output_design),
            'rmse_context': tf.compat.v1.metrics.root_mean_squared_error(contextsin,
                                                     reconstruction_output_context),
        }

        estimator_spec = tf.estimator.EstimatorSpec(mode=mode,
                                                    loss=loss,
                                                    train_op=train_op,
                                                    eval_metric_ops=eval_metric_ops)
        return estimator_spec

    return tf.estimator.Estimator(model_fn=_model_fn, params=hparams, config=run_config)