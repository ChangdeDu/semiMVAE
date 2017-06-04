from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import prettytensor as pt
import tensorflow as tf
import numpy as np
import utils

from neuralnetworks import FullyConnected
from prettytensor import bookkeeper

class GenerativeClassifier( object ):

    def __init__(   self, dim_x1, dim_x2, dim_z, dim_y,
					num_examples, num_lab, num_batches,
					p_x1 = 'gaussian',
                p_x2 = 'gaussian',
					q_z = 'gaussian_mixture_marg',
					p_z = 'gaussian_marg',
					hidden_layers_px1 = [250],
                     hidden_layers_px2 = [250],
					hidden_layers_qz = [250],
					hidden_layers_qy = [250],
					nonlin_px1 = tf.nn.softplus,
                     nonlin_px2 = tf.nn.softplus,
					nonlin_qz = tf.nn.softplus,
					nonlin_qy = tf.nn.softplus,
					beta = 0.1,
					l2_loss = 0.0,
                     zeta_reg = 1e6  ):


        self.dim_x1, self.dim_x2, self.dim_z, self.dim_y = dim_x1, dim_x2, dim_z, dim_y

        self.zeta_reg = zeta_reg
        self.distributions = { 		'p_x1': 	p_x1,
                                     'p_x2': 	p_x2,
									'q_z': 	q_z,			
									'p_z': 	p_z,            
									'p_y':	'uniform'	}

        self.num_examples = num_examples
        self.num_batches = num_batches
        self.num_lab = num_lab
        self.num_ulab = self.num_examples - num_lab

#        assert self.num_lab % self.num_batches == 0, '#Labelled % #Batches != 0'
#        assert self.num_ulab % self.num_batches == 0, '#Unlabelled % #Batches != 0'
#        assert self.num_examples % self.num_batches == 0, '#Examples % #Batches != 0'
       
        self.num_lab_batch = self.num_lab // self.num_batches
        self.num_ulab_batch = self.num_ulab // self.num_batches
        self.batch_size = self.num_examples // self.num_batches

#        self.alpha = beta * ( float(self.batch_size) / self.num_lab_batch )
        self.alpha = beta *  float(self.batch_size) 

        ''' Create Graph '''

        self.G = tf.Graph()

        with self.G.as_default():

            def weight_variable(shape):
                initial = tf.truncated_normal(shape, stddev=0.01)
                return tf.Variable(initial)
#            zeta = weight_variable([2])
            init_value = (np.sqrt(1.0/2)).astype('float32')
            zeta = tf.Variable(tf.constant(init_value, shape=[2]))
#            zeta = tf.constant(init_value, shape=[2])
            zeta = zeta * zeta 
            self.zeta1 = zeta[0]
            self.zeta2 = zeta[1]
            
            self.x1_labelled_mu 			= tf.placeholder( tf.float32, [None, self.dim_x1] )
            self.x1_labelled_lsgms 		= tf.placeholder( tf.float32, [None, self.dim_x1] )
            self.x1_unlabelled_mu 		= tf.placeholder( tf.float32, [None, self.dim_x1] )
            self.x1_unlabelled_lsgms 	    = tf.placeholder( tf.float32, [None, self.dim_x1] )
            self.x2_labelled_mu 			= tf.placeholder( tf.float32, [None, self.dim_x2] )
            self.x2_labelled_lsgms 		= tf.placeholder( tf.float32, [None, self.dim_x2] )
            self.x2_unlabelled_mu 		= tf.placeholder( tf.float32, [None, self.dim_x2] )
            self.x2_unlabelled_lsgms    	= tf.placeholder( tf.float32, [None, self.dim_x2] )
            self.y_lab      			    = tf.placeholder( tf.float32, [None, self.dim_y] )

            self.classifier = FullyConnected( 	dim_output 		= self.dim_y, 
												hidden_layers 	= hidden_layers_qy,
												nonlinearity 	= nonlin_qy,
												l2loss 			= l2_loss 	)

            self.encoder_1 = FullyConnected( 		dim_output 		= 2 * self.dim_z,
												hidden_layers 	= hidden_layers_qz,
												nonlinearity 	= nonlin_qz,
												l2loss 			= l2_loss 	)
            
            self.encoder_2 = FullyConnected( 		dim_output 		= 2 * self.dim_z,
												hidden_layers 	= hidden_layers_qz,
												nonlinearity 	= nonlin_qz,
												l2loss 			= l2_loss 	)

            self.decoder_1 = FullyConnected( 		dim_output 		= 2 * self.dim_x1,
												hidden_layers 	= hidden_layers_px1,
												nonlinearity 	= nonlin_px1,
												l2loss 			= l2_loss 	)
            
            self.decoder_2 = FullyConnected( 		dim_output 		= 2 * self.dim_x2,
												hidden_layers 	= hidden_layers_px2,
												nonlinearity 	= nonlin_px2,
												l2loss 			= l2_loss 	)

            self._objective()
            self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
            self.session = tf.Session()



    def _draw_sample( self, mu, log_sigma_sq ):

        epsilon = tf.random_normal( ( tf.shape( mu ) ), 0, 1 )
        sample = tf.add( mu, 
				 tf.mul(  
				 tf.exp( 0.5 * log_sigma_sq ), epsilon ) )

        return sample

    def _generate_yx1x2( self, x1_mu, x2_mu, x1_log_sigma_sq, x2_log_sigma_sq, phase = pt.Phase.train, reuse = False ):

        x1_sample = self._draw_sample( x1_mu, x1_log_sigma_sq )
        x2_sample = self._draw_sample( x2_mu, x2_log_sigma_sq )
        with tf.variable_scope('classifier', reuse = reuse):
            y_logits = self.classifier.output( tf.concat( 1, [x1_sample, x2_sample] ) , phase )

        return y_logits, x1_sample, x2_sample

    def _generate_zx1y( self, x1, y, reuse = False ):

        with tf.variable_scope('encoder_1', reuse = reuse):
            encoder_out = self.encoder_1.output( tf.concat( 1, [x1, y] ) )
            z1_mu, z1_lsgms   = encoder_out.split( split_dim = 1, num_splits = 2 )
        z1_sample        = self._draw_sample( z1_mu, z1_lsgms )

        return z1_sample, z1_mu, z1_lsgms 
        
    def _generate_zx2y( self, x2, y, reuse = False ):

        with tf.variable_scope('encoder_2', reuse = reuse):
            encoder_out = self.encoder_2.output( tf.concat( 1, [x2, y] ) )
            z2_mu, z2_lsgms   = encoder_out.split( split_dim = 1, num_splits = 2 )
        z2_sample        = self._draw_sample( z2_mu, z2_lsgms )

        return z2_sample, z2_mu, z2_lsgms 
        
    def _mixture_z(self, z1, z2, phase = pt.Phase.train, reuse = False ):
        
        z_sample = self.zeta1 * z1 + self.zeta2 * z2
        
        return z_sample

    def _generate_x1zy( self, z, y, reuse = False ):

        with tf.variable_scope('decoder_1', reuse = reuse):
            decoder_out = self.decoder_1.output( tf.concat( 1, [z, y] ) )
        x1_recon_mu, x1_recon_lsgms   = decoder_out.split( split_dim = 1, num_splits = 2 )
#        x1_recon_mu = tf.nn.sigmoid( x1_recon_mu )
#        x1_recon_lsgms = tf.nn.sigmoid( x1_recon_lsgms )
        
        return x1_recon_mu, x1_recon_lsgms

    def _generate_x2zy( self, z, y, reuse = False ):

        with tf.variable_scope('decoder_2', reuse = reuse):
            decoder_out = self.decoder_2.output( tf.concat( 1, [z, y] ) )
        x2_recon_mu, x2_recon_lsgms   = decoder_out.split( split_dim = 1, num_splits = 2 )
#        x2_recon_mu = tf.nn.sigmoid( x2_recon_mu )
#        x2_recon_lsgms = tf.nn.sigmoid( x2_recon_lsgms )
        
        return x2_recon_mu, x2_recon_lsgms
        
    def _objective( self ):

		###############
        ''' L(x,y) ''' 
		###############

        def L(x1_recon_z1, x1_recon_z2, x2_recon_z1, x2_recon_z2, x1, x2, y, z1, z2, zeta1, zeta2):

            if self.distributions['p_z'] == 'gaussian_marg':

                log_prior_z1 = tf.reduce_sum( utils.tf_gaussian_marg( z1[1], z1[2] ), 1 )
                log_prior_z2 = tf.reduce_sum( utils.tf_gaussian_marg( z2[1], z2[2] ), 1 )
                log_prior_z = tf.add( tf.mul(zeta1, log_prior_z1) , tf.mul(zeta2, log_prior_z2))
            elif self.distributions['p_z'] == 'gaussian':

                log_prior_z1 = tf.reduce_sum( utils.tf_stdnormal_logpdf( z1[0] ), 1 )
                log_prior_z2 = tf.reduce_sum( utils.tf_stdnormal_logpdf( z2[0] ), 1 )
                log_prior_z = tf.add( tf.mul(zeta1, log_prior_z1) , tf.mul(zeta2, log_prior_z2))

            if self.distributions['p_y'] == 'uniform':

                y_prior = (1. / self.dim_y) * tf.ones_like( y )
                log_prior_y = - tf.nn.softmax_cross_entropy_with_logits( y_prior, y )

            if self.distributions['p_x1'] == 'gaussian':

                log_lik_x1_z1 = tf.reduce_sum( utils.tf_normal_logpdf( x1, x1_recon_z1[0], x1_recon_z1[1] ), 1 )
                log_lik_x1_z2 = tf.reduce_sum( utils.tf_normal_logpdf( x1, x1_recon_z2[0], x1_recon_z2[1] ), 1 )
                log_lik_x1 = zeta1 * log_lik_x1_z1 + zeta2 * log_lik_x1_z2
                
            if self.distributions['p_x2'] == 'gaussian':

                log_lik_x2_z1 = tf.reduce_sum( utils.tf_normal_logpdf( x2, x2_recon_z1[0], x2_recon_z1[1] ), 1 )
                log_lik_x2_z2 = tf.reduce_sum( utils.tf_normal_logpdf( x2, x2_recon_z2[0], x2_recon_z2[1] ), 1 )
                log_lik_x2 = zeta1 * log_lik_x2_z1 + zeta2 * log_lik_x2_z2

            if self.distributions['q_z'] == 'gaussian_mixture_marg':

                log_post_z = tf.reduce_sum( utils.tf_gaussian_mixture_ent( z1[1], z2[1], z1[2], z2[2], zeta1, zeta2), 1 )

            _L =  log_lik_x1 + log_lik_x2 + log_prior_y + log_prior_z - log_post_z

            return  _L

		###########################
        ''' Labelled Datapoints '''
		###########################

        self.y_lab_logits, self.x1_lab, self.x2_lab = self._generate_yx1x2( self.x1_labelled_mu, self.x2_labelled_mu, self.x1_labelled_lsgms, self.x2_labelled_lsgms )
        self.z1_lab, self.z1_lab_mu, self.z1_lab_lsgms = self._generate_zx1y( self.x1_lab, self.y_lab )
        self.z2_lab, self.z2_lab_mu, self.z2_lab_lsgms = self._generate_zx2y( self.x2_lab, self.y_lab )
        
        self.x1_recon_lab_mu_z1, self.x1_recon_lab_lsgms_z1 = self._generate_x1zy( self.z1_lab, self.y_lab )
        self.x1_recon_lab_mu_z2, self.x1_recon_lab_lsgms_z2 = self._generate_x1zy( self.z2_lab, self.y_lab, reuse = True )
        
        self.x2_recon_lab_mu_z1, self.x2_recon_lab_lsgms_z1 = self._generate_x2zy( self.z1_lab, self.y_lab )
        self.x2_recon_lab_mu_z2, self.x2_recon_lab_lsgms_z2 = self._generate_x2zy( self.z2_lab, self.y_lab, reuse = True )

        L_lab = L([self.x1_recon_lab_mu_z1, self.x1_recon_lab_lsgms_z1],
                  [self.x1_recon_lab_mu_z2, self.x1_recon_lab_lsgms_z2],
                  [self.x2_recon_lab_mu_z1, self.x2_recon_lab_lsgms_z1],
                  [self.x2_recon_lab_mu_z2, self.x2_recon_lab_lsgms_z2],
                  self.x1_lab, self.x2_lab, self.y_lab,
				  [self.z1_lab, self.z1_lab_mu, self.z1_lab_lsgms], 
                  [self.z2_lab, self.z2_lab_mu, self.z2_lab_lsgms], 
                  self.zeta1, self.zeta2 )

        L_lab += - self.alpha * tf.nn.softmax_cross_entropy_with_logits( self.y_lab_logits, self.y_lab )

		############################
        ''' Unabelled Datapoints '''
		############################

        def one_label_tensor( label ):

            indices = []
            values = []
            for i in range(self.num_ulab_batch):
                indices += [[ i, label ]]
                values += [ 1. ]

            _y_ulab = tf.sparse_tensor_to_dense( 
					  tf.SparseTensor( indices=indices, values=values, shape=[ self.num_ulab_batch, self.dim_y ] ), 0.0 )

            return _y_ulab

        self.y_ulab_logits, self.x1_ulab, self.x2_ulab = self._generate_yx1x2( self.x1_unlabelled_mu, self.x2_unlabelled_mu, self.x1_unlabelled_lsgms, self.x2_unlabelled_lsgms, reuse = True )

        for label in range(self.dim_y):

            _y_ulab = one_label_tensor( label )
            self.z1_ulab, self.z1_ulab_mu, self.z1_ulab_lsgms = self._generate_zx1y( self.x1_ulab, _y_ulab, reuse = True )
            self.z2_ulab, self.z2_ulab_mu, self.z2_ulab_lsgms = self._generate_zx2y( self.x2_ulab, _y_ulab, reuse = True )
            
            self.x1_recon_ulab_mu_z1, self.x1_recon_ulab_lsgms_z1 = self._generate_x1zy( self.z1_ulab, _y_ulab, reuse = True )
            self.x1_recon_ulab_mu_z2, self.x1_recon_ulab_lsgms_z2 = self._generate_x1zy( self.z2_ulab, _y_ulab, reuse = True )
            
            self.x2_recon_ulab_mu_z1, self.x2_recon_ulab_lsgms_z1 = self._generate_x2zy( self.z1_ulab, _y_ulab, reuse = True )
            self.x2_recon_ulab_mu_z2, self.x2_recon_ulab_lsgms_z2 = self._generate_x2zy( self.z2_ulab, _y_ulab, reuse = True )
            
            _L_ulab =   tf.expand_dims(
						L( [self.x1_recon_ulab_mu_z1, self.x1_recon_ulab_lsgms_z1],
                           [self.x1_recon_ulab_mu_z2, self.x1_recon_ulab_lsgms_z2],
                           [self.x2_recon_ulab_mu_z1, self.x2_recon_ulab_lsgms_z1],
                           [self.x2_recon_ulab_mu_z2, self.x2_recon_ulab_lsgms_z2],
                           self.x1_ulab, self.x2_ulab, _y_ulab, 
						  [self.z1_ulab, self.z1_ulab_mu, self.z1_ulab_lsgms],
                           [self.z2_ulab, self.z2_ulab_mu, self.z2_ulab_lsgms], 
                           self.zeta1, self.zeta2), 1)

            if label == 0: L_ulab = tf.identity( _L_ulab )
            else: L_ulab = tf.concat( 1, [L_ulab, _L_ulab] )

        self.y_ulab = self.y_ulab_logits.softmax_activation()

        U = tf.reduce_sum( 
			tf.mul( self.y_ulab, 
			tf.sub( L_ulab, 
			tf.log( self.y_ulab ) ) ), 1 )

		########################
        ''' Prior on Weights '''
		########################

        L_weights = 0.
        _weights = tf.trainable_variables()
        for w in _weights: 
            L_weights += tf.reduce_sum( utils.tf_stdnormal_logpdf( w ) )

		##################
        ''' Total Cost '''
		##################

        L_lab_tot = tf.reduce_sum( L_lab )
        U_tot = tf.reduce_sum( U )
        zeta_loss = tf.pow(self.zeta1 + self.zeta2 -1, 2)
        
        self.cost = ( ( L_lab_tot + U_tot - self.zeta_reg * zeta_loss) * self.num_batches + L_weights ) / ( 
				- self.num_batches * self.batch_size )

		##################
        ''' Evaluation '''
		##################

        self.y_test_logits, _ , _ = self._generate_yx1x2( self.x1_labelled_mu, self.x2_labelled_mu, self.x1_labelled_lsgms, self.x2_labelled_lsgms,
			phase = pt.Phase.test, reuse = True )
        self.y_test_pred = self.y_test_logits.softmax( self.y_lab )

        self.eval_accuracy = self.y_test_pred\
				.softmax.evaluate_classifier( self.y_lab, phase = pt.Phase.test )
        self.eval_cross_entropy = self.y_test_pred.loss
        self.eval_precision, self.eval_recall = self.y_test_pred.softmax\
				.evaluate_precision_recall( self.y_lab, phase = pt.Phase.test )


    def train(      self, x1_labelled, x2_labelled, y, x1_unlabelled, x2_unlabelled,
					epochs,
					x1_valid, x2_valid, y_valid,
					print_every = 1,
					learning_rate = 3e-4,
					beta1 = 0.9,
					beta2 = 0.999,
					seed = 31415,
					stop_iter = 100,
					save_path = None,
					load_path = None    ):


        ''' Session and Summary '''
        if save_path is None: 
            self.save_path = 'models/model_GC_{}-{}-{}.cpkt'.format(
				self.num_lab,learning_rate,self.batch_size)
        else:
            self.save_path = save_path

        np.random.seed(seed)
        tf.set_random_seed(seed)

        with self.G.as_default():

            self.optimiser = tf.train.AdamOptimizer( learning_rate = learning_rate, beta1 = beta1, beta2 = beta2 )
            self.train_op = self.optimiser.minimize( self.cost )
            init = tf.global_variables_initializer()
            self._test_vars = None
			
		
        _data_labelled = np.hstack( [x1_labelled, x2_labelled, y] )
        _data_unlabelled = np.hstack( [x1_unlabelled, x2_unlabelled] ) 
        x1_valid_mu, x1_valid_lsgms = x1_valid[ :, :self.dim_x1 ], x1_valid[ :, self.dim_x1:2*self.dim_x1 ]
        x2_valid_mu, x2_valid_lsgms = x2_valid[ :, :self.dim_x2 ], x2_valid[ :, self.dim_x2:2*self.dim_x2 ]

        with self.session as sess:

            sess.run(init)
            if load_path == 'default': self.saver.restore( sess, self.save_path )
            elif load_path is not None: self.saver.restore( sess, load_path )	

            best_eval_accuracy = 0.
            stop_counter = 0

            for epoch in range(epochs):

                ''' Shuffle Data '''
                np.random.shuffle( _data_labelled )
                np.random.shuffle( _data_unlabelled )
                

                ''' Training '''
				
                for x1_l_mu, x2_l_mu, x1_l_lsgms, x2_l_lsgms, y, x1_u_mu, x2_u_mu, x1_u_lsgms, x2_u_lsgms in utils.feed_numpy_semisupervised_multiview(	
					self.num_lab_batch,
                    self.num_ulab_batch, 
                    _data_labelled[:, :2*self.dim_x1], 
                    _data_labelled[:, 2*self.dim_x1:(2*self.dim_x1 + 2*self.dim_x2)], 
                    _data_labelled[:, (2*self.dim_x1 + 2*self.dim_x2):], 
                    _data_unlabelled[:, :2*self.dim_x1], 
                    _data_unlabelled[:, 2*self.dim_x1:] ):

                    training_result = sess.run( [self.train_op, self.cost, self.zeta1, self.zeta2],
											feed_dict = {	self.x1_labelled_mu:			x1_l_mu,
                                                              self.x2_labelled_mu:			x2_l_mu,
															self.x1_labelled_lsgms: 		x1_l_lsgms,
                                                              self.x2_labelled_lsgms: 		x2_l_lsgms,
															self.y_lab: 				y,
															self.x1_unlabelled_mu: 		x1_u_mu,
                                                              self.x2_unlabelled_mu: 		x2_u_mu,
															self.x1_unlabelled_lsgms: 	x1_u_lsgms,
                                                              self.x2_unlabelled_lsgms: 	x2_u_lsgms} )

                    training_cost = training_result[1]
                    zeta1 = training_result[2]
                    zeta2 = training_result[3]

                ''' Evaluation '''

                stop_counter += 1

                if epoch % print_every == 0:

                    test_vars = tf.get_collection(bookkeeper.GraphKeys.TEST_VARIABLES)
                    if test_vars:
                        if test_vars != self._test_vars:
                            self._test_vars = list(test_vars)
                            self._test_var_init_op = tf.variables_initializer(test_vars)
                        self._test_var_init_op.run()


                    eval_accuracy, eval_cross_entropy = \
						sess.run( [self.eval_accuracy, self.eval_cross_entropy],
									feed_dict = { 	self.x1_labelled_mu: 	x1_valid_mu,
                                                      self.x2_labelled_mu: 	x2_valid_mu,
													self.x1_labelled_lsgms:	x1_valid_lsgms,
                                                      self.x2_labelled_lsgms:	x2_valid_lsgms,
													self.y_lab:				y_valid } )
      
                    if eval_accuracy > best_eval_accuracy:

                        best_eval_accuracy = eval_accuracy
                        self.saver.save( sess, self.save_path )
                        stop_counter = 0

                    utils.print_metrics( 	epoch+1,
											['Training', 'cost', training_cost],
											['Validation', 'accuracy', eval_accuracy],
											['Validation', 'cross-entropy', eval_cross_entropy],
                                             ['Training', 'zeta1', zeta1],
                                             ['Training', 'zeta2', zeta2])

                if stop_counter >= stop_iter:
                    print('Stopping GC training')
                    print('No change in validation accuracy for {} iterations'.format(stop_iter))
                    print('Best validation accuracy: {}'.format(best_eval_accuracy))
                    print('Model saved in {}'.format(self.save_path))
                    break

    def predict_labels( self, x1_test, x2_test, y_test ):
     
        test_vars = tf.get_collection(bookkeeper.GraphKeys.TEST_VARIABLES)
        tf.initialize_variables(test_vars).run()

        x1_test_mu = x1_test[:,:self.dim_x1]
        x1_test_lsgms = x1_test[:,self.dim_x1:2*self.dim_x1]
        x2_test_mu = x2_test[:,:self.dim_x2]
        x2_test_lsgms = x2_test[:,self.dim_x2:2*self.dim_x2]

        accuracy, cross_entropy, precision, recall, y_test_pred= \
			self.session.run( [self.eval_accuracy, self.eval_cross_entropy, self.eval_precision, self.eval_recall, self.y_test_pred],
				feed_dict = {self.x1_labelled_mu: x1_test_mu,
                             self.x2_labelled_mu: x2_test_mu,
                             self.x1_labelled_lsgms: x1_test_lsgms, 
                             self.x2_labelled_lsgms: x2_test_lsgms,
                             self.y_lab: y_test} )
        
        utils.print_metrics(	'X',
								['Test', 'accuracy', accuracy],
								['Test', 'cross-entropy', cross_entropy],
								['Test', 'precision', precision],
				 				['Test', 'recall', recall])
  
        return  y_test_pred