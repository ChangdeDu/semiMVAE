import tensorflow as tf
import numpy as np

logc = np.log(2.*np.pi)
c = - 0.5 * np.log(2*np.pi)

def tf_normal_logpdf(x, mu, log_sigma_sq):

    return ( - 0.5 * logc - log_sigma_sq / 2. - tf.div( tf.square( tf.sub( x, mu ) ), 2 * tf.exp( log_sigma_sq ) ) )

def tf_normal_pdf(x, mu, sigma_sq):
    
    return ( tf.exp(-0.5 * tf.div( tf.square( tf.sub( x, mu ) ),  sigma_sq )) / tf.sqrt(2*np.pi * sigma_sq) )

def tf_stdnormal_logpdf(x):

    return ( - 0.5 * ( logc + tf.square( x ) ) )

def tf_gaussian_ent(log_sigma_sq):

    return ( - 0.5 * ( logc + 1.0 + log_sigma_sq ) )
 
def tf_gaussian_mixture_ent(mu_1, mu_2, log_sigma_sq_1, log_sigma_sq_2, zeta1, zeta2):
    Lp_temp_11 = zeta1 * (2*np.pi)**(-0.5) * (2 * tf.exp(log_sigma_sq_1))**(-0.5)

    Lp_temp_12 = zeta2 * tf.exp(-0.5 * tf.square((mu_1-mu_2)) / (tf.exp(log_sigma_sq_1) + tf.exp(log_sigma_sq_2))) * (2*np.pi)**(-0.5) * (tf.exp(log_sigma_sq_1) + tf.exp(log_sigma_sq_2))**(-0.5)
    
    Lp_temp_21 = zeta1 * tf.exp(-0.5 * tf.square((mu_2-mu_1)) / (tf.exp(log_sigma_sq_2) + tf.exp(log_sigma_sq_1))) * (2*np.pi)**(-0.5) * (tf.exp(log_sigma_sq_2) + tf.exp(log_sigma_sq_1))**(-0.5)

    Lp_temp_22 = zeta2 * (2*np.pi)**(-0.5) * (2 * tf.exp(log_sigma_sq_2))**(-0.5)
   
    Lp_temp = zeta1 * tf.log(Lp_temp_11 + Lp_temp_12) + zeta2 * tf.log(Lp_temp_21 + Lp_temp_22)
    
    return Lp_temp

def tf_gaussian_marg(mu, log_sigma_sq):

    return ( - 0.5 * ( logc + ( tf.square( mu ) + tf.exp( log_sigma_sq ) ) ) )

def tf_binary_xentropy(x, y, const = 1e-10):

    return - ( x * tf.log ( tf.clip_by_value( y, const, 1.0 ) ) + \
             (1.0 - x) * tf.log( tf.clip_by_value( 1.0 - y, const, 1.0 ) ) )

def feed_numpy_semisupervised(num_lab_batch, num_ulab_batch, x_lab, y, x_ulab):

    size = x_lab.shape[0] + x_ulab.shape[0]
    batch_size = num_lab_batch + num_ulab_batch
    count = int(size / batch_size)

    dim = x_lab.shape[1]

    for i in xrange(count):
        start_lab = i * num_lab_batch
        end_lab = start_lab + num_lab_batch
        start_ulab = i * num_ulab_batch
        end_ulab = start_ulab + num_ulab_batch

        yield [	x_lab[start_lab:end_lab,:dim/2],
                x_lab[start_lab:end_lab,dim/2:dim],
                y[start_lab:end_lab],
				x_ulab[start_ulab:end_ulab,:dim/2],
                 x_ulab[start_ulab:end_ulab,dim/2:dim] ]

def feed_numpy_semisupervised_multiview(num_lab_batch, num_ulab_batch, x1_lab, x2_lab, y, x1_ulab, x2_ulab):

    size = x1_lab.shape[0] + x1_ulab.shape[0]
    batch_size = num_lab_batch + num_ulab_batch
    count = int(size / batch_size)

    dim1 = x1_lab.shape[1]
    dim2 = x2_lab.shape[1]

    for i in xrange(count):
        start_lab = i * num_lab_batch
        end_lab = start_lab + num_lab_batch
        start_ulab = i * num_ulab_batch
        end_ulab = start_ulab + num_ulab_batch

        yield [	x1_lab[start_lab:end_lab,:dim1/2], 
                x2_lab[start_lab:end_lab,:dim2/2], 
                x1_lab[start_lab:end_lab,dim1/2:dim1], 
                x2_lab[start_lab:end_lab,dim2/2:dim2],
                y[start_lab:end_lab],
			    x1_ulab[start_ulab:end_ulab,:dim1/2], 
                x2_ulab[start_ulab:end_ulab,:dim2/2],
                x1_ulab[start_ulab:end_ulab,dim1/2:dim1], 
                x2_ulab[start_ulab:end_ulab,dim2/2:dim2] ]

def feed_numpy(batch_size, x):

    size = x.shape[0]
    count = int(size / batch_size)

    for i in xrange(count):
        start = i * batch_size
        end = start + batch_size

        yield x[start:end]

def feed_numpy_multiview(batch_size, x1, x2):
    size = x1.shape[0]
    count = int(size / batch_size)

    for i in xrange(count):
        start = i * batch_size
        end = start + batch_size

        yield x1[start:end], x2[start:end]

def print_metrics(epoch, *metrics):

    print(25*'-')
    for metric in metrics: 
        print('[{}] {} {}: {}'.format(epoch, metric[0],metric[1],metric[2]))
    print(25*'-')
    
def fetch_multiview_batch(batch_size, multiview_fea):
    x1 = multiview_fea[0]
    num_views = len(multiview_fea)
    size = x1.shape[0]
    count = int(size / batch_size)    
    for i in xrange(count):
        start = i * batch_size
        end = start + batch_size
        multiview_fea_batch = []
        for c in range(num_views):
            view = multiview_fea[c]
            view = view[start:end,:]
            multiview_fea_batch.append(view)
        yield multiview_fea_batch        