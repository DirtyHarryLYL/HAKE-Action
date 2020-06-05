from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.framework import ops

from ult.config import cfg
from ult.visualization import draw_bounding_boxes_HOI
from ult.obj_80_768_averg_matrix import obj_matrix
from ult.matrix_sentence_76 import sentence_only

import numpy as np

def resnet_arg_scope(is_training=True,
                     weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
    batch_norm_params = {
        'is_training': False,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'trainable': False,
        'updates_collections': ops.GraphKeys.UPDATE_OPS
    }
    with arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY),
        weights_initializer = slim.variance_scaling_initializer(),
        biases_regularizer  = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY), 
        biases_initializer  = tf.constant_initializer(0.0),
        trainable           = is_training,
        activation_fn       = tf.nn.relu,
        normalizer_fn       = slim.batch_norm,
        normalizer_params   = batch_norm_params):
        with arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc

class ResNet50(): 
    def __init__(self):
        self.visualize = {}
        self.intermediate = {}
        self.predictions = {}
        self.score_summaries = {}
        self.event_summaries = {}
        self.train_summaries = []
        self.losses = {}

        self.image       = tf.placeholder(tf.float32, shape=[1, None, None, 3], name = 'image') #scene stream
        self.H_boxes     = tf.placeholder(tf.float32, shape=[None, 5], name = 'H_boxes') # Human stream
        self.R_boxes     = tf.placeholder(tf.float32, shape=[None, 5], name = 'R_boxes') # PaSta stream
        self.O_boxes     = tf.placeholder(tf.float32, shape=[None, 5], name = 'O_boxes') # PaSta stream
        self.P_boxes     = tf.placeholder(tf.float32, shape=[None, 10, 5], name = 'P_boxes') # PaSta stream
        self.gt_class_P0 = tf.placeholder(tf.float32, shape=[None, 12], name = 'gt_class_p1s') # target ankle status
        self.gt_class_P1 = tf.placeholder(tf.float32, shape=[None, 10], name = 'gt_class_p2s') # target knee status
        self.gt_class_P2 = tf.placeholder(tf.float32, shape=[None, 5], name = 'gt_class_p3s') # target hip status
        self.gt_class_P3 = tf.placeholder(tf.float32, shape=[None, 31], name = 'gt_class_p4s') # target hand status
        self.gt_class_P4 = tf.placeholder(tf.float32, shape=[None, 5], name = 'gt_class_p5s') # target shoulder status
        self.gt_class_P5 = tf.placeholder(tf.float32, shape=[None, 13], name = 'gt_class_p6s') # target head status
        self.gt_verb     = tf.placeholder(tf.float32, shape=[None, 117], name = 'gt_class_verb') # target verb
        self.gt_10v      = tf.placeholder(tf.float32, shape=[None, 10], name = 'gt_class_vec') # target vec
        self.gt_class_HO = tf.placeholder(tf.float32, shape=[None, 600], name = 'gt_class_HO') # target HOI
        self.gt_object   = tf.placeholder(tf.float32, shape=[None, 80], name = 'gt_object') # object class
        self.H_num       = tf.placeholder(tf.int32)
        self.HO_weight   = np.array(
                [39.86157711904592, 42.40444304026247, 44.834823527125415, 39.73991804849558, 39.350051894572104, 43.56609116327042, 37.78352039671705, 37.27879696661561, 56.2961038839078, 27.187192997462514, 48.69942543701149, 29.31727479695975, 40.91791293317506, 41.48885009402292, 48.00306615559755, 50.012214583404685, 41.67212390491823, 42.921511271001236, 28.880648716145704, 29.52003435670287, 28.7948786160738, 39.76397874615436, 59.30640384054761, 16.891810699279997, 45.78457865943398, 38.21737256387448, 41.27866658762785, 55.327003753827235, 50.27550397062817, 35.892058594766205, 23.930101537298007, 42.1883315501357, 39.01256606369551, 46.40605772692243, 41.63484517972581, 51.90277694560517, 39.86157711904592, 44.3232983026516, 26.24108136535154, 33.525745002186696, 36.51886783101932, 28.541819963426093, 30.205498384606926, 45.882177032325544, 59.30640384054761, 19.277187457752916, 40.44149658882279, 40.27550397062818, 34.99276619895774, 45.24100203620805, 53.86572339704485, 50.012214583404685, 48.69942543701149, 17.603639967967627, 38.389734264590764, 54.535191293350984, 35.660893887007894, 47.692723818197855, 48.00306615559755, 43.80412030999668, 30.501126058559556, 32.35158707564564, 54.535191293350984, 62.31670379718742, 23.78093836099101, 48.00306615559755, 62.31670379718742, 36.67004315466653, 44.535191293350984, 42.58542526119043, 62.31670379718742, 48.892476988965356, 48.16697031747924, 34.600828988374865, 39.88632331032448, 14.392366637557455, 62.31670379718742, 56.2961038839078, 40.3301329276432, 43.50856787437951, 54.535191293350984, 43.86572339704485, 45.414742996902284, 57.545491249990796, 62.31670379718742, 34.76558113323671, 45.15667036083943, 41.8245235704856, 43.285803927267985, 31.29236674037406, 62.31670379718742, 13.477875959525374, 47.84512348376523, 38.73735532718288, 32.71199602184443, 27.673298950910752, 52.31670379718742, 38.425042953542096, 42.49399146679174, 55.327003753827235, 55.327003753827235, 45.15667036083943, 47.26520401398836, 45.414742996902284, 54.535191293350984, 41.27866658762785, 26.897169052605058, 55.327003753827235, 28.62825872892921, 27.253006626232384, 19.566586419407308, 42.44898645452497, 59.30640384054761, 51.52489133671117, 51.90277694560517, 37.18452779650803, 48.16697031747924, 39.86157711904592, 49.76397874615436, 49.30640384054761, 40.22155365176111, 49.30640384054761, 43.12592287342668, 50.27550397062817, 45.327003753827235, 41.244604100708735, 48.33730371046705, 57.545491249990796, 25.50972350021107, 48.69942543701149, 46.753678789514545, 33.13115849168469, 48.16697031747924, 40.824512670633624, 51.90277694560517, 62.31670379718742, 53.285803927267985, 42.97171928475174, 38.794878616073795, 30.481158460998806, 35.24100203620806, 31.561234183262115, 42.8228037307383, 42.31670379718742, 48.16697031747924, 21.951612511860127, 32.40887687915604, 43.92821288981487, 41.38248694556508, 57.545491249990796, 42.06364514453972, 51.52489133671117, 35.81362856586806, 30.962196803732283, 30.042979374291058, 30.29727316317092, 38.200506737555116, 50.55579120663061, 57.545491249990796, 19.54955885088451, 42.921511271001236, 49.529167787659134, 35.17340619973509, 38.13369088398997, 44.39278690220489, 55.327003753827235, 62.31670379718742, 40.55579120663061, 53.86572339704485, 2.660919516910389, 52.31670379718742, 52.31670379718742, 52.77427870279417, 22.931506545422504, 49.09451084984823, 42.631874311648076, 34.750342714728944, 39.88632331032448, 47.26520401398836, 59.30640384054761, 48.16697031747924, 62.31670379718742, 41.863474009320846, 41.21080669419493, 55.327003753827235, 25.14999404158607, 41.863474009320846, 41.59788372412617, 56.2961038839078, 59.30640384054761, 34.66001824959728, 37.45948953237162, 53.285803927267985, 28.543641286505434, 47.4030868588447, 57.545491249990796, 38.794878616073795, 26.440716499974968, 53.86572339704485, 49.76397874615436, 45.327003753827235, 38.83365516670581, 47.692723818197855, 48.33730371046705, 47.84512348376523, 59.30640384054761, 59.30640384054761, 27.72277891959511, 32.56698385420673, 43.285803927267985, 49.30640384054761, 50.27550397062817, 32.22644637631832, 19.113163469010704, 52.77427870279417, 37.17122627058457, 56.2961038839078, 41.67212390491823, 37.292432597343094, 39.26319010272118, 51.90277694560517, 48.16697031747924, 57.545491249990796, 25.21383732015851, 45.327003753827235, 43.12592287342668, 33.7494148933586, 62.31670379718742, 37.05331102328898, 59.30640384054761, 34.37182333059573, 24.936036649412728, 39.350051894572104, 39.32817303309036, 24.636245656163254, 49.09451084984823, 44.46340544707976, 42.31670379718742, 40.793820353356864, 59.30640384054761, 37.157965360070634, 40.44149658882279, 32.457950224103485, 51.17727027411905, 42.921511271001236, 29.26534060775103, 20.70871826180388, 41.27866658762785, 33.15743168021626, 34.81934064149681, 33.8906114010818, 19.271367693122244, 43.56609116327042, 47.00191462676487, 57.545491249990796, 59.30640384054761, 19.587154348705695, 62.31670379718742, 45.78457865943398, 45.78457865943398, 62.31670379718742, 59.30640384054761, 62.31670379718742, 34.842585718323186, 37.2516534731387, 42.23070207956824, 37.51663436761591, 43.56609116327042, 40.763343422536806, 47.545491249990796, 51.90277694560517, 43.50856787437951, 25.56809238980931, 42.23070207956824, 56.2961038839078, 48.16697031747924, 46.51886783101932, 39.37204153557148, 48.892476988965356, 62.31670379718742, 62.31670379718742, 57.545491249990796, 30.025006771796413, 39.692192899883125, 35.00888104052353, 39.57512530455062, 62.31670379718742, 38.606025174470055, 41.143990840629776, 23.637317288279576, 43.92821288981487, 45.882177032325544, 54.535191293350984, 51.90277694560517, 27.024967764570196, 40.88655579464647, 41.70972539365131, 37.375157857002996, 50.27550397062817, 50.55579120663061, 43.231853608400925, 46.876023353684666, 45.24100203620805, 62.31670379718742, 16.424023402560195, 52.31670379718742, 44.68242386155804, 41.41765268279344, 37.95507732677986, 45.882177032325544, 43.12592287342668, 62.31670379718742, 27.177871941076496, 46.08421089320842, 42.02286602033532, 59.30640384054761, 51.17727027411905, 59.30640384054761, 50.85542344040504, 43.340432884283004, 35.679694543290935, 39.46113070710968, 44.99276619895774, 34.09502300350725, 28.227523588719624, 62.31670379718742, 47.84512348376523, 50.55579120663061, 52.77427870279417, 32.15891623329701, 40.85542344040504, 41.863474009320846, 47.692723818197855, 54.535191293350984, 59.30640384054761, 25.613317385913, 42.44898645452497, 37.22467857387639, 35.8624811036965, 50.012214583404685, 38.532724787706044, 25.93880550356513, 40.19482775314784, 51.52489133671117, 48.69942543701149, 59.30640384054761, 36.15720328062341, 33.27496011434579, 45.414742996902284, 45.327003753827235, 59.30640384054761, 48.514591380071366, 43.50856787437951, 40.38545781364281, 54.535191293350984, 17.279205413234312, 40.67317523934305, 41.67212390491823, 62.31670379718742, 42.10481080648804, 38.87278106033631, 45.327003753827235, 30.602364787757338, 45.882177032325544, 57.545491249990796, 45.50429142343155, 38.932138861141375, 30.87719263294779, 45.982019241391555, 38.91226264878624, 33.71931813521595, 35.00081614532003, 41.8245235704856, 40.498267917739696, 35.8624811036965, 20.759093668408187, 39.09451084984823, 34.73515757751352, 46.2961038839078, 62.31670379718742, 34.66001824959728, 55.327003753827235, 23.89185955307172, 38.73735532718288, 46.753678789514545, 34.37879995027924, 50.85542344040504, 38.407352726153626, 25.571766624223923, 62.31670379718742, 55.327003753827235, 53.86572339704485, 37.2516534731387, 45.59572521783024, 44.834823527125415, 62.31670379718742, 33.41249360917828, 62.31670379718742, 57.545491249990796, 57.545491249990796, 46.63468655651747, 62.31670379718742, 62.31670379718742, 57.545491249990796, 53.86572339704485, 59.30640384054761, 28.45349805824696, 56.2961038839078, 51.17727027411905, 43.231853608400925, 55.327003753827235, 49.76397874615436, 48.69942543701149, 33.27496011434579, 46.753678789514545, 48.514591380071366, 62.31670379718742, 31.942438817781188, 62.31670379718742, 43.12592287342668, 46.753678789514545, 45.414742996902284, 41.38248694556508, 39.668525567092054, 44.60818368076598, 45.15667036083943, 55.327003753827235, 62.31670379718742, 26.128903552125276, 59.30640384054761, 46.51886783101932, 56.2961038839078, 48.892476988965356, 33.22114350477567, 39.621574355008256, 44.39278690220489, 57.545491249990796, 29.10279101407053, 47.545491249990796, 62.31670379718742, 57.545491249990796, 41.52489133671117, 39.37204153557148, 51.52489133671117, 31.147307331679862, 40.61408664323785, 38.389734264590764, 42.8228037307383, 35.92183890450156, 59.30640384054761, 51.52489133671117, 52.77427870279417, 33.01740819634154, 40.94949812562335, 35.93181122764105, 38.200506737555116, 36.188865229990064, 32.39558891931792, 44.68242386155804, 30.061026662792713, 48.892476988965356, 30.590674485088822, 21.711991869210635, 55.327003753827235, 44.12126444176874, 42.77427870279417, 49.09451084984823, 41.38248694556508, 47.692723818197855, 55.327003753827235, 33.95979808226316, 32.813055253426185, 32.26060934358462, 24.087181391712605, 55.327003753827235, 46.51886783101932, 44.46340544707976, 40.35770727309509, 36.814420266636475, 34.316410204746084, 34.1808939115055, 33.73735114999313, 27.336356560317153, 38.442805533800126, 42.631874311648076, 62.31670379718742, 47.26520401398836, 25.597573672771553, 44.535191293350984, 39.78817348738849, 42.72628987397649, 41.143990840629776, 38.62454522308599, 37.0147068151566, 41.863474009320846, 37.55999191394312, 51.90277694560517, 43.99161467012506, 62.31670379718742, 62.31670379718742, 40.793820353356864, 22.874345359252622, 48.892476988965356, 52.31670379718742, 62.31670379718742, 32.133860712922115, 38.68058399826598, 41.70972539365131, 38.5509342266223, 62.31670379718742, 52.31670379718742, 51.90277694560517, 47.545491249990796, 47.545491249990796, 62.31670379718742, 24.51353067578591, 38.73735532718288, 54.535191293350984, 37.347407316455275, 50.55579120663061, 57.545491249990796, 47.692723818197855, 59.30640384054761, 38.87278106033631, 40.44149658882279, 48.33730371046705, 62.31670379718742, 26.219692773393426, 45.59572521783024, 37.90761297653525, 44.834823527125415, 57.545491249990796, 29.810064602554988, 39.30640384054761, 38.46064106120429, 62.31670379718742, 43.02251454004449, 28.545463373722857, 44.75795524046251, 62.31670379718742, 43.12592287342668, 51.17727027411905, 47.545491249990796, 47.84512348376523, 33.17327222599302, 18.763338182063613, 59.30640384054761, 59.30640384054761, 62.31670379718742, 46.51886783101932, 53.86572339704485, 62.31670379718742, 57.545491249990796, 50.55579120663061, 44.834823527125415, 57.545491249990796, 55.327003753827235, 35.577283810846545, 38.661823948278425, 37.26520401398836, 62.31670379718742, 31.974431189481912, 51.17727027411905, 50.012214583404685, 49.09451084984823, 48.514591380071366, 21.94044709804023, 48.514591380071366, 40.643530449705665, 45.414742996902284, 41.902776945605176, 49.30640384054761, 37.87625583800666, 39.37204153557148, 50.85542344040504, 21.85956320777875, 31.97844685765432, 30.622898844067926, 54.535191293350984, 50.85542344040504, 52.77427870279417, 52.77427870279417, 31.27524829164734, 18.933732894860512, 50.55579120663061, 50.012214583404685, 62.31670379718742, 28.163630874931748, 47.84512348376523, 36.178285578426724, 45.59572521783024, 43.50856787437951, 62.31670379718742, 62.31670379718742, 19.859850154855067, 53.285803927267985, 59.30640384054761, 56.2961038839078, 43.50856787437951, 41.67212390491823]
            , dtype = 'float32').reshape(1,600) # HOI loss weight
        self.transfer_mask_1 = np.ones((1, 600), dtype = 'float32')

        self.pasta0_weight   = np.array([ 0.061241616 , 0.33891392 , 0.92476034 , 5.8682323 , 23.186672 , 17.28461 , 3.0083976 , 16.112774 , 21.125637 , 0.4825653 , 11.593336 , 0.0128623145], dtype='float32').reshape(1, -1)
        self.pasta1_weight   = np.array([ 1.1284876 , 7.0736904 , 28.294762 , 21.092459 , 0.58887583 , 1.6362275 , 0.24495044 , 25.779675 , 14.147381 , 0.013494855], dtype='float32').reshape(1, -1)
        self.pasta2_weight   = np.array([ 2.3774152 , 9.6559 , 9.986509 , 77.28022 , 0.6999586], dtype='float32').reshape(1, -1)
        self.pasta3_weight   = np.array([ 0.003327214 , 0.43338323 , 0.064010605 , 0.05624761 , 0.23240773 , 1.5325437 , 0.3501507 , 0.6822291 , 0.60773283 , 6.6090946 , 1.6785003 , 15.106503 , 10.574552 , 0.15573713 , 1.7335329 , 0.80110234 , 1.0266554 , 0.40360883 , 0.15997808 , 0.19618833 , 0.3659014 , 0.3596786 , 0.39165002 , 0.15573713 , 0.51084787 , 52.872757 , 0.175657 , 0.2360391 , 1.5782912 , 0.9441564 , 0.0018035152], dtype='float32').reshape(1, -1)
        self.pasta4_weight   = np.array([ 13.624015 , 24.18162 , 32.617817 , 29.30141 , 0.27514917], dtype='float32').reshape(1, -1)
        self.pasta5_weight   = np.array([ 0.08787887 , 0.009932528 , 94.55767 , 1.5009155 , 0.09127188 , 0.8081853 , 0.06847043 , 0.8081853 , 0.28653836 , 0.2781108 , 1.390554 , 0.11111359 , 0.0011817936], dtype='float32').reshape(1, -1)
        self.pasta_matrix  = np.array(sentence_only, dtype='float32')[:,:1536]
        self.obj_matrix  = np.array(obj_matrix, dtype='float32')
        self.num_classes = 600 # HOI
        self.num_pasta0    = 12 # pasta0 ankle
        self.num_pasta1    = 10 # pasta1 knee
        self.num_pasta2    = 5 # pasta2 hip
        self.num_pasta3    = 31 # pasta3 hand
        self.num_pasta4    = 5 # pasta4 shoulder
        self.num_pasta5    = 13 # pasta5 head
        self.num_verbs   = 117
        self.num_vec     = 10
        self.num_fc      = 1024
        self.scope       = 'resnet_v1_50'
        self.stride      = [16, ]
        self.lr          = tf.placeholder(tf.float32)
        if tf.__version__ == '1.1.0':
            self.blocks     = [resnet_utils.Block('block1', resnet_v1.bottleneck,[(256,   64, 1)] * 2 + [(256,   64, 2)]),
                               resnet_utils.Block('block2', resnet_v1.bottleneck,[(512,  128, 1)] * 3 + [(512,  128, 2)]),
                               resnet_utils.Block('block3', resnet_v1.bottleneck,[(1024, 256, 1)] * 5 + [(1024, 256, 1)]),
                               resnet_utils.Block('block4', resnet_v1.bottleneck,[(2048, 512, 1)] * 3),
                               resnet_utils.Block('block5', resnet_v1.bottleneck,[(2048, 512, 1)] * 3)]
        else: # we use tf 1.2.0 here, Resnet-50
            from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block
            self.blocks = [resnet_v1_block('block1', base_depth=64,  num_units=3, stride=2),
                           resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
                           resnet_v1_block('block3', base_depth=256, num_units=6, stride=1),
                           resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
                           resnet_v1_block('block5', base_depth=512, num_units=3, stride=1)]

    def build_base(self):
        with tf.variable_scope(self.scope, self.scope):
            net = resnet_utils.conv2d_same(self.image, 64, 7, stride=2, scope='conv1') 
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool1')
        return net

    def image_to_head(self, is_training):
        with slim.arg_scope(resnet_arg_scope(is_training=False)):
            net    = self.build_base()
            net, _ = resnet_v1.resnet_v1(net,
                                         self.blocks[0:cfg.RESNET.FIXED_BLOCKS], 
                                         global_pool=False,
                                         include_root_block=False,
                                         scope=self.scope)
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
            head, _ = resnet_v1.resnet_v1(net,
                                          self.blocks[cfg.RESNET.FIXED_BLOCKS:-2], 
                                          global_pool=False,
                                          include_root_block=False,
                                          scope=self.scope)
        return head

    def res5(self, pool5_H, pool5_R, is_training, name):
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):

            pool5_H, _ = resnet_v1.resnet_v1(pool5_H, 
                                           self.blocks[-2:-1], 
                                           global_pool=False,
                                           include_root_block=False,
                                           reuse=False,
                                           scope=self.scope)

            fc5_H = tf.reduce_mean(pool5_H, axis=[1, 2])

            pool5_R, _ = resnet_v1.resnet_v1(pool5_R,
                                       self.blocks[-1:], 
                                       global_pool=False,
                                       include_root_block=False,
                                       reuse=False,
                                       scope=self.scope)

            fc5_R = tf.reduce_mean(pool5_R, axis=[1, 2])
        
        return fc5_H, fc5_R

    def crop_pool_layer(self, bottom, rois, name):
        with tf.variable_scope(name) as scope:

            batch_ids    = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
            bottom_shape = tf.shape(bottom)
            height       = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self.stride[0])
            width        = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self.stride[0])
            x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
            y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
            x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
            y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height

            bboxes        = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
            if cfg.RESNET.MAX_POOL:
                pre_pool_size = cfg.POOLING_SIZE * 2
                crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size], name="crops")
                crops = slim.max_pool2d(crops, [2, 2], padding='SAME')
            else:
                crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [cfg.POOLING_SIZE, cfg.POOLING_SIZE], name="crops")
        return crops

    def ROI_for_parts(self, head, fc5_H, fc5_R, fc5_S, P_boxes, name):
        with tf.variable_scope(name) as scope:
            pool5_P0 = tf.reduce_mean(self.crop_pool_layer(head, P_boxes[:, 0, :], 'crop_P0'), axis=[1, 2]) # RAnk
            pool5_P1 = tf.reduce_mean(self.crop_pool_layer(head, P_boxes[:, 1, :], 'crop_P1'), axis=[1, 2]) # RKnee
            pool5_P2 = tf.reduce_mean(self.crop_pool_layer(head, P_boxes[:, 2, :], 'crop_P2'), axis=[1, 2]) # LKnee
            pool5_P3 = tf.reduce_mean(self.crop_pool_layer(head, P_boxes[:, 3, :], 'crop_P3'), axis=[1, 2]) # LAnk
            pool5_P4 = tf.reduce_mean(self.crop_pool_layer(head, P_boxes[:, 4, :], 'crop_P4'), axis=[1, 2]) # Hip
            pool5_P5 = tf.reduce_mean(self.crop_pool_layer(head, P_boxes[:, 5, :], 'crop_P5'), axis=[1, 2]) # Head
            pool5_P6 = tf.reduce_mean(self.crop_pool_layer(head, P_boxes[:, 6, :], 'crop_P6'), axis=[1, 2]) # RHand
            pool5_P7 = tf.reduce_mean(self.crop_pool_layer(head, P_boxes[:, 7, :], 'crop_P7'), axis=[1, 2]) # RSho
            pool5_P8 = tf.reduce_mean(self.crop_pool_layer(head, P_boxes[:, 8, :], 'crop_P8'), axis=[1, 2]) # LSho
            pool5_P9 = tf.reduce_mean(self.crop_pool_layer(head, P_boxes[:, 9, :], 'crop_P9'), axis=[1, 2]) # LHand
            
            fc5_S    = tf.tile(fc5_S, [tf.shape(pool5_P0)[0], 1])
            fc5_P0 = tf.concat([pool5_P0, pool5_P3, fc5_H, fc5_R, fc5_S], axis=1)
            fc5_P1 = tf.concat([pool5_P1, pool5_P2, fc5_H, fc5_R, fc5_S], axis=1)
            fc5_P2 = tf.concat([pool5_P4, fc5_H, fc5_R, fc5_S], axis=1)
            fc5_P3 = tf.concat([pool5_P6, pool5_P9, fc5_H, fc5_R, fc5_S], axis=1)
            fc5_P4 = tf.concat([pool5_P7, pool5_P8, fc5_H, fc5_R, fc5_S], axis=1)
            fc5_P5 = tf.concat([pool5_P5, fc5_H, fc5_R, fc5_S], axis=1)
            
        return fc5_P0, fc5_P1, fc5_P2, fc5_P3, fc5_P4, fc5_P5

    def part_classification(self, pool5_P, is_training, initializer, num_state, name):
        with tf.variable_scope(name) as scope:
            fc6_P    = slim.fully_connected(pool5_P, 512)
            fc6_P    = slim.dropout(fc6_P, keep_prob=0.5, is_training=is_training)
            fc7_P    = slim.fully_connected(fc6_P, 512)
            fc7_P    = slim.dropout(fc7_P, keep_prob=0.5, is_training=is_training)
            cls_score_P  = slim.fully_connected(fc7_P, num_state, 
                                               weights_initializer=initializer,
                                               trainable=is_training,
                                               activation_fn=None)
            cls_prob_P   = tf.nn.sigmoid(cls_score_P) 
            tf.reshape(cls_prob_P, [1, num_state])

        return cls_score_P, cls_prob_P, fc7_P

    def pasta_classification(self, pool5_P0, pool5_P1, pool5_P2, pool5_P3, pool5_P4, pool5_P5, is_training, initializer, name):
        with tf.variable_scope(name) as scope:
            cls_score_P0, cls_prob_P0, fc7_P0 = self.part_classification(pool5_P0, is_training, initializer, self.num_pasta0, 'cls_pasta_0')
            cls_score_P1, cls_prob_P1, fc7_P1 = self.part_classification(pool5_P1, is_training, initializer, self.num_pasta1, 'cls_pasta_1')
            cls_score_P2, cls_prob_P2, fc7_P2 = self.part_classification(pool5_P2, is_training, initializer, self.num_pasta2, 'cls_pasta_2')
            cls_score_P3, cls_prob_P3, fc7_P3 = self.part_classification(pool5_P3, is_training, initializer, self.num_pasta3, 'cls_pasta_3')
            cls_score_P4, cls_prob_P4, fc7_P4 = self.part_classification(pool5_P4, is_training, initializer, self.num_pasta4, 'cls_pasta_4')
            cls_score_P5, cls_prob_P5, fc7_P5 = self.part_classification(pool5_P5, is_training, initializer, self.num_pasta5, 'cls_pasta_5')
            
            self.predictions["cls_score_pasta0"]  = cls_score_P0
            self.predictions["cls_score_pasta1"]  = cls_score_P1
            self.predictions["cls_score_pasta2"]  = cls_score_P2
            self.predictions["cls_score_pasta3"]  = cls_score_P3
            self.predictions["cls_score_pasta4"]  = cls_score_P4
            self.predictions["cls_score_pasta5"]  = cls_score_P5
            cls_prob_PaSta = tf.concat([cls_prob_P0, cls_prob_P1, cls_prob_P2, cls_prob_P3, cls_prob_P4, cls_prob_P5], axis=1)
            self.predictions["cls_prob_PaSta"]    = cls_prob_PaSta
            self.predictions['cls_score_PaSta']   = tf.concat([cls_score_P0, cls_score_P1, cls_score_P2, cls_score_P3, cls_score_P4, cls_score_P5], axis=1)
            
        return fc7_P0, fc7_P1, fc7_P2, fc7_P3, fc7_P4, fc7_P5

    def language_head(self):

        split_pos = [self.num_pasta0, self.num_pasta1, self.num_pasta2, self.num_pasta3, self.num_pasta4, self.num_pasta5]
        split_pos = np.array(split_pos).cumsum(axis=0)[:-1]
        pasta_P0, pasta_P1, pasta_P2, pasta_P3, pasta_P4, pasta_P5 = np.split(self.pasta_matrix, split_pos, axis=0)
        fc7_L = tf.concat([
            tf.matmul(self.predictions["cls_prob_PaSta"][:, :split_pos[0]], pasta_P0),
            tf.matmul(self.predictions["cls_prob_PaSta"][:, split_pos[0]:split_pos[1]], pasta_P1),
            tf.matmul(self.predictions["cls_prob_PaSta"][:, split_pos[1]:split_pos[2]], pasta_P2),
            tf.matmul(self.predictions["cls_prob_PaSta"][:, split_pos[2]:split_pos[3]], pasta_P3),
            tf.matmul(self.predictions["cls_prob_PaSta"][:, split_pos[3]:split_pos[4]], pasta_P4),
            tf.matmul(self.predictions["cls_prob_PaSta"][:, split_pos[4]:], pasta_P5),
        ], axis=1)

        obj_L = tf.matmul(self.gt_object, self.obj_matrix)
        fc7_L = tf.concat([fc7_L, obj_L], axis=1)

        return fc7_L

    def verb_classification(self, fc7_P, is_training, initializer, name):
        with tf.variable_scope(name) as scope:
            cls_score_verb = slim.fully_connected(fc7_P, self.num_verbs, 
                                               weights_initializer=initializer,
                                               trainable=is_training,
                                               activation_fn=None)
            cls_prob_verb   = tf.nn.sigmoid(cls_score_verb) 
            tf.reshape(cls_prob_verb, [1, self.num_verbs])
        
        self.predictions["cls_score_verb"]   = cls_score_verb
        self.predictions["cls_prob_verb"]   = cls_prob_verb
        return     

    def vec_classification(self, fc7_P, is_training, initializer, name):
        with tf.variable_scope(name) as scope:
            fc8_vec       = slim.fully_connected(fc7_P, 1024, weights_initializer=initializer, trainable=is_training)
            fc8_vec       = slim.dropout(fc8_vec, keep_prob=0.5, is_training=is_training)
            fc9_vec       = slim.fully_connected(fc8_vec, 1024, weights_initializer=initializer, trainable=is_training)
            fc9_vec       = slim.dropout(fc9_vec, keep_prob=0.5, is_training=is_training)
            cls_score_vec = slim.fully_connected(fc9_vec, self.num_vec, 
                                               weights_initializer=initializer,
                                               trainable=is_training,
                                               activation_fn=None)
            cls_prob_vec   = tf.nn.sigmoid(cls_score_vec) 
            tf.reshape(cls_prob_vec, [1, self.num_vec])
        
        self.predictions["cls_score_vec"]  = cls_score_vec
        self.predictions["cls_prob_vec"]   = cls_prob_vec
        return

    def vec_attention(self, cls_prob_vec, fc5_P0, fc5_P1, fc5_P2, fc5_P3, fc5_P4, fc5_P5, fc5_O, initializer, is_training, name):
        with tf.variable_scope(name) as scope:
            fc5_P0 = tf.multiply(fc5_P0, (cls_prob_vec[:, 0:1]+cls_prob_vec[:, 3:4])/2)
            fc5_P1 = tf.multiply(fc5_P1, (cls_prob_vec[:, 1:2]+cls_prob_vec[:, 2:3])/2)
            fc5_P2 = tf.multiply(fc5_P2, cls_prob_vec[:, 4:5])
            fc5_P3 = tf.multiply(fc5_P3, (cls_prob_vec[:, 6:7]+cls_prob_vec[:, 9:10])/2)
            fc5_P4 = tf.multiply(fc5_P4, (cls_prob_vec[:, 7:8]+cls_prob_vec[:, 8:9])/2)
            fc5_P5 = tf.multiply(fc5_P5, cls_prob_vec[:, 5:6])
            
            fc5_P_att = tf.concat([fc5_P0, fc5_P1, fc5_P2, fc5_P3, fc5_P4, fc5_P5, fc5_O], axis=1)
            
            fc7_P_att       = slim.fully_connected(fc5_P_att, 4096, weights_initializer=initializer, trainable=is_training)
            fc7_P_att       = slim.dropout(fc7_P_att, keep_prob=0.5, is_training=is_training)

        return fc7_P_att
    
    def P_classification(self, fc7_P, is_training, initializer, name):
        with tf.variable_scope(name) as scope:

            cls_score_P = slim.fully_connected(fc7_P, self.num_classes, 
                                               weights_initializer=initializer,
                                               trainable=is_training,
                                               activation_fn=None, scope='cls_score_P')
            cls_prob_P  = tf.nn.sigmoid(cls_score_P, name='cls_prob_P') 
            tf.reshape(cls_prob_P, [1, self.num_classes]) 
            self.predictions["cls_score_P"] = cls_score_P
            self.predictions["cls_prob_P"]  = cls_prob_P

        return

    def A_classification(self, fc7_P, is_training, initializer, name):
        with tf.variable_scope(name) as scope:

            cls_score_P = slim.fully_connected(fc7_P, self.num_classes, 
                                               weights_initializer=initializer,
                                               trainable=is_training,
                                               activation_fn=None, scope='cls_score_A')
            cls_prob_P  = tf.nn.sigmoid(cls_score_P, name='cls_prob_A') 
            tf.reshape(cls_prob_P, [1, self.num_classes]) 
            self.predictions["cls_score_A"] = cls_score_P
            self.predictions["cls_prob_A"]  = cls_prob_P

        return

    def L_classification(self, fc7_L, is_training, initializer, name):
        with tf.variable_scope(name) as scope:
            
            fc8_L    = slim.fully_connected(fc7_L, 4096, 
                                            weights_initializer=initializer, trainable=is_training)
            fc8_L    = slim.dropout(fc8_L, keep_prob=0.5, is_training=is_training)
                
            cls_score_L = slim.fully_connected(fc8_L, self.num_classes, 
                                            weights_initializer=initializer,
                                            trainable=is_training,
                                            activation_fn=None, scope='cls_score_L')

            cls_prob_L  = tf.nn.sigmoid(cls_score_L, name='cls_prob_L') 
            tf.reshape(cls_prob_L, [1, self.num_classes]) 
            
            self.predictions["cls_score_L"] = cls_score_L
            self.predictions["cls_prob_L"]  = cls_prob_L

        return

    def build_network(self, is_training):
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)

        head       = self.image_to_head(is_training) 
        pool5_H    = self.crop_pool_layer(head, self.H_boxes, 'Crop_H') 
        pool5_O    = self.crop_pool_layer(head, self.O_boxes, 'Crop_O') 
        pool5_R    = self.crop_pool_layer(head, self.R_boxes, 'Crop_R') 

        fc5_H, fc5_R = self.res5(pool5_H, pool5_R, is_training, name='res5_HR')
        fc5_O = tf.reduce_mean(pool5_O, axis=[1, 2])
        fc5_S = tf.reduce_mean(head, axis=[1, 2])
        fc5_P0, fc5_P1, fc5_P2, fc5_P3, fc5_P4, fc5_P5 = self.ROI_for_parts(head, fc5_H, fc5_R, fc5_S, self.P_boxes, 'ROI_for_parts')

        fc7_P0, fc7_P1, fc7_P2, fc7_P3, fc7_P4, fc7_P5 = self.pasta_classification(fc5_P0, fc5_P1, fc5_P2, fc5_P3, fc5_P4, fc5_P5, is_training, initializer, 'pasta_classification')
        fc7_P = tf.concat([fc7_P0, fc7_P1, fc7_P2, fc7_P3, fc7_P4, fc7_P5], axis=1)
        
        self.vec_classification(fc7_P, is_training, initializer, 'vec_classification') 
        
        fc7_P_att = self.vec_attention(self.predictions['cls_prob_vec'], fc7_P0, fc7_P1, fc7_P2, fc7_P3, fc7_P4, fc7_P5, fc5_O, initializer, is_training, name='vec_attention')
        
        fc7_L = self.language_head()
        
        self.verb_classification(fc7_P, is_training, initializer, 'verb_classification') 

        self.P_classification(fc7_P, is_training, initializer, 'region_classification') 
        self.A_classification(fc7_P_att, is_training, initializer, 'region_classification') 
        self.L_classification(fc7_L, is_training, initializer, 'region_classification') 

        self.score_summaries.update(self.predictions)
        return

    def create_architecture(self, is_training):

        self.build_network(is_training)

        for var in tf.trainable_variables():
            self.train_summaries.append(var)

        self.add_loss()
        layers_to_output = {}
        layers_to_output.update(self.losses)

        val_summaries = []
        with tf.device("/cpu:0"):
            for key, var in self.event_summaries.items():
                val_summaries.append(tf.summary.scalar(key, var))
        
        val_summaries.append(tf.summary.scalar('lr', self.lr))
        self.summary_op     = tf.summary.merge_all()
        self.summary_op_val = tf.summary.merge(val_summaries)

        return layers_to_output

    def add_loss(self):

        with tf.variable_scope('LOSS') as scope:
            cls_score_P = self.predictions["cls_score_P"]
            cls_score_A = self.predictions["cls_score_A"]
            cls_score_L = self.predictions["cls_score_L"]
            cls_score_P0 = self.predictions["cls_score_pasta0"]
            cls_score_P1 = self.predictions["cls_score_pasta1"]
            cls_score_P2 = self.predictions["cls_score_pasta2"]
            cls_score_P3 = self.predictions["cls_score_pasta3"]
            cls_score_P4 = self.predictions["cls_score_pasta4"]
            cls_score_P5 = self.predictions["cls_score_pasta5"]
            cls_score_verb = self.predictions['cls_score_verb']
            cls_score_vec = self.predictions['cls_score_vec']

            label_HO     = self.gt_class_HO
            label_P0     = self.gt_class_P0
            label_P1     = self.gt_class_P1
            label_P2     = self.gt_class_P2
            label_P3     = self.gt_class_P3
            label_P4     = self.gt_class_P4
            label_P5     = self.gt_class_P5
            label_verb   = self.gt_verb
            label_vec    = self.gt_10v

            ### the HOI contained by gt, give the wts, the others are given wts 1, to enhance the corresponding HOIs' loss
            self.transfer_1_HO = tf.multiply(self.HO_weight, label_HO) # --> [wts 0 wts 0 0], label is [0 or 1]
            self.transfer_2_HO = tf.subtract(self.transfer_mask_1, label_HO) # --> [1 1 1 1 1] - [1 0 1 0 0] = [0 1 0 1 1]
            self.transfer_3_HO = tf.add(self.transfer_1_HO, self.transfer_2_HO) # --> [wts 0 wts 0 0] + [0 1 0 1 1] = [wts 1 wts 1 1], then * loss, element-wise

            # pasta loss
            P0_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=label_P0[:self.H_num, :], logits=cls_score_P0[:self.H_num, :])
            P1_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=label_P1[:self.H_num, :], logits=cls_score_P1[:self.H_num, :])
            P2_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=label_P2[:self.H_num, :], logits=cls_score_P2[:self.H_num, :])
            P3_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=label_P3[:self.H_num, :], logits=cls_score_P3[:self.H_num, :])
            P4_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=label_P4[:self.H_num, :], logits=cls_score_P4[:self.H_num, :])
            P5_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=label_P5[:self.H_num, :], logits=cls_score_P5[:self.H_num, :])
            P0_cross_entropy = tf.reduce_mean(tf.multiply(P0_cross_entropy, self.pasta0_weight))
            P1_cross_entropy = tf.reduce_mean(tf.multiply(P1_cross_entropy, self.pasta1_weight))
            P2_cross_entropy = tf.reduce_mean(tf.multiply(P2_cross_entropy, self.pasta2_weight))
            P3_cross_entropy = tf.reduce_mean(tf.multiply(P3_cross_entropy, self.pasta3_weight))
            P4_cross_entropy = tf.reduce_mean(tf.multiply(P4_cross_entropy, self.pasta4_weight))
            P5_cross_entropy = tf.reduce_mean(tf.multiply(P5_cross_entropy, self.pasta5_weight))
            PaSta_cross_entropy = P0_cross_entropy + P1_cross_entropy + P2_cross_entropy + P3_cross_entropy + P4_cross_entropy + P5_cross_entropy

            # 10v loss and verb loss
            vec_cross_entropy  = tf.nn.sigmoid_cross_entropy_with_logits(labels=label_vec, logits=cls_score_vec)
            verb_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=label_verb, logits=cls_score_verb)
            vec_cross_entropy  = tf.reduce_mean(vec_cross_entropy)
            verb_cross_entropy  = tf.reduce_mean(verb_cross_entropy)

            # P stream
            P_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=label_HO, logits=cls_score_P)
            P_cross_entropy = tf.reduce_mean(tf.multiply(P_cross_entropy, self.transfer_3_HO)) #self.HO_weight))
            A_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=label_HO, logits=cls_score_A)
            A_cross_entropy = tf.reduce_mean(tf.multiply(A_cross_entropy, self.transfer_3_HO)) #self.HO_weight))
            L_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=label_HO, logits=cls_score_L)
            L_cross_entropy = tf.reduce_mean(tf.multiply(L_cross_entropy, self.transfer_3_HO)) #self.HO_weight))

            # ipdb.set_trace()
            self.losses['P_cross_entropy']  = P_cross_entropy
            self.losses['A_cross_entropy']  = A_cross_entropy
            self.losses['L_cross_entropy']  = L_cross_entropy
            self.losses['pasta0_cross_entropy']  = P0_cross_entropy
            self.losses['pasta1_cross_entropy']  = P1_cross_entropy
            self.losses['pasta2_cross_entropy']  = P2_cross_entropy
            self.losses['pasta3_cross_entropy']  = P3_cross_entropy
            self.losses['pasta4_cross_entropy']  = P4_cross_entropy
            self.losses['pasta5_cross_entropy']  = P5_cross_entropy
            self.losses['vec_cross_entropy']   = vec_cross_entropy
            self.losses['verb_cross_entropy']  = verb_cross_entropy


            # we may need interative training of S and D, so we add a switch here to control total loss
            # 1--pasta+vec+verb+hoi, 2--pasta+vec, 3--hoi+verb
            if cfg.TRAIN_MODULE == 1:
                loss = A_cross_entropy + L_cross_entropy + P_cross_entropy + PaSta_cross_entropy + vec_cross_entropy + verb_cross_entropy
            elif cfg.TRAIN_MODULE == 2:
                loss = (A_cross_entropy + L_cross_entropy + P_cross_entropy + verb_cross_entropy) * 0 + PaSta_cross_entropy + vec_cross_entropy
            elif cfg.TRAIN_MODULE == 3:
                loss = A_cross_entropy + L_cross_entropy + P_cross_entropy + verb_cross_entropy + (PaSta_cross_entropy + vec_cross_entropy) * 0

            self.losses['total_loss'] = loss
            self.event_summaries.update(self.losses)

        return loss

    def add_score_summary(self, key, tensor):
        tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)

    def add_train_summary(self, var):
        tf.summary.histogram('TRAIN/' + var.op.name, var)

    def train_step(self, sess, blobs, lr, train_op):
        feed_dict = {self.image: blobs['image'], 
                     self.H_boxes: blobs['H_boxes'], self.P_boxes: blobs['P_boxes'], self.R_boxes: blobs['R_boxes'], self.O_boxes: blobs['O_boxes'], 
                     self.gt_class_HO: blobs['gt_class_HO'], 
                     self.gt_class_P0: blobs['gt_class_P0'], self.gt_class_P1: blobs['gt_class_P1'], 
                     self.gt_class_P2: blobs['gt_class_P2'], self.gt_class_P3: blobs['gt_class_P3'], 
                     self.gt_class_P4: blobs['gt_class_P4'], self.gt_class_P5: blobs['gt_class_P5'],
                     self.gt_10v: blobs['gt_10v'], self.gt_verb: blobs['gt_verb'], self.gt_object: blobs['gt_object'], 
                     self.lr: lr, self.H_num: blobs['H_num']}
        
        loss, _ = sess.run([self.losses['total_loss'],
                            train_op],
                            feed_dict=feed_dict)
        return loss

    def train_step_with_summary(self, sess, blobs, lr, train_op):
        feed_dict = {self.image: blobs['image'], 
                     self.H_boxes: blobs['H_boxes'], self.P_boxes: blobs['P_boxes'], self.R_boxes: blobs['R_boxes'], self.O_boxes: blobs['O_boxes'], 
                     self.gt_class_HO: blobs['gt_class_HO'], 
                     self.gt_class_P0: blobs['gt_class_P0'], self.gt_class_P1: blobs['gt_class_P1'], 
                     self.gt_class_P2: blobs['gt_class_P2'], self.gt_class_P3: blobs['gt_class_P3'], 
                     self.gt_class_P4: blobs['gt_class_P4'], self.gt_class_P5: blobs['gt_class_P5'], 
                     self.gt_10v: blobs['gt_10v'], self.gt_verb: blobs['gt_verb'], self.gt_object: blobs['gt_object'], 
                     self.lr: lr, self.H_num: blobs['H_num']}

        loss, summary, _ = sess.run([self.losses['total_loss'], 
                                     self.summary_op, 
                                     train_op], 
                                     feed_dict=feed_dict)
        return loss, summary

    def test_image_HO(self, sess, image, blobs):
        feed_dict = {self.image: image, 
                     self.H_boxes: blobs['H_boxes'], self.P_boxes: blobs['P_boxes'], self.R_boxes: blobs['R_boxes'], self.O_boxes: blobs['O_boxes'], 
                     self.H_num: blobs['H_num'], self.gt_object: blobs['gt_object'], }
        predictions = sess.run([self.predictions["cls_prob_P"], self.predictions["cls_prob_A"], self.predictions["cls_score_L"]], feed_dict=feed_dict)

        return predictions
