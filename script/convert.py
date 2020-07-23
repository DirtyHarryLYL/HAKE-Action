from tensorflow.python import pywrap_tensorflow
import tensorflow as tf
import sys

def rename_var(ckpt_path, new_ckpt_path):
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(ckpt_path):
            print(var_name)
            var = tf.contrib.framework.load_variable(ckpt_path, var_name)
            new_var_name = var_name.replace('pvp', 'pasta')
            var = tf.Variable(var, name=new_var_name)

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.save(sess, new_ckpt_path)

if len(sys.argv) != 3:
    assert 0, "Two arguments required"
rename_var(sys.argv[1], sys.argv[2])