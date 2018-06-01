import os
import numpy as np
import data
import model
import tensorflow as tf

# configs
FLAGS = tf.app.flags.FLAGS
# mode
tf.app.flags.DEFINE_boolean('is_training', True, 'training or testing')
# data
tf.app.flags.DEFINE_string('root_dir', '/data/DL_HW2', 'data root dir')
tf.app.flags.DEFINE_string('dataset', 'dset1', 'dset1 or dset2')
tf.app.flags.DEFINE_integer('n_label', 65, 'number of classes')
# trainig
tf.app.flags.DEFINE_integer('batch_size', 64, 'mini batch for a training iter')
tf.app.flags.DEFINE_string('save_dir', './checkpoints', 'dir to the trained model')
# test
tf.app.flags.DEFINE_string('my_best_model', './checkpoints/model.ckpt-3000', 'for test')


'''TODO: you may add more configs such as base learning rate, max_iteration,
display_iteration, valid_iteration and etc. '''

# hyperparameters
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
tf.app.flags.DEFINE_integer('max_iteration', 3000, 'number of batch for training')
tf.app.flags.DEFINE_integer('display_iteration', 100, 'display the loss and accuracy on train set')
tf.app.flags.DEFINE_integer('valid_iteration', 100, 'display the loss and accuracy on validation set')

def train_wrapper(model):
    '''Data loader'''
    train_set = data.DataSet(FLAGS.root_dir, FLAGS.dataset, 'train',
                        FLAGS.batch_size, FLAGS.n_label,
                        data_aug=False, shuffle=True)
    valid_set = data.DataSet(FLAGS.root_dir, FLAGS.dataset, 'val',
                        FLAGS.batch_size, FLAGS.n_label,
                        data_aug=False, shuffle=False)
    '''create a tf session for training and validation
    TODO: to run your model, you may call model.train(), model.save(), model.valid()'''
    best_accuracy = 0
    acc_train = []
    acc_valid = []
    for step in range(1, FLAGS.max_iteration+1):
        if not train_set.has_next_batch():
            train_set.init_epoch()     
        batch_x, batch_y = train_set.next_batch()
        if len(batch_x) == FLAGS.batch_size:
            loss, acc = model.train(batch_x, batch_y)
            if step == 1 or step % 10 == 0:
                print("Step " + str(step) + ", Minibatch Loss= " + \
                "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.3f}".format(acc))
        if step % FLAGS.valid_iteration == 0:
            acc_train.append(acc)
            tot_acc = 0.0
            tot_input = 0
            while valid_set.has_next_batch():
                valid_ims, valid_labels = valid_set.next_batch()
                loss, acc = model.valid(valid_ims, valid_labels)
                tot_acc += acc*len(valid_ims)
                tot_input += len(valid_ims)
            acc = tot_acc / tot_input
            valid_set.init_epoch()
            print("Current Accuracy= " + "{:.3f}".format(acc))
            acc_valid.append(acc)          
            if acc > best_accuracy:
                model.save(step)
                best_accuracy = acc

    print("Optimization Finished!")


def test_wrapper(model):
    '''Test your code.'''    
    test_set = data.DataSet(FLAGS.root_dir, FLAGS.dataset, 'test',
                       FLAGS.batch_size, FLAGS.n_label,
                       data_aug=False, shuffle=False)
    '''TODO: Your code here.'''
    model.load()
    tot_acc = 0.0
    tot_input = 0
    while test_set.has_next_batch():
        test_ims, test_labels = test_set.next_batch()
        _, acc = model.valid(test_ims, test_labels)
        tot_acc += acc*len(test_ims)
        tot_input += len(test_ims)
    acc = tot_acc / tot_input
    print("Test Accuracy= " + "{:.3f}".format(acc))
    print("Test Finished!")


def main(argv=None):
    print('Initializing models')
    model = model.Model()
    if FLAGS.is_training:
        train_wrapper(model)
    else:
        test_wrapper(model)


if __name__ == '__main__':
    tf.app.run()