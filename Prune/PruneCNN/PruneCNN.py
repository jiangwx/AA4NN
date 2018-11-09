import caffe
import numpy as np
from numpy import array
from numpy.random import normal
from matplotlib import pyplot
import os, sys
import struct
prototxt = '../Darknet19/Darknet19.prototxt'
caffemodel = '../Darknet19/Darknet19.caffemodel'
TEST_ITER = 375
caffe.set_device(0)  
caffe.set_mode_gpu()

def test_accuracy(net, test_iter):
    loss = np.zeros(1)
    top1 = np.zeros(1)
    top5 = np.zeros(1)
    test_loss = 0
    top1_accuracy = 0
    top5_accuracy = 0
    for test_it in range(test_iter):
        net.forward()
        test_loss += net.blobs['loss'].data
        top1_accuracy += net.blobs['accuracy'].data  
        top5_accuracy += net.blobs['accuracy-top5'].data
    loss[0] = test_loss / test_iter 
    top1[0] = top1_accuracy / test_iter
    top5[0] = top5_accuracy / test_iter
    print 'Loss: ', loss[0], 'Accuracy Top1:', top1[0], 'Accuracy Top5:', top5[0]
    return loss[0], top1[0], top5[0]

def prune_fm(net, layer, iteration):

    zeros = np.zeros(net.params[layer][0].data[0].shape)
    loss = np.zeros(net.blobs[layer].channels)
    top1 = np.zeros(net.blobs[layer].channels)
    top5 = np.zeros(net.blobs[layer].channels)
    
    for i in range(net.blobs[layer].channels):
        net.copy_from(caffemodel)
        net.params[layer][0].data[i] = zeros
        print 'pruning', layer, 'feature map',i, '...'
        loss[i],top1[i],top5[i] = test_accuracy(net, iteration)
    return loss,top1,top5

def test_pruned_fm(prototxt, caffemodel, layer, iteration, prune_rate):
    
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    zeros=np.zeros(net.params[layer][0].data[0].shape)
    loss = np.zeros(net.blobs[layer].channels)
    top1 = np.zeros(net.blobs[layer].channels)
    top5 = np.zeros(net.blobs[layer].channels)
    
    prune_cnt = int(prune_rate*net.blobs[layer].channels)
    
    [loss,top1,top5]=prune_fm(net, layer, iteration)
    
    print 'cut unimportant features...'

    loss_index = np.argsort(loss)
    top1_index = np.argsort(-top1)
    top5_index = np.argsort(-top5)
    
    print 'prune basing on top1...'
    print 'fm to be pruned:', top1_index[:prune_cnt]
    net.copy_from(caffemodel)
    for i in range(prune_cnt):
        net.params[layer][0].data[top1_index[i]]=zeros
    test_accuracy(net, TEST_ITER)
    
    print 'prune basing on loss...'
    print 'fm to be pruned:', loss_index[:prune_cnt]
    net.copy_from(caffemodel)
    for i in range(prune_cnt):
        net.params[layer][0].data[loss_index[i]]=zeros
    test_accuracy(net, TEST_ITER)
    
    print 'cut important features...'

    loss_index = np.argsort(-loss)
    top1_index = np.argsort(top1)
    top5_index = np.argsort(top5)
    
    print 'prune basing on top1...'
    print 'fm to be pruned:', top1_index[:prune_cnt]
    net.copy_from(caffemodel)
    for i in range(prune_cnt):
        net.params[layer][0].data[top1_index[i]]=zeros
    test_accuracy(net, TEST_ITER)
    
    print 'prune basing on loss...'
    print 'fm to be pruned:', loss_index[:prune_cnt]
    net.copy_from(caffemodel)
    for i in range(prune_cnt):
        net.params[layer][0].data[loss_index[i]]=zeros
    test_accuracy(net, TEST_ITER)

test_pruned_fm(prototxt, caffemodel, 'conv1', 2, 0.375)