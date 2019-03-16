import tensorflow as tf
import csv
import numpy as np
import random


train_file_path ='/Users/macbookpro/Documents/script/python/KDD_traindata.csv'
test_file_path='/Users/macbookpro/Documents/script/python/KDD_testdata.csv'
line_num=494021
feature=[]
label=[]
test_feature=[]
test_label=[]

def str_to_float(arr):
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            arr[i][j]=float(arr[i][j])
def next_batch(feature_list,label_list,size,ss):
    feature_batch_temp=[]
    label_batch_temp=[]
    f_list = range(ss*batch_size,ss*batch_size+batch_size)
    for i in f_list:
        feature_batch_temp.append(feature_list[i])
    for i in f_list:
        label_batch_temp.append(label_list[i])
    return feature_batch_temp,label_batch_temp
with (open(train_file_path,'r')) as data_from:
    csv_reader=csv.reader(data_from)
    for i in csv_reader:          
        feature.append(i[0:41])
        label.append(i[41:])
    feature=np.array(feature)
    label=np.array(label)
    # str_to_float(feature)
    # str_to_float(label)
  
with (open(test_file_path,'r')) as data_from:
    csv_reader=csv.reader(data_from)
    for i in csv_reader:          
        test_feature.append(i[0:41])
        test_label.append(i[41:])

    test_feature=np.array(test_feature)
    test_label=np.array(test_label)
    # str_to_float(test_feature)
    # str_to_float(test_label)

training_epoch=10
batch_size=64
learn_rate1=0.0002
learn_rate2=0.0001
n_h=90
X=tf.placeholder(tf.float32,[None,41])
Y=tf.placeholder(tf.float32,[None,2])
init = tf.random_uniform_initializer(minval=-0.05, maxval=0.05, seed=123)
W1=tf.get_variable(name='w1',shape=[41, n_h], initializer=init)
b1=tf.get_variable(name='b1',shape=[n_h], initializer=init)
L1=tf.nn.relu(tf.matmul(X,W1)+b1)

W2=tf.get_variable(name='w2',shape=[n_h, n_h], initializer=init)
b2=tf.get_variable(name='b2',shape=[n_h], initializer=init)
L2=tf.nn.relu(tf.matmul(L1,W2)+b2)

W3=tf.get_variable(name='w3',shape=[n_h, n_h], initializer=init)
b3=tf.get_variable(name='b3',shape=[n_h], initializer=init)
L3=tf.nn.relu(tf.matmul(L2,W3)+b3)

W4=tf.get_variable(name='w4',shape=[n_h, n_h], initializer=init)
b4=tf.get_variable(name='b4',shape=[n_h], initializer=init)
L4=tf.nn.relu(tf.matmul(L3,W4)+b4)

W5=tf.get_variable(name='w5',shape=[n_h, 2], initializer=init)
b5=tf.get_variable(name='b5',shape=[2], initializer=init)
hypothesis = tf.nn.sigmoid(tf.matmul(L4, W5) + b5)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer1 = tf.train.AdamOptimizer(learning_rate=learn_rate1).minimize(cost)
optimizer2 = tf.train.AdamOptimizer(learning_rate=learn_rate2).minimize(cost)
is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1)) # Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epoch):
        if epoch<=5:
            avg_cost=0
            iteration=int(line_num/batch_size)
            for s in range(iteration):
                x_batch,y_batch=next_batch(feature,label, batch_size,s)
                costval,_=sess.run([cost,optimizer1],feed_dict={X:x_batch,Y:y_batch})
               
                #avg_cost+=costval/iteration
            #print('epoch:','%3d'%(epoch+1),'costval=','{:.9f}'.format(avg_cost))
                if s%1000==0:
                    print('ite:','%3d'%(s+1),'costval=','{:.9f}'.format(costval))
            #print('accuracy= ',sess.run(accuracy,feed_dict={X:test_feature,Y:test_label})) 
                    TN=0
                    FN=0
                    TP=0
                    FP=0
                    DR=0
                    FAR=0
                    hypothesis1 = (hypothesis.eval({X:test_feature[0:100000],Y:test_label[0:100000]},sess)).tolist()
                    print(np.shape(hypothesis1))
                    print(type(hypothesis1))
                    for aa in range(len(test_feature[0:100000])):
                        if hypothesis1[aa][0]<hypothesis1[aa][1] and int(test_label[aa][0])==0:
                            TN+=1
                        if hypothesis1[aa][0]<hypothesis1[aa][1] and int(test_label[aa][0])==1:
                            FN+=1
                        if hypothesis1[aa][0]>=hypothesis1[aa][1] and int(test_label[aa][0])==0:
                            FP+=1
                        if hypothesis1[aa][0]>=hypothesis1[aa][1] and int(test_label[aa][0])==1:
                            TP+=1
                    DR=TP/(TP+FN)
                    FAR=FP/(TN+FP)
                    print('DR=','{:.9f}'.format(DR),'FAR=','{:.9}'.format(FAR))
        else:
            avg_cost=0
            iteration=int(line_num/batch_size)
            for s in range(iteration):
                x_batch,y_batch=next_batch(feature,label, batch_size,s)
                costval,_=sess.run([cost,optimizer2],feed_dict={X:x_batch,Y:y_batch})
                #avg_cost+=costval/iteration
            #print('epoch:','%3d'%(epoch+1),'costval=','{:.9f}'.format(avg_cost))
                if s%1000==0:
                    print('ite:','%3d'%(s+1),'costval=','{:.9f}'.format(costval))
    #print('learnfinished!')    
            print('accuracy= ',sess.run(accuracy,feed_dict={X:test_feature,Y:test_label})) 
