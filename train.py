import itertools
from feature_group_list import Feature_group_list
from lookup_table import Lookup_table 
from Class_CNN import CNN
import theano
import os
import sys
import os.path
import struct
import numpy as np
import theano.tensor as T
import time
from readMaxGroup import ReadMaxGroup
from train_feature_group_list import Train_feature_group_list
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
sys.setrecursionlimit(1000000)

print("proc ID is:----------------------------------------------------->%s"%(os.getpid(),))
readMaxGroup = ReadMaxGroup()
res = readMaxGroup.read_max_group("/media/new/Data/tangbonew/kaggle_avazu_pCTR_compitition/converted2#_tr_va_te_dataset/FeaExtractTrain")

'''
feature_group_list = Feature_group_list()
res = feature_group_list.readSparseDataFromBinFolder("/media/new/Data/tangbonew/kaggle_avazu_pCTR_compitition/converted2#dataset/train")
print "trainSet count_of_file is:%d"%(feature_group_list.count_of_file,)
'''
'''
testSet_feature_group_list = Feature_group_list()
test = testSet_feature_group_list.readSparseDataFromBinFolder("/media/new/Data/tangbonew/kaggle_avazu_pCTR_compitition/converted2#dataset/test")
count_of_file = testSet_feature_group_list.count_of_file
print "testSet count_of_file is:%d"%(count_of_file,)
'''
pattern_groupid_list = readMaxGroup.pattern_groupid_list
print 'res:%d \n'%(res)
print 'pattern_groupid_list:'
print pattern_groupid_list
lookup_table = Lookup_table(readMaxGroup.maxGroupIndex)
print 'lookup_table\'s table_count:%d\n'%(lookup_table.total_length)


cnn = CNN(readMaxGroup.input_groupid,lookup_table,pattern_groupid_list)
learning_rate = cnn.alpha

batchsize = 500
origin_batchsize = batchsize
epochs = 2
epoch = 0
ite=  0
count = 0# is iterations
stop = False
old_log_loss = 999
epsilo = 1e-8
sum_time = 0

while epoch < epochs and not stop:
	t0 = time.time()
		
	block_size = 100000
	train_feature_group_list = Train_feature_group_list("/media/new/Data/tangbonew/kaggle_avazu_pCTR_compitition/converted2#_tr_va_te_dataset/FeaExtractTrain/train_tr.bin")
	train_feature_group_list.input_groupid = readMaxGroup.input_groupid
	train_feature_group_list.readBlockFile(block_size)
	trainInstanceNum = readMaxGroup.total_instance_num
	blocks = float(trainInstanceNum) / block_size
	blocks = int(blocks)+1
	print "blocks"
	print blocks

	for b in range(0,blocks):
		print("block:%s"%(b))
		if b==blocks-1 and  trainInstanceNum%block_size!=0:
			block_size = trainInstanceNum % block_size
		cnn.feature_group_list = train_feature_group_list
		batchsize = 500
		origin_batchsize = 500
		chunk = int( round( float(block_size) / batchsize ) )
		for batch_index in range(0,chunk):
			count += 1
			if count%100==0:
				print("epoch:%s,load and train data,finished:%s"%(epoch,float(count*origin_batchsize)/trainInstanceNum))
			'''
			induct:every 1000 times mini-batch iterate,calculate log-loss of validation set.
			'''
			if count%10000==0:
				va_block_size = 10000
				print("valid set block size:%s,epoch:%s,start calculating... "%(va_block_size,epoch,))
				block_size_backup = va_block_size
				va_feature_group_list = Train_feature_group_list("/media/new/Data/tangbonew/kaggle_avazu_pCTR_compitition/converted2#_tr_va_te_dataset/FeaExtractValid/train_va.bin")
				va_feature_group_list.input_groupid = readMaxGroup.input_groupid
				vaInstanceNum = va_feature_group_list.count_of_file
				va_blocks = float(vaInstanceNum) / va_block_size
				import math
				va_blocks = int(math.ceil(va_blocks))
				va_labels = []
				va_preds = []
				auc_preds_list = []
				va_log_loss = 0
				
	 
				for bb in range(0,va_blocks):
					if bb==va_blocks-1 and  vaInstanceNum%va_block_size!=0:
						va_block_size = vaInstanceNum % va_block_size
					va_feature_group_list.readBlockFile(va_block_size)
					cnn.feature_group_list = va_feature_group_list
					[(trainx,trainy,list_feature_batch,list_group_batch)] = cnn.load_data(0,block_size_backup,va_block_size)
					va_labels.append(trainy)
					auc = cnn.calc_auc(trainx,trainy,batchsize,list_feature_batch)
					va_preds.append(cnn.pred)
					for value_list in cnn.pred:
						prd_value = value_list[1]
						prd_value = float(prd_value)
						auc_preds_list.append(prd_value)
					va_feature_group_list.list_target = []
					va_feature_group_list.list_feature = []
					va_feature_group_list.list_group = []
					print("calc valid log_loss:va_blocks:%s"%(bb,))
					if bb==va_blocks-1:
						print "va_trainx"
#						print trainx
				va_feature_group_list.BlockTrainFile.close()
				va_labels = list(itertools.chain.from_iterable(va_labels))
				va_preds =  list(itertools.chain.from_iterable(va_preds))
				va_log_loss = log_loss(va_labels,va_preds)
				va_auc = roc_auc_score(va_labels,auc_preds_list)
				va_mse = mean_squared_error(va_labels,auc_preds_list)
				print("------------------------------epoch %s,iterate %s time,valid log_loss:---------------------:%s,AUC:%s,MSE:%s"%(epoch,count,va_log_loss,va_auc,va_mse))
#				print("theta3:")
#				print cnn.theta3
#				print("b3:")
#				print cnn.b3
#				print("theta2:")
#				print cnn.theta2
#				print("b2:")
#				print cnn.b2
#				print("theta1:")
#				print cnn.theta1
#				print("vs:")
#				print cnn.vs
				if abs(old_log_loss - va_log_loss) < epsilo:
					stop = True
					break
				old_log_loss = va_log_loss
				
				
			if batch_index==chunk-1 and block_size%batchsize!=0:
				batchsize = block_size%batchsize
			t_before_load_time = time.time()
			cnn.feature_group_list = train_feature_group_list
			[(trainx,trainy,list_feature_batch,list_group_batch)] = cnn.load_data(batch_index,origin_batchsize,batchsize)
			t_after_load_time = time.time()
			t_before_train_time = time.time()
			cnn.train(batchsize,trainx,trainy,list_feature_batch,list_group_batch)
			t_after_train_time = time.time()

				
				
		batchsize = 500      #for batch_index in one block
		if stop == True :
			break
		train_feature_group_list.list_target = []
		train_feature_group_list.list_feature = []
		train_feature_group_list.list_group = []
		label = train_feature_group_list.BlockTrainFile.tell()
#		train_feature_group_list.BlockTrainFile.close()
		if b==blocks-1:
			break
		train_feature_group_list.BlockTrainFile.seek(label,0)
		if b==blocks-2 and trainInstanceNum % block_size!=0:
			train_feature_group_list.readBlockFile( trainInstanceNum % block_size )
			continue
		train_feature_group_list.readBlockFile(block_size)
	train_feature_group_list.BlockTrainFile.close()   #for block in blocks
	if stop == True :
		break
	epoch += 1


	if epoch == 2:
		cnn.alpha = learning_rate / 100
		#learning_rate = learning_rate / 10
#	if epoch == 4:
#		cnn.alpha = learning_rate / 100
	t1 = time.time()
	print('this epoch takes time(seconds):')
	print t1-t0
	sum_time += (t1-t0)
	batchsize = origin_batchsize
	print 'all epochs take time(seconds):'
	print sum_time
#end while
if stop == True:
	print("old_log_loss - trainset_log_loss < epsilo ----->epoch converagence.")
else:
	print("not converagence,only epoch equals epochs.")
'''
reading test file by blocking,because test file is scalar.
'''


test_feature_group_list = Train_feature_group_list("/media/new/Data/tangbonew/kaggle_avazu_pCTR_compitition/converted2#_tr_va_te_dataset/FeaExtractTest/test.bin")
test_feature_group_list.input_groupid = readMaxGroup.input_groupid
test_instance_num = test_feature_group_list.count_of_file
cnn_test = CNN(readMaxGroup.input_groupid,lookup_table,pattern_groupid_list)
test_block_size = 10000
origin_test_block_size = test_block_size
blocks = int(math.ceil(float(test_instance_num)/test_block_size))

te_labels = []
te_preds = []
auc_preds_list = []

for block_index in range(0,blocks):
	if block_index==blocks-1 and test_instance_num%test_block_size!=0:
		test_block_size = test_instance_num%test_block_size
	test_feature_group_list.readBlockFile(test_block_size)
	cnn_test.feature_group_list = test_feature_group_list
	print len(test_feature_group_list.list_target)
	print test_block_size
	print origin_test_block_size
	[(testx,testy,list_feature_batch,list_group_batch)] = cnn_test.load_data(0,origin_test_block_size,test_block_size)
	te_labels.append(testy)

	auc = cnn.calc_auc(testx,testy,batchsize,list_feature_batch)
	te_preds.append(cnn.pred)
	for value_list in cnn.pred:
		prd_value = value_list[1]
		prd_value = float(prd_value)
		auc_preds_list.append(prd_value)

	test_feature_group_list.list_target = []
	test_feature_group_list.list_feature = []
	test_feature_group_list.list_group = []
#	print 'epoch :%d ,AUC:%s'%(epoch,auc)
test_feature_group_list.BlockTrainFile.close()

te_labels = list(itertools.chain.from_iterable(te_labels))
te_preds =  list(itertools.chain.from_iterable(te_preds))

va_log_loss = log_loss(te_labels,te_preds)
va_auc = roc_auc_score(te_labels,auc_preds_list)
va_mse = mean_squared_error(te_labels,auc_preds_list)

print("test set log_loss:%s, AUC:%s, MSE:%s"%(va_log_loss,va_auc,va_mse))


