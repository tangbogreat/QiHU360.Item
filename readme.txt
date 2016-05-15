注意：2016/4/18日上午，改了ClassCNN.py文件，主要是之前改变batchsize时，代码出现问题，肯定是因为之前设置的batchsize=self.hid2_nodenum引起的。
所以，其他文件夹下面的ClassCNN.py文件都需要更换为这个目录下面的ClassCNN.py文件。

接下来，还可以改进的地方：
1、把k_size设置较大，如：20,30,...
2、把隐含层设置大一些，如：500:200,500:150...
3、分析数据特征。
4、把lamda设置大一些或小一些，如：0.1...等

1.
l2_norm.log文件里面，有2范数，lamda:=0.05，lr:=0.001，k_size:=3，layer:=400:100:2，embedding_size:=12。

2.
log1.log文件里面，有2范数，其余都保持不变.只调整lr:
因为，通过excel绘制valid loss图发现，从epoch=1开始之后(迭代了一个epoch)，之后loss波动有点大，并且没有下降，
所以：
自epoch=1开始，以后的lr := lr/10.即：之后的每个epoch的lr等于epoch=0的lr/10

3.
log1.log文件里面的第一次iteration，会比较高，因为还没有经过任何训练。
4.
log2.log文件里面，其他参数相同，注意学习速率lr,与log1.log有区别：
lr := lr/10 ，即：当前的lr等于上一次epoch的lr/10
5.
log3.log文件，lr=0.001,k=5,batchsize=300.count%30000,1 epoch= 4 iters目的：调整不同的batchsize(这里设置为300)。

6.
log4.log文件，init lr=0.001,从第3个epoch开始，lr=0.00001，k=5, batchsize=500,count%10000, 1 epoch= 6 iters。目的，跑avazu 9/10个样本训练集，和1/10样本测试集(从真实训练集切割出来的)，与冠军模型和其他模型可以比较AUC,MSE,LOG-LOSS等其他benchmark

7.
log5.log文件，目的，跑的数据和（6.）相同，区别在于，学习速率一直不变，0.001.为了观察AUC、MSE值的变化情况。

8.
log6.log文件，目的，跑到数和(6.)相同,init lr=0.001, 第3、4个epoch,lr=0.0001，从第5个epoch后，lr=0.00001;k=5, batchsize=700. count%5000, 1 epoch ~= 8 iters

9.
log7.log文件，lr=0.001固定, batchsize=900, k=4, count%5000, 1 epoch = 8 iters. va_block_size=50000

10.
log8.log文件，初始化lr=0.001,从第3个epoch开始，lr=0.00001，k=15, batchsize=500, count%10000, 1 epoch= 6 iters.目的：设置较大的k，使得每个因子分解项更多。表达能力更强。

11.
log9.log文件，配置同10，区别在于：对数据集进行了特征提取，即将groupid=1的ad identifer特征删除了。因此，max_groupid=22.修改了文件：feature_group_list.py和readMaxGroup

12.
log10.log文件，配置：。初始化lr=0.001,从第3个epoch开始，lr=0.00001，k=15, batchsize=500, count%10000, 1 epoch= 6 iters.net: 900:100:2, embedding=12.

13.
log11.log文件，配置：。初始化lr=0.001,从第3个epoch开始，lr=0.00001，k=15, batchsize=500, count%10000, 1 epoch= 6 iters.net: 900:100:2,embedding=5.

14.
log12.log文件，配置：。初始化lr=0.001,从第3个epoch开始，lr=0.00001，k=15, batchsize=500, count%10000, 1 epoch= 6 iters.net: 900:100:2,embedding=40.

15.
log13.log文件，配置：。初始化lr=0.001,从第3个epoch开始，lr=0.00001，k=15, batchsize=500, count%10000, 1 epoch= 6 iters.net: 900:100:2,embedding=40,epochs=2，测试部分代码是否正确.
