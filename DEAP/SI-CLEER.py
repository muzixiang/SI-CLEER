import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold
from utils import encode
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from models.encoder import TSEncoder
from CNNmodel_multi_class import Net
import torch.nn as nn
import torch
from models.losses import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from sklearn import metrics
from glob import glob
import mne

def take_per_row(A, indx, num_elem):
    all_indx = indx[:,None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:,None], all_indx]

def train():
    model1.train()
    model2.train()

    total_correct = 0.0 #记录所有批次中，预测正确的样本总数
    total = 0.0 #记录所有批次的样本总数
    list_loss = [] #记录每个批次的损失


    # 遍历训练数据加载器的每个批次
    for ii,data in enumerate(train_loader, 1):  
        x, labels = data
        x = x.to(device)
        labels = labels.to(device)

        # 在时间维度上随机裁剪信号
        ts_l = x.size(1)
        crop_l = np.random.randint(low=2 ** (1), high=ts_l+1) # 从2的指数开始（最小为2），到信号长度加1为止，生成一个随机整数
        crop_left = np.random.randint(ts_l - crop_l + 1) # 在信号长度内随机选择左侧裁剪的起始位置
        crop_right = crop_left + crop_l # 计算裁剪的右侧边界位置
        crop_eleft = np.random.randint(crop_left + 1) # 在裁剪范围内随机选择左侧扩展的起始位置
        crop_eright = np.random.randint(low=crop_right, high=ts_l + 1) # 在裁剪范围内随机选择右侧扩展的结束位置
        crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))  # 随机生成裁剪的偏移量，确保裁剪后的片段位于原信号范围内  # 生成一个大小为x.size(0)的随机整数数组，每个值在[-crop_eleft, ts_l - crop_eright + 1]范围内

        # 使用模型进行前向传播得到输出
        out1 = model1(take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft))  #从输入数据 x 中按行提取从 crop_offset + crop_eleft 开始，长度为 crop_right - crop_eleft 的部分时间序列，然后，将提取的时间序列数据传递给模型 model 进行前向传播，以获取其特征表示。结果保存 out1 中
        out1 = out1[:, -crop_l:]
        out2 = model1(take_per_row(x, crop_offset + crop_left, crop_eright - crop_left)) #从输入数据 x 中按行提取从 crop_offset + crop_left 开始，长度为 crop_eright - crop_left 的另一部分时间序列。特征保存到out中
        out2 = out2[:, :crop_l]

        # 计算 hierarchical_contrastive_loss
        loss1 = hierarchical_contrastive_loss(out1,out2)
    

        # 将输入转换为numpy
        x = x.cpu().detach().numpy()

        #得到训练集的表征train_repr
        train_repr = encode(model1, x, encoding_window='full_series')

        #从NumPy数组转换为PyTorch tensor
        train_repr = torch.from_numpy(train_repr)
        train_repr = torch.as_tensor(train_repr,dtype = torch.float32,device="cuda") #将PyTorch tensor转换为指定设备（GPU）上的tensor，并设置数据类型为float32
        train_repr = train_repr.reshape(train_repr.shape[0],train_repr.shape[1]) # 重新调整张量的形状


        #使用模型2进行前向传播并计算classifier损失
        y_pred  = model2(train_repr)
        loss2 = criterion(y_pred, labels.to(torch.int64))


        # 清除之前的梯度
        opt.zero_grad()

        # 计算总损失
        loss_comb = loss1 + loss2

        # 计算预测精度
        preds = torch.argmax(y_pred, dim=1)

        # 计算总数和预测正确的数量
        total_correct += torch.sum(preds == labels)
        total += len(labels)
        mean_acc = total_correct/total

        # 反向传播并更新参数
        loss_comb.backward()
        opt.step()

        # list_loss记录每个批次的损失
        list_loss.append(loss_comb.cpu())

    # 计算所有批次的平均损失值
    mean_loss = compute_mean(list_loss)

    return mean_acc,mean_loss #返回所有批次的哦平均准确率和平均损失值

def test():
    model1.eval()
    model2.eval()

    total = 0
    total_correct = 0


    with torch.no_grad():
         # 遍历测试数据加载器的每个批次
        for ii, data in enumerate(test_loader, 1):  
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

             # 将输入数据转换为NumPy数组
            inputs = inputs.cpu().detach().numpy()

            # 对得到测试集的表征test_repr
            test_repr = encode(model1, inputs, encoding_window='full_series')

            #表征转换为tensor
            test_repr = torch.from_numpy(test_repr)
            test_repr = torch.as_tensor(test_repr,dtype = torch.float32,device="cuda") #将PyTorch tensor转换为指定设备（GPU）上的tensor，并设置数据类型为float32
            test_repr = test_repr.reshape(test_repr.shape[0],test_repr.shape[1]) # 重新调整张量的形状

            # 使用模型2-下游分类器进行前向传播
            outputs = model2(test_repr)
            preds = torch.argmax(outputs, dim=1)

            total_correct += (preds == labels).sum().item()
            total += labels.size(0)


    return total_correct / total

def compute_mean(lst):
    """
    计算给定列表中所有数字的平均值。
    """
    if not lst:
        return None  # 处理空列表的情况，返回None

    total_sum = 0 # 用于存储所有数字的总和
    count = 0 # 记录数字的数量

    # 遍历列表中的每个数字
    for number in lst:
        total_sum += number 
        count += 1

    mean = total_sum / count # 计算平均值
    return mean


# 定义读取数据路径函数，获取病人和健康被试的文件路径
def read_data_path(path, expNo, TEST):
    all_files_path=glob(path)
    negative_file_path = []
    positive_file_path = []
    exp_trial_scope = np.arange(expNo*40+1,(expNo+1)*40+1, 1)
    if TEST=='valance':
        split_idx=3
        strlen=7
    elif TEST=='arousal':
        split_idx=4
        strlen=7
    else:
        split_idx=5
        strlen=9
    for i in all_files_path:
        if int(i.split('/')[4].split('_')[0]) in exp_trial_scope:
            if  '-1' == (i.split('/')[4]).split('_')[split_idx][strlen:]: # 匹配负性的trial数据:
                negative_file_path.append(i)
            elif '1' == (i.split('/')[4]).split('_')[split_idx][strlen:]:
                positive_file_path.append(i)
            else:
                continue
        else:
            continue

    return negative_file_path,positive_file_path

# 定义读取数据函数
def read_data(file_path):
    data=mne.read_epochs(file_path,preload=True) #DEAP数据集的采样率是128
    #data = data.resample(256)
    epochs=data.get_data()
    return epochs


if __name__ == '__main__':
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------
    #1. 参数
    n_splits = 5 #控制几倍交叉验证
    loss_c = 0.25 #平衡模型1和模型2的损失
    num_epochs = 30 #每折训练30个epoch
    exps_accs = []
    #TEST='valence'
    TEST='arousal'
    # TEST='dominance'
    for expNo in range(32):
        print('validation data from experiment (subject) =============================================== {}'.format(expNo+1))
        #2. 读取每类trial数据的路径，15个用户，参加3次实验，每次实验15个trial,分正、中、负三类性质，共675个trial
        neg_file_path,pos_file_path = read_data_path('../data/Preprocessed_CL/DEAP/*.fif', expNo, TEST)
        #-----------------------------------------------------------------------------------------------------------------------------------------------------------

        #3. 读取每个trial的数据
        neg_epochs_array=[read_data(trial) for trial in neg_file_path] # 负性trial的Epochs数据数组, 是列表的列表，其中每个列表是一个trial的所有epoch样本
        pos_epochs_array=[read_data(trial) for trial in pos_file_path] # 正性trial的Epochs数据数组
        n_channles = np.shape(neg_epochs_array[0])[1] #获取数据集的通道数
        #-----------------------------------------------------------------------------------------------------------------------------------------------------------

        #4. 生成标签
        neg_epochs_labels=[len(i)*[0] for i in neg_epochs_array] # 负性trial的各个epoch的标签列表,最后neg_epoch_labels也是list的list
        pos_epochs_labels=[len(i)*[1] for i in pos_epochs_array] # 正性trial的各个epoch的标签列表

        #-----------------------------------------------------------------------------------------------------------------------------------------------------------

        #5. 合并有病和健康的数据、标签
        data_list=neg_epochs_array + pos_epochs_array
        label_list=neg_epochs_labels + pos_epochs_labels
        np.save('DEAP_data_list_exp_{}_2class.npy'.format(expNo+1), data_list)
        np.save('DEAP_label_list_exp_{}_2class.npy'.format(expNo+1), label_list)

        # data_list = np.load('SEED_data_list_exp_{}_3class.npy'.format(expNo+1), allow_pickle=True)
        # label_list = np.load('SEED_label_list_exp_{}_3class.npy'.format(expNo+1), allow_pickle=True)


        cross_validation_results = []  # 初始化用于存储划分结果的列表

        #7. 堆叠数据
        exp_data_list = data_list
        exp_label_list = label_list
        exp_data_array=np.vstack(exp_data_list) # 将所有trial的数据数组垂直堆叠，得到一个大的数据数组
        exp_data_array=np.moveaxis(exp_data_array,1,2) # 调整数据数组的轴，将通道轴移动到第二个位置
        exp_label_array=np.hstack(exp_label_list) # 将所有trial的标签数组水平堆叠，得到一个大的标签数组

        #-----------------------------------------------------------------------------------------------------------------------------------------------------------

        #8. 交叉验证
        accs = []  # 记录每折的最高准确率
        #gkf=GroupKFold(n_splits=n_splits)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2023)
        #sgkf = StratifiedGroupKFold(n_splits=n_splits)
        fold = 1
        for train_index, val_index in skf.split(exp_data_array, exp_label_array):
        #for train_index, val_index in sgkf.split(exp_data_array, exp_label_array, exp_group_array):
            max_acc = 0
            max_f1 = 0
            #--------------------------------------

            # 8.2 获取neg和pos样本的索引
            neg_indices = np.where(exp_label_array == 0)[0]
            pos_indices = np.where(exp_label_array == 1)[0]

            # 8.3 在训练集和测试集中保证neg和pos样本的均衡分布
            train_neg_indices = np.intersect1d(neg_indices, train_index)
            train_pos_indices = np.intersect1d(pos_indices, train_index)
            test_neg_indices = np.intersect1d(neg_indices, val_index)
            test_pos_indices = np.intersect1d(pos_indices, val_index)

            # 8.4 将neg和pos样本索引合并，形成训练集和测试集的索引
            train_indices = np.concatenate((train_neg_indices, train_pos_indices))
            test_indices = np.concatenate((test_neg_indices, test_pos_indices))

            # 8.5 用于最后计算各折的样本类型
            cross_validation_results.append({
                "train_indices": train_indices,
                "test_indices": test_indices
            })
            #---------------------------------------

            # 8.6 获取训练集和测试集数据及标签
            train_data,train_label=exp_data_array[train_indices],exp_label_array[train_indices]
            test_data,test_label=exp_data_array[test_indices],exp_label_array[test_indices]
            print("train_data",train_data.shape,"train_label",train_label.shape)
            print("test_data",test_data.shape,"test_label",test_label.shape)
            #---------------------------------------

            # 8.7 数据标准化,（均值为0，标准差为1）,然后，将标准化后的数据重新转换为与原始形状相同的形状
            scaler=StandardScaler()
            train_data = scaler.fit_transform(train_data.reshape(-1, train_data.shape[-1])).reshape(train_data.shape)
            test_data = scaler.transform(test_data.reshape(-1, test_data.shape[-1])).reshape(test_data.shape)
            #---------------------------------------

            # 8.8 创建训练集和测试集的数据加载器
            train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float),torch.from_numpy(train_label).to(torch.float))
            train_loader = DataLoader(train_dataset, batch_size=min(32, len(train_dataset)), shuffle=True, drop_last=True)

            test_dataset = TensorDataset(torch.from_numpy(test_data).to(torch.float),torch.from_numpy(test_label).to(torch.float))
            test_loader = DataLoader(test_dataset, batch_size=min(32, len(test_dataset)), shuffle=True, drop_last=True)
            #---------------------------------------

            # 8.9 实例化模型
            n_channels=32
            model1 = TSEncoder(input_dims = n_channels).to(device)
            model2 = Net().to(device)
            #---------------------------------------

            # 8.10 初始化一些参数
            criterion = nn.CrossEntropyLoss()
            opt = torch.optim.AdamW([{'params': model1.parameters()}, {'params': model2.parameters()}], 0.001)
            #---------------------------------------

            #8.11
            for epoch in range(num_epochs):
                #训练过程
                train_acc,train_loss = train()
                #测试过程
                test_acc = test()

                #保存最高准确率
                if test_acc>max_acc:
                    max_acc = test_acc

                print(f"Epoch {epoch + 1}/{num_epochs}: train_acc={train_acc:.4f}, train_loss={train_loss:.4f},test_acc={test_acc:.4f}")
            accs.append(max_acc)
        #-----------------------------------------------------------------------------------------------------------------------------------------------------------

        # 9. 打印交叉验证结果
            print(f"Fold {fold}: "
                  f"Train: Negative={len(train_neg_indices)}, "
                  f"Positive={len(train_pos_indices)}/ "
                  f"Test: Negative={len(test_neg_indices)}, "
                  f"Positive={len(test_pos_indices)}"
                  )
            fold = fold+1

        print("Experiment_{} 每折的最高准确率：{}".format(expNo+1, [item for item in accs]))
        print('Experiment_{} 平均最高准确率：{}'.format(expNo+1, np.mean([item for item in accs])))

        exps_accs.append([item for item in accs])
    np.save('{}_exps_accs_2class.npy'.format(TEST), exps_accs)