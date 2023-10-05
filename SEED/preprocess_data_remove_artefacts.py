import multiprocessing

import mne
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import threading
from mne_icalabel import label_components



def preprocess_eeg(p):
    exp_start=p*3
    exp_end=(p+1)*3
    for exp in range(exp_start, exp_end, 1):
        exp_data = loadmat(r'./SEED_EEG/Preprocessed_EEG/{}'.format(experiments[exp]))
        #各个trial的标签 1,0,-1,-1,0,1,-1,0,1,1,0,-1,0,1,-1
        trial_labels = [1,0,-1,-1,0,1,-1,0,1,1,0,-1,0,1,-1]
        for trial_key in exp_data:
            #print(exp_key)
            if trial_key in ['__header__','__version__','__globals__']:
                continue
            trial_data = exp_data[trial_key]
            # 将mat数据和电极信息结合成raw，raw包含数据和电极信息
            raw = mne.io.RawArray(trial_data, info)
            raw.set_eeg_reference()
            # raw_ref.set_eeg_reference(ref_channels=['A1','A2'])  raw_ref.set_eeg_reference() 可以设置指定电极、全局平均做重参考

            #绘制图形查看
            raw.plot(scalings = {'eeg': 50}, block=True, title='可检查并排除坏导')  # 定义坏导
            plt.show()
            # raw.load_data()
            # raw.interpolate_bads()
            # 滤波
            raw.filter(l_freq=1, h_freq=49)  # 带通滤波
            #raw.notch_filter(freqs=60)  # 凹陷滤波
            #psd图
            # raw.compute_psd().plot()
            # plt.show()

            # 独立分析法ICA
            ica = mne.preprocessing.ICA()  # 相当于先定义一个ica方法
            #ica = mne.preprocessing.ICA(method='infomax', fit_params=dict(extended=True))
            ica.fit(raw)  # 训练raw

            ic_labels = label_components(raw, ica, method='iclabel')
            labels = ic_labels['labels']
            bad_ICs = [idx for idx, label in enumerate(labels) if label not in ['brain', 'other']]
            print('exclude these bad ICA components: {}'.format(bad_ICs))


            # 可视化每个独立分布的头皮分布
            #ica.plot_components()
            # 绘制mei个独立分布的一些属性
            # pick_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # [0,1,2,3,4,5,6,7,8,9]
            # ica.plot_properties(raw, picks=pick_id)

            # ica.plot_sources(raw, show_scrollbars=False, block=True,
            #                  title='请选择需要去除的成分')  # show_scrollbars绘图初始化时是否显示滚动条。
            # plt.show()


            raw = ica.apply(raw, exclude=bad_ICs)  # 把选取的成分去除掉 去除weiji
            raw.plot(scalings = {'eeg': 50}, block=True, title='可检查并排除坏导')  # 定义坏导
            plt.show()


            epochs = mne.make_fixed_length_epochs(raw, duration=2, overlap=0.2) #Divide continuous raw data into equal-sized consecutive epochs. seed是200hz,2秒就是400个时间戳
            #epochs.plot()

            # 保存fif格式的数据文件
            trial_label = trial_labels[int(trial_key[-1])-1]
            sub_name = trial_key.split('_')[0]
            eegx = trial_key.split('_')[1]

            exp_trials = 15
            trial_No = int(trial_key.split('_')[1][3:])
            eponame = r'../data/Preprocessed_CL/SEED/{}_{}_{}_{}_epo.fif'.format(str(exp*exp_trials+trial_No), sub_name, eegx, trial_label)
            epochs.save(eponame, overwrite=True)

if __name__ == '__main__':
    # 配置设置电极信息
    # 读取MNE中biosemi电极位置信息（MNE中的默认电极拓扑）
    # biosemi_montage = mne.channels.make_standard_montage('biosemi64')
    # print(biosemi_montage.get_positions())
    # sensor_data = biosemi_montage.get_positions()['ch_pos']
    # print(sensor_data)
    # sensor_dataframe = pd.DataFrame(sensor_data).T
    # print(sensor_dataframe)
    # sensor_dataframe.to_excel('sensor_dataframe.xlsx')

    # 获取的除ch_pos外的信息
    '''
    'coord_frame': 'unknown', 'nasion': array([ 5.27205792e-18,  8.60992398e-02, -4.01487349e-02]),
    'lpa': array([-0.08609924, -0.        , -0.04014873]), 'rpa': array([ 0.08609924,  0.        , -0.04014873]),
    'hsp': None, 'hpi': None
    '''
    # 将获取的电极位置信息修改并补充缺失的SEED电极位置，整合为1020.xlsx
    data1020 = pd.read_excel('SEED_1020.xlsx', index_col=0)
    channels1020 = np.array(data1020.index)
    value1020 = np.array(data1020)

    # 将电极通道名称和对应三维坐标位置存储为字典形式
    list_dic = dict(zip(channels1020, value1020))
    print(list_dic)
    # 封装为MNE的格式，参考原biosemi的存储格式
    montage_1020 = mne.channels.make_dig_montage(ch_pos=list_dic,
                                                 nasion=[5.27205792e-18, 8.60992398e-02, -4.01487349e-02],  # 鼻根
                                                 lpa=[-0.08609924, -0., -0.04014873],  # 左耳前突
                                                 rpa=[0.08609924, 0., -0.04014873])  # 右耳前突

    # 图示电极位置
    # montage_1020.plot()
    # plt.show()

    # SEED channel
    ch_names_1 = ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6'
        , 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5'
        , 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2'
        , 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7'
        , 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'Oz', 'O2', 'CB2']

    ch_types_1 = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                  'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                  'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                  'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                  'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg']

    info = mne.create_info(ch_names=ch_names_1, ch_types=ch_types_1, sfreq=200)
    info.set_montage(montage_1020)

    data_dir = 'SEED_EEG/Preprocessed_EEG/'
    experiments = os.listdir(data_dir)
    experiments.sort(key=lambda x:int(x.split('_')[0]))
    #experiments.sort()
    #experiments = experiments[:-2]  # 有两个文件无用
    print('total experiments: {}'.format(len(experiments)))



    processes = []
    num_process=15
    for p in range(num_process):
        process = multiprocessing.Process(target=preprocess_eeg, kwargs={'p': p})
        processes.append(process)
        process.start()
    for pp in processes:
        pp.join()