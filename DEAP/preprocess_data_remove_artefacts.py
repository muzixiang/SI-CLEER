import multiprocessing

import mne
from scipy.io import loadmat
import _pickle as cPickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import threading
from mne_icalabel import label_components



def preprocess_eeg(p):
    sub_start=p*4
    sub_end=(p+1)*4
    for sub in range(sub_start, sub_end, 1):
        sub_data = cPickle.load(open('./DEAP data_preprocessed_python/{}'.format(experiments[sub]), 'rb'), encoding='iso-8859-1')
        sub_EEG_data = sub_data['data'][:,0:32,:]
        label_data = loadmat(r'./trial_labels_general_valence_arousal_dominance.mat')
        total_trial_label = label_data['trial_labels']
        #各个trial的标签 1,0,-1,-1,0,1,-1,0,1,1,0,-1,0,1,-1
        trial_labels = total_trial_label[sub*40:(sub+1)*40]
        for trial in range(40):
            trial_data = sub_EEG_data[trial,:,:]
            # 将mat数据和电极信息结合成raw，raw包含数据和电极信息
            raw = mne.io.RawArray(trial_data, info)
            raw.set_eeg_reference()
            # raw_ref.set_eeg_reference(ref_channels=['A1','A2'])  raw_ref.set_eeg_reference() 可以设置指定电极、全局平均做重参考

            #绘制图形查看
            # raw.plot(scalings = {'eeg': 50}, block=True, title='可检查并排除坏导')  # 定义坏导
            # plt.show()
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
            # raw.plot(scalings = {'eeg': 50}, block=True, title='可检查并排除坏导')  # 定义坏导
            # plt.show()


            epochs = mne.make_fixed_length_epochs(raw, duration=2, overlap=0.2) #Divide continuous raw data into equal-sized consecutive epochs. seed是200hz,2秒就是400个时间戳，交叠区域是0.2秒
            #epochs.plot()

            # 保存fif格式的数据文件
            sub_name = 'sub'+str(sub+1)
            sub_trials = 40
            trial_No = 'trial'+str(trial+1)
            valence_label = 'valence'+str(trial_labels[trial][0])
            arousal_label = 'arousal'+str(trial_labels[trial][1])
            dominance_label = 'dominance'+str(trial_labels[trial][2])

            eponame = r'../data/Preprocessed_CL/DEAP/{}_{}_{}_{}_{}_{}_epo.fif'.format(str(sub*sub_trials+(trial+1)), sub_name, trial_No,valence_label, arousal_label, dominance_label)
            epochs.save(eponame, overwrite=True)

if __name__ == '__main__':
    # 配置设置电极信息
    # 读取MNE中biosemi电极位置信息（MNE中的默认电极拓扑）
    biosemi_montage = mne.channels.make_standard_montage('biosemi32')
    print(biosemi_montage.get_positions())
    sensor_data = biosemi_montage.get_positions()['ch_pos']
    print(sensor_data)
    sensor_dataframe = pd.DataFrame(sensor_data).T
    print(sensor_dataframe)
    sensor_dataframe.to_excel('sensor_dataframe.xlsx')


    # 图示电极位置
    # biosemi_montage.plot()
    # plt.show()

    # DEAP channel
    ch_names_1 = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4',
                  'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2',  'P4', 'P8', 'PO4', 'O2']

    ch_types_1 = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                  'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                  'eeg', 'eeg', 'eeg', 'eeg']

    info = mne.create_info(ch_names=ch_names_1, ch_types=ch_types_1, sfreq=128)
    info.set_montage(biosemi_montage)

    data_dir = 'DEAP data_preprocessed_python/'
    experiments = os.listdir(data_dir)
    experiments.sort(key=lambda x:int(x.split('.')[0][1:]))
    #experiments.sort()
    #experiments = experiments[:-2]  # 有两个文件无用
    print('total experiments: {}'.format(len(experiments)))



    processes = []
    num_process=8
    for p in range(num_process):
        process = multiprocessing.Process(target=preprocess_eeg, kwargs={'p': p})
        processes.append(process)
        process.start()
    for pp in processes:
        pp.join()