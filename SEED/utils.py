import os
import numpy as np
import pickle
import torch
import random
from datetime import datetime
import torch.nn.functional as F #reLU
from torch.utils.data import TensorDataset, DataLoader
from glob import glob
import mne


def fuzhi(pd_count,hc_count):
    data2 = []
    k = 0
    pd_count2 = 0
    hc_count2 = 0 
    temp = []
    for i in range(0,pd_count):
        data_j = []
        k+= 1
        for j in range(len(data[i])):
            den = np.mean(np.abs(np.fft.fft(data[i][j])))
            data_j.append(den)
        print(k,"===",np.mean(data_j))
        temp.append(np.mean(data_j))
        if(np.mean(data_j)<4):
            data2.append(data[i])
            pd_count2+=1

    for i in range(pd_count,hc_count+pd_count):
        data_j = []
        k+= 1
        for j in range(len(data[i])):
            den = np.mean(np.abs(np.fft.fft(data[i][j])))
            data_j.append(den)
        print(k,"===",np.mean(data_j))
        temp.append(np.mean(data_j))
        if(np.mean(data_j)<4):
            data2.append(data[i])
            hc_count2+=1

    data = np.array(data2)

    print("振幅筛选后:",data.shape,pd_count2,hc_count2)
    return data,pd_count2,hc_count2


def _eval_with_pooling(net, x, mask=None, encoding_window=None):
    out = net(x.to('cuda', non_blocking=True), mask)
    if encoding_window == 'full_series':
        out = F.max_pool1d(out.transpose(1, 2),kernel_size = out.size(1),).transpose(1, 2)
    return out.cpu()

def encode(net, data, mask=None, encoding_window=None,batch_size=16):
    org_training = net.training
    net.eval()
    dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
    loader = DataLoader(dataset, batch_size=batch_size)
    with torch.no_grad():
        output = []
        for batch in loader:
            x = batch[0]
            out = _eval_with_pooling(net, x, mask, encoding_window=encoding_window)
            if encoding_window == 'full_series':
                out = out.squeeze(1)
            output.append(out)
        output = torch.cat(output, dim=0)
    net.train(org_training)
    return output.numpy()

def read_data(file_path):
    # pick = ['O2','PO3','P7','PO4','Cz','FC2','AF4','Fp2','P8','Oz','C4','F8','O1','FC6','P3','C3','FC5','F7','F4','CP6','F3','Fz','AF3'] #
    # pick = ['C3','Cz','C4','01','02']
   
    if "unm" in file_path:
        data=mne.read_epochs(file_path,preload=True)
        data = data.resample(256)
        # epochs=data.get_data(units='uV', picks=pick)*10000
        epochs=data.get_data()*10000
    elif "uc_auto_process_data" in file_path:
        data=mne.read_epochs(file_path,preload=True)
        data = data.resample(256)
        # epochs=data.get_data(units='uV', picks=pick)*10000
        epochs=data.get_data()*10000
    else:
        data=mne.read_epochs(file_path,preload=True)
        data = data.resample(256)
        # epochs=data.get_data(units='uV', picks=pick)
        epochs=data.get_data()
    return epochs

def unm_data(path):
    all_files_path=glob(path)
    patient_file_path=[i for i in all_files_path if  'PD' in i.split('\\')[-1]]
    healthy_file_path=[i for i in all_files_path if  'HC' in i.split('\\')[-1]]
    return patient_file_path,healthy_file_path

def ui_data(path):
    all_files_path=glob(path)
    patient_file_path=[i for i in all_files_path if  'PD' in i.split('\\')[-1]]
    healthy_file_path=[i for i in all_files_path if  'ol' in i.split('\\')[-1]]
    return patient_file_path,healthy_file_path

def uc_data(path):
    all_files_path=glob(path)
    patient_file_path=[i for i in all_files_path if  'pd' in i.split('\\')[-1]]
    healthy_file_path=[i for i in all_files_path if  'h' in i.split('\\')[-1]]
    return patient_file_path,healthy_file_path


def pkl_save(name, var):
    with open(name, 'wb') as f:
        pickle.dump(var, f)

def pkl_load(name):
    with open(name, 'rb') as f:
        return pickle.load(f)
    
def torch_pad_nan(arr, left=0, right=0, dim=0):
    if left > 0:
        padshape = list(arr.shape)
        padshape[dim] = left
        arr = torch.cat((torch.full(padshape, np.nan), arr), dim=dim)
    if right > 0:
        padshape = list(arr.shape)
        padshape[dim] = right
        arr = torch.cat((arr, torch.full(padshape, np.nan)), dim=dim)
    return arr
    
def pad_nan_to_target(array, target_length, axis=0, both_side=False):
    assert array.dtype in [np.float16, np.float32, np.float64]
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    if both_side:
        npad[axis] = (pad_size // 2, pad_size - pad_size//2)
    else:
        npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode='constant', constant_values=np.nan)

def split_with_nan(x, sections, axis=0):
    assert x.dtype in [np.float16, np.float32, np.float64]
    arrs = np.array_split(x, sections, axis=axis)
    target_length = arrs[0].shape[axis]
    for i in range(len(arrs)):
        arrs[i] = pad_nan_to_target(arrs[i], target_length, axis=axis)
    return arrs

def take_per_row(A, indx, num_elem):
    all_indx = indx[:,None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:,None], all_indx]

def centerize_vary_length_series(x):
    prefix_zeros = np.argmax(~np.isnan(x).all(axis=-1), axis=1)
    suffix_zeros = np.argmax(~np.isnan(x[:, ::-1]).all(axis=-1), axis=1)
    offset = (prefix_zeros + suffix_zeros) // 2 - prefix_zeros
    rows, column_indices = np.ogrid[:x.shape[0], :x.shape[1]]
    offset[offset < 0] += x.shape[1]
    column_indices = column_indices - offset[:, np.newaxis]
    return x[rows, column_indices]

def data_dropout(arr, p):
    B, T = arr.shape[0], arr.shape[1]
    mask = np.full(B*T, False, dtype=np.bool)
    ele_sel = np.random.choice(
        B*T,
        size=int(B*T*p),
        replace=False
    )
    mask[ele_sel] = True
    res = arr.copy()
    res[mask.reshape(B, T)] = np.nan
    return res

def name_with_datetime(prefix='default'):
    now = datetime.now()
    return prefix + '_' + now.strftime("%Y%m%d_%H%M%S")

def init_dl_program(
    device_name,
    seed=None,
    use_cudnn=True,
    deterministic=False,
    benchmark=False,
    use_tf32=False,
    max_threads=None
):
    import torch
    if max_threads is not None:
        torch.set_num_threads(max_threads)  # intraop
        if torch.get_num_interop_threads() != max_threads:
            torch.set_num_interop_threads(max_threads)  # interop
        try:
            import mkl
        except:
            pass
        else:
            mkl.set_num_threads(max_threads)
        
    if seed is not None:
        random.seed(seed)
        seed += 1
        np.random.seed(seed)
        seed += 1
        torch.manual_seed(seed)
        
    if isinstance(device_name, (str, int)):
        device_name = [device_name]
    
    devices = []
    for t in reversed(device_name):
        t_device = torch.device(t)
        devices.append(t_device)
        if t_device.type == 'cuda':
            assert torch.cuda.is_available()
            torch.cuda.set_device(t_device)
            if seed is not None:
                seed += 1
                torch.cuda.manual_seed(seed)
    devices.reverse()
    torch.backends.cudnn.enabled = use_cudnn
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        
    return devices if len(devices) > 1 else devices[0]

