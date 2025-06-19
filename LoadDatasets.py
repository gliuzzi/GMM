import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os
from urllib import request, parse
import zipfile
import torch
from sklearn.datasets import load_svmlight_file
import requests
import bz2

NUM_WORKERS = 0
BATCH_SIZE = 16000

def get_dataset(dataset_name, save_path='Dataset'): 
    # returns torch dataloader and the number of classes (if 1 then is regression)

    if dataset_name in [  'a1a', 'a2a', 'a3a', 'a4a', 'a5a', 'a6a', 'a7a', 'a8a', 'a9a', 'breast-cancer_scale', 
                            'cod-rna', 'covtype.libsvm.binary.bz2', 'diabetes_scale', 'fourclass_scale',
                            'german.numer_scale', 'gisette_scale.bz2', 'ijcnn1.bz2',
                            'liver-disorders_scale', 'phishing', 'sonar_scale', 'svmguide1', 'svmguide3',
                            'w1a', 'w2a', 'w3a', 'w4a', 'w5a', 'w6a', 'w7a', 'w8a'  ]:
        
        LIBSVM_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"

        base_libsvm_dir = os.path.join(save_path, "LIBSVM")
        if not os.path.exists(base_libsvm_dir):
            os.mkdir(base_libsvm_dir)

        if not os.path.exists(os.path.join(base_libsvm_dir, dataset_name + '_data')):
            
            url = parse.urljoin(LIBSVM_URL, dataset_name)
        
            response = requests.get(url, stream=True, verify=False)
            response.raise_for_status()  # Raises an error for bad status codes

            with open(os.path.join(base_libsvm_dir, dataset_name + '_data'), 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            if '.bz2' in dataset_name:
                with bz2.open(os.path.join(base_libsvm_dir, dataset_name + '_data'), 'rb') as f_in, open(os.path.join(base_libsvm_dir, dataset_name + '_data_uncompressed'), 'wb') as f_out:
                    f_out.write(f_in.read())

                os.remove(os.path.join(base_libsvm_dir, dataset_name + '_data'))
                os.rename(os.path.join(base_libsvm_dir, dataset_name + '_data_uncompressed'),
                          os.path.join(base_libsvm_dir, dataset_name + '_data'))
    
        X, y = load_svmlight_file(os.path.join(base_libsvm_dir, dataset_name + '_data'))
        
        labels = np.unique(y)
        y[y==labels[0]] = 0
        y[y==labels[1]] = 1
        
        X = X.toarray()

        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).float().view(-1,1)

        n_examples_per_class=5000

        indices = []

        for class_label in range(2):
            class_indices = np.where(y_tensor == class_label)[0]
            selected_indices = class_indices[:n_examples_per_class]
            indices.extend(selected_indices)

        X_tensor = X_tensor[indices]
        y_tensor = y_tensor[indices]

    dataset = TensorDataset(X_tensor, y_tensor)

    return DataLoader(dataset, batch_size=min(BATCH_SIZE, len(dataset)), shuffle=False, num_workers=NUM_WORKERS, drop_last=False, pin_memory=False),  X_tensor.shape[1], y_tensor.shape[1]


def UCI_download_and_unzip(url, base_uci_dir, dataset_name):
    request.urlretrieve(url, os.path.join(base_uci_dir, dataset_name + '.zip'))

    os.mkdir(os.path.join(base_uci_dir, dataset_name + '_data'))
    with zipfile.ZipFile(os.path.join(base_uci_dir, dataset_name + '.zip'), 'r') as zip_ref:
        zip_ref.extractall(os.path.join(base_uci_dir, dataset_name + '_data'))