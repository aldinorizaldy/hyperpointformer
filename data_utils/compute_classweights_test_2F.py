import numpy as np
label_1 = np.load('../data_test_1/Houston_2018_sem_seg_data/Test_Fold_1.npy')[:,-1]
label_2 = np.load('../data_test_2/Houston_2018_sem_seg_data/Test_Fold_2.npy')[:,-1]

def compute_classweights_1():
    class_weights = 1 - (np.unique(label_1, return_counts=True)[1]/len(label_1))
    print('Labels are: ', np.unique(label_1))
    n_class = len(np.unique(label_1))
    return class_weights, n_class

def compute_classweights_2():
    class_weights = 1 - (np.unique(label_2, return_counts=True)[1]/len(label_2))
    print('Labels are: ', np.unique(label_2))
    n_class = len(np.unique(label_2))
    return class_weights, n_class
