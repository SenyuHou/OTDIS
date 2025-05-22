import numpy as np
import utils
import os
import numpy as np
import torch
import torchvision
from math import inf
from scipy import stats
from torchvision.transforms import transforms
import torch.nn.functional as F
import torch.nn as nn



# flip clean labels to noisy labels
# train set and val set split
def dataset_split(train_images, train_labels, dataset='mnist', noise_type='symmetric', noise_rate=0.5, split_per=0.9, random_seed=1, num_classes=10):
    
    clean_train_labels = train_labels[:, np.newaxis]
    
    if noise_type == 'symmetric':
         noisy_labels, real_noise_rate, transition_matrix = utils.noisify_multiclass_symmetric(clean_train_labels, 
                                                                                               noise=noise_rate, 
                                                                                               random_state=random_seed, 
                                                                                               nb_classes=num_classes)
    if noise_type == 'asymmetric' and dataset == 'mnist':
        noisy_labels, real_noise_rate, transition_matrix = utils.noisify_multiclass_asymmetric_mnist(clean_train_labels,
                                                                                                    noise=noise_rate,
                                                                                                    random_state=random_seed,
                                                                                                    nb_classes=num_classes)
  
    
    if noise_type == 'asymmetric' and dataset == 'cifar10':
        noisy_labels, real_noise_rate, transition_matrix = utils.noisify_multiclass_asymmetric_cifar10(clean_train_labels,
                                                                                                      noise=noise_rate,
                                                                                                      random_state=random_seed,
                                                                                                      nb_classes=num_classes)
        
    if noise_type == 'asymmetric' and dataset == 'cifar100':
        noisy_labels, real_noise_rate, transition_matrix = utils.noisify_multiclass_asymmetric_cifar100(clean_train_labels,
                                                                                                       noise=noise_rate,
                                                                                                       random_state=random_seed,
                                                                                                       nb_classes=num_classes)
        

    noisy_labels = noisy_labels.squeeze()
    num_samples = int(noisy_labels.shape[0])
    np.random.seed(random_seed)
    train_set_index = np.random.choice(num_samples, int(num_samples*split_per), replace=False)
    index = np.arange(train_images.shape[0])
    val_set_index = np.delete(index, train_set_index)

    train_set, val_set = train_images[train_set_index, :], train_images[val_set_index, :]
    train_labels, val_labels = noisy_labels[train_set_index], noisy_labels[val_set_index]

    return train_set, val_set, train_labels, val_labels

def dataset_split_without_noise(train_images, train_labels, noise_rate, split_per=0.9, random_seed=1, num_class=196):
    total_labels = train_labels[:, np.newaxis]
    num_samples = int(total_labels.shape[0])
    np.random.seed(random_seed)
    train_set_index = np.random.choice(num_samples, int(num_samples * split_per), replace=False)
    index = np.arange(train_images.shape[0])
    val_set_index = np.delete(index, train_set_index)
    print(train_images.shape)
    train_set, val_set = train_images[train_set_index], train_images[val_set_index]
    train_labels, val_labels = total_labels[train_set_index], total_labels[val_set_index]

    return train_set, val_set, train_labels.squeeze(), val_labels.squeeze()

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets, _ in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    print(mean)
    print(std)
    return mean, std


def transform_target(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()
    return target  