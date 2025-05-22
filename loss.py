from __future__ import print_function
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from numpy.testing import assert_array_almost_equal
from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import GaussianMixture
import warnings

warnings.filterwarnings('ignore')

def ot_process(loss):
    loss = loss.detach().cpu().numpy()
    dim_1, dim_2 = loss.shape[0], loss.shape[1]
    if dim_2 >= 5:
        lof = LocalOutlierFactor(n_neighbors=2, algorithm='auto', contamination=0.1, n_jobs=-1)
        t_o = []
        for i in range(dim_1):
            loss_single = loss[i].reshape((-1, 1))
            outlier_predict_bool = lof.fit_predict(loss_single)
            outlier_number = np.sum(outlier_predict_bool>0)
            loss_single[outlier_predict_bool==1] = 0.
            loss[i,:] = loss_single.transpose()
            t_o.append(outlier_number)
        t_o = np.array(t_o).reshape((dim_1, 1))
    else:
        t_o = np.zeros((dim_1, 1))
    loss = torch.from_numpy(loss).cuda().float()
    return loss, t_o


def get_otdis_loss(epoch, before_loss_1, before_loss_2, sn_1, sn_2, y_1, y_2, t, ind, noise_or_not, co_lambda, loss_bound, gamma):
    s = torch.tensor(epoch + 1).float()
    co_lambda = torch.tensor(co_lambda).float()
    loss_bound = torch.tensor(loss_bound).float()

    # ---------- Model 1 ----------
    loss_1 = F.cross_entropy(y_1, t, reduction='none')
    before_and_loss_1 = torch.cat((torch.from_numpy(before_loss_1).cuda().float(), loss_1.unsqueeze(1)), 1)
    before_and_loss_1_hard, t_o_1 = ot_process(before_and_loss_1)
    loss_1_mean = torch.mean(before_and_loss_1_hard, dim=1)

    # ---------- Model 2 ----------
    loss_2 = F.cross_entropy(y_2, t, reduction='none')
    before_and_loss_2 = torch.cat((torch.from_numpy(before_loss_2).cuda().float(), loss_2.unsqueeze(1)), 1)
    before_and_loss_2_hard, t_o_2 = ot_process(before_and_loss_2)
    loss_2_mean = torch.mean(before_and_loss_2_hard, dim=1)

    # ---------- GMM Co-Divide ----------
    def gmm_separate(loss_tensor):
        loss_np = loss_tensor.detach().cpu().view(-1, 1).numpy()
        gmm = GaussianMixture(n_components=2, random_state=0).fit(loss_np)
        pred = gmm.predict(loss_np)
        means = gmm.means_.reshape(-1)
        clean_component = np.argmin(means)
        noisy_component = np.argmax(means)
        clean_idx = np.where(pred == clean_component)[0]
        noisy_idx = np.where(pred == noisy_component)[0]
        return clean_idx, noisy_idx

    clean_1, noise_1 = gmm_separate(loss_1_mean)
    clean_2, noise_2 = gmm_separate(loss_2_mean)

    # ---------- Pure ratio ----------
    pure_ratio_1 = np.sum(noise_or_not[ind[clean_1]]) / (len(clean_1) + 1e-6)
    pure_ratio_2 = np.sum(noise_or_not[ind[clean_2]]) / (len(clean_2) + 1e-6)

    # ---------- L_PL ----------
    ce_1 = F.cross_entropy(y_1[clean_2], t[clean_2])
    ce_2 = F.cross_entropy(y_2[clean_1], t[clean_1])

    # ---------- L_NL ----------
    y_1_softmax = F.softmax(y_1[noise_2], dim=1)
    y_2_softmax = F.softmax(y_2[noise_1], dim=1)
    neg_1 = F.kl_div(torch.log(y_1_softmax + 1e-8), torch.full_like(y_1_softmax, 1.0 / y_1_softmax.size(1)), reduction='batchmean')
    neg_2 = F.kl_div(torch.log(y_2_softmax + 1e-8), torch.full_like(y_2_softmax, 1.0 / y_2_softmax.size(1)), reduction='batchmean')

    loss_1_total = ce_1 + gamma * neg_1
    loss_2_total = ce_2 + gamma * neg_2

    return (
        loss_1_total,
        loss_2_total,
        pure_ratio_1,
        pure_ratio_2,
        clean_1.tolist(),
        clean_2.tolist(),
        loss_1,
        loss_2
    )




