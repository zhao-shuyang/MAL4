#!/usr/bin/python
__author__ = "Zhao Shuyang"
__email__ = "shuyang.zhao@tut.fi"

import numpy as np
import h5py
import sys

def cosine_similarity(v1, v2):
    #print (v1, v2)
    sim = np.dot(v1, v2)/np.sqrt(np.sum(v1**2))/np.sqrt(np.sum(v2**2))
    #print (sim)
    return sim 
    #return np.sqrt(np.sum((v1-v2)**2))
    
def euclidean_distance(v1, v2):
    return np.sqrt(np.sum((v1-v2)**2))

def dtw(s1, s2):
    n = len(s1)
    m = len(s2)
    DTW = np.zeros((n, m))
    
    for i in range(n):
       DTW[i, 0] = np.inf
    for i in range(m):
       DTW[0, i] = np.inf
    DTW[0, 0] = 0
    
    for i in range(n):
        for j in range(m):
            cost = 1 - cosine_similarity(s1[i], s2[j])
            DTW[i, j] = cost + np.min((DTW[i-1, j], DTW[i  , j-1],    DTW[i-1,j-1]))

    return DTW[n-1, m-1]


def gau_kl(pm, pv, qm, qv):
    """
    # Copyright (c) 2008 Carnegie Mellon University
    #
    # You may copy and modify this freely under the same terms as
    # Sphinx-III
    
    Divergence and distance measures for multivariate Gaussians and
    multinomial distributions.

    This module provides some functions for calculating divergence or
    distance measures between distributions, or between one distribution
    and a codebook of distributions.
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    Diagonal covariances are assumed.  Divergence is expressed in nats.
    """
    if (len(qm.shape) == 2):
        axis = 1
    else:
        axis = 0
        
    #Diagonal
    if len(pv .shape) == 1:
        

        # Determinants of diagonal covariances pv, qv
        dpv = pv.prod() + sys.float_info.min
        dqv = qv.prod(axis) + sys.float_info.min
    
        # Inverse of diagonal covariance qv
        iqv = 1./qv
        # Difference between means pm, qm
        diff = qm - pm

        return  (0.5 *
            (np.log(dqv / dpv)            # log |\Sigma_q| / |\Sigma_p|
             + np.sum(iqv * pv)          # + tr(\Sigma_q^{-1} * \Sigma_p)
             + np.sum(diff * iqv * diff) # + (\mu_q-\mu_p)^T\Sigma_q^{-1}(\mu_q-\mu_p)
             - len(pm)))                     # - N

    #Full
    elif len(pv .shape) == 2:
        
        # Determinants of diagonal covariances pv, qv
        dpv = np.linalg.det(pv)
        dqv = np.linalg.det(qv)
        # Inverse of diagonal covariance qv
        ipv = np.linalg.inv(pv)
        iqv = np.linalg.inv(qv)
        #print dpv, dqv, np.log(dqv / dpv) 

        # Difference between means pm, qm
        diff = qm - pm
        eps = 1e-60
        a = (dqv + eps)/(dpv + eps) + eps
        b = np.log(np.nan_to_num(a))
        return  0.5  *(
             np.log(dqv/dpv)                   # log |\Sigma_q| / |\Sigma_p|
             + np.trace(np.dot(iqv, pv))        # + tr(\Sigma_q^{-1} * \Sigma_p)
             + np.dot(np.dot(diff, iqv), diff) # + (\mu_q-\mu_p)^T\Sigma_q^{-1}(\mu_q-\mu_p)
             - len(pm))                     # - N
    
def GaussianKL(ft_hdf5, dist_hdf5):
    N = len(ft_hdf5.keys())
    dist_mat = np.zeros((N, N))
    file_list = []
    i, j = 0, 0
    for k_i in ft_hdf5.keys():
        print (i)
        ft_i = ft_hdf5[k_i][:]
        file_list.append(k_i)
        for k_j in ft_hdf5.keys():
            #print (k_i, k_j)
            if j == i:
                continue
            ft_j = f1_obj[k_j][:]            
            
            #dist_mat[i][j] = gau_kl(np.mean(ft_i, axis=0), np.std(ft_i, axis=0), np.mean(ft_j, axis=0), np.std(ft_j, axis=0)) + gau_kl(np.mean(ft_j, axis=0), np.std(ft_j, axis=0), np.mean(ft_i, axis=0), np.std(ft_i, axis=0))
            dist_mat[i][j] = gau_kl(np.mean(ft_i, axis=0), np.std(ft_i, axis=0), np.mean(ft_j, axis=0), np.std(ft_j, axis=0)) + gau_kl(np.mean(ft_j, axis=0), np.std(ft_j, axis=0), np.mean(ft_i, axis=0), np.std(ft_i, axis=0)) 
            #print (dist_mat[i][j]
            dist_mat[j][i] = dist_mat[i][j]
            j += 1
        i += 1
        j = 0
    print (dist_mat, dist_mat.shape)
    
    dist_hdf5["dist_mat"] = dist_mat

def GaussianKLDTW(ft_hdf5, dist_hdf5):
    N = len(ft_hdf5.keys())
    dist_mat_kl = np.zeros((N, N))
    dist_mat_dtw = np.zeros((N, N))
    file_list = []
    i, j = 0, 0
    for k_i in ft_hdf5.keys():
        ft_i = ft_hdf5[k_i][:]
        print (i, ft_i)
        file_list.append(k_i)
        for k_j in ft_hdf5.keys():
            if j == i:
                continue
            ft_j = f1_obj[k_j][:]            
            
            #dist_mat[i][j] = gau_kl(np.mean(ft_i, axis=0), np.std(ft_i, axis=0), np.mean(ft_j, axis=0), np.std(ft_j, axis=0)) + gau_kl(np.mean(ft_j, axis=0), np.std(ft_j, axis=0), np.mean(ft_i, axis=0), np.std(ft_i, axis=0))
            dist_mat_kl[i][j] = gau_kl(np.mean(ft_i, axis=0), np.std(ft_i, axis=0), np.mean(ft_j, axis=0), np.std(ft_j, axis=0)) + gau_kl(np.mean(ft_j, axis=0), np.std(ft_j, axis=0), np.mean(ft_i, axis=0), np.std(ft_i, axis=0))
            dist_mat_dtw[i][j] = dtw(ft_i, ft_j)
            #print (dist_mat[i][j]
            dist_mat_kl[j][i] = dist_mat_kl[i][j]
            dist_mat_dtw[j][i] = dist_mat_dtw[i][j]
            j += 1
        i += 1
        j = 0
    print (dist_mat, dist_mat.shape)
    
    dist_hdf5["dist_mat"] = dist_mat_kl/np.max(dist_mat_kl) + dist_mat_dtw/np.max(dist_mat_dtw)

def Cosine(ft_hdf5, dist_hdf5):
    N = len(ft_hdf5.keys())
    dist_mat = np.zeros((N, N))
    file_list = []
    i, j = 0, 0
    for k_i in ft_hdf5.keys():
        #print (i,j)
        ft_i = ft_hdf5[k_i][:]
        print (k_i, ft_i[:6], np.linalg.norm(ft_i)) #Visualize 6 dimensions of the feature
        #print (k_i)
        file_list.append(k_i)
        for k_j in ft_hdf5.keys():
            if j == i:
                continue
            #print (k_i, k_j)
            #print (ft_i, ft_j)
            ft_j = ft_hdf5[k_j][:]
            dist_mat[i][j] = 1 - cosine_similarity(ft_i, ft_j)
            #dist_mat[i][j] = euclidean_distance(ft_i, ft_j)
            dist_mat[j][i] = dist_mat[i][j]
            
            j += 1
        i += 1
        j = 0
    print (dist_mat, dist_mat.shape)
    
    dist_hdf5["dist_mat"] = dist_mat

def Dist(ft_hdf5, dist_hdf5, type='l2'):
    N = len(ft_hdf5.keys())
    dist_mat = np.zeros((N, N))
    file_list = []
    i, j = 0, 0
    for k_i in ft_hdf5.keys():
        #print (i,j)
        ft_i = ft_hdf5[k_i][:]
        print (k_i, ft_i[:6], np.linalg.norm(ft_i))
        #print (k_i)
        file_list.append(k_i)
        for k_j in ft_hdf5.keys():
            if j == i:
                continue
            #print (k_i, k_j)
            #print (ft_i, ft_j)
            ft_j = ft_hdf5[k_j][:]
            #dist_mat[i][j] = 1 - cosine_similarity(ft_i, ft_j)
            dist_mat[i][j] = np.linalg.norm(ft_i - ft_j)
            #dist_mat[i][j] = euclidean_distance(ft_i, ft_j)
            dist_mat[j][i] = dist_mat[i][j]
            
            j += 1
        i += 1
        j = 0
    print (dist_mat, dist_mat.shape)
    
    dist_hdf5["dist_mat"] = dist_mat

def Dist_gpu(ft_hdf5, dist_hdf5, dist_type='cosine'):
    import torch
    import torch.nn.functional as Fx
    usegpu = True
    device = torch.device("cuda" if usegpu else "cpu")
    #torch.cuda.set_device(1)

    data = ft_hdf5['max'][:]
    N,D = data.shape
    #dist_mat = np.zeros((N, N))
    torch_data = torch.from_numpy(data).float()
    dist_mat = 1-torch.mm(torch_data, torch_data.transpose(0,1)).cpu().numpy()/torch.ger(torch.norm(torch_data,dim=1), torch.norm(torch_data,dim=1))

    """
    for i in range(N):
        #print (i,j)
        ft_i = data[i]
        #print (ft_i[:6], np.linalg.norm(ft_i))

        fi_torch = torch.from_numpy(ft_i).float().to(device)
        one_torch = torch.ones(N).float().to(device)
        torch_data1 = torch.ger(one_torch, fi_torch).to(device)
        torch_data2 = torch.from_numpy(data).float().to(device)

        if dist_type == 'cosine':
            #dist_mat[i] = one_torch - Fx.cosine_similarity(torch_data1, torch_data2)#.cpu().numpy()
            dist_mat[i] = -torch.mm(torch_data1, torch_data2.transpose(0,1))
        #print (dist_mat)
    """
    print (dist_mat, dist_mat.shape)
    #dist_mat = dist_mat.T + dist_mat
    dist_hdf5["dist_mat"] = dist_mat
    
if __name__ == '__main__':
    
    #f1_obj = h5py.File('data/ESC10_audiosetT.hdf5','r')
    #f2_obj = h5py.File('data/ESC10_dist_audiosetT.hdf5','w')
    f1_obj = h5py.File('data/ESC10_d1226.hdf5','r')
    f2_obj = h5py.File('data/ESC10_dist_test.hdf5','w')
    Cosine(f1_obj, f2_obj)
    """
    f1_obj = h5py.File('data/ESC10_MFCC.hdf5','r')
    f2_obj = h5py.File('data/ESC10_dist_MFCC.hdf5','w')
    GaussianKL(f1_obj, f2_obj)
    """
