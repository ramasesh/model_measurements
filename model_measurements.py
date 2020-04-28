import torch
import torch.nn as nn
import measurement_utils

def L1_norm(mat1):
  return mat1.norm(p=1).item()

def Linfty_norm(mat1):
  return mat1.norm(p=float('inf')).item()

def spectral_norm(mat1):
  if len(mat1.size()) == 2:
    _, S, _ = mat1.svd()
    return torch.max(S).item()
  elif len(mat1.size()) == 1:
    return mat1.norm(p=2).item()

def L2_norm(mat1):
  return mat1.norm(p=2).item()

def Euclidean_distance(mat1, mat2):
  return L2_norm(mat1 - mat2)

measurement_functions = {'L1': L1_norm,
                         'Linf': Linfty_norm,
                         'Spectral': spectral_norm,
                         'L2': L2_norm,
                         'L2_to_init': Euclidean_distance}
