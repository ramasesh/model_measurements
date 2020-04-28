import torch
import torch.nn as nn
from meters import AverageMeter
import model_measurements as mm

possible_measurements = {'logit_sum': measure_logit_sum,
                         'correct_logit': measure_correct_logit,
                         'logit_margin': measure_logit_margin,
                         'highest_incorrect_logit': measure_highest_incorrect_logit,
                         'accuracy': measure_accuracy,
                         'cross_entropy': measure_cross_entropy}


def measure_logit_sum(model_outputs, model_parameters, targets, initial_parameters, name=None, msmt_type=None):
  return torch.sum(model_outputs, dim=1)

def measure_correct_logit(model_outputs, model_parameters, targets, initial_parameters, name=None, msmt_type=None):
  return model_outputs[torch.arange(model_outputs.size(0)), targets]

def measure_logit_margin(model_outputs, model_parameters, targets, initial_parameters, name=None, msmt_type=None):
  return measure_correct_logit(model_outputs, model_parameters, targets, initial_parameters) - measure_highest_incorrect_logit(model_outputs, model_parameters, targets, initial_parameters)

def measure_highest_incorrect_logit(model_outputs, model_parameters, targets, initial_parameters, name=None, msmt_type=None):
  " sets correct logit to -infinity and takes the highest logit "
  negative_inf = -1*float('inf')
  cloned_outputs = model_outputs.clone()
  for i in range(cloned_outputs.size(0)):
    cloned_outputs[i, targets[i]] = negative_inf

  maxes, _ = torch.max(cloned_outputs, dim=1)
  return maxes

def measure_accuracy(model_outputs, model_parameters, targets, initial_parameters, name=None, msmt_type=None):
  _, preds = torch.max(model_outputs, dim=1)
  return preds.eq(targets).float()

def measure_cross_entropy(model_outputs, model_parameters, targets, initial_parameters, name=None, msmt_type=None):
  loss = nn.CrossEntropyLoss(reduction='none')
  return loss(model_outputs, targets)

### Measurement functions ###
def measure_L2_norm(model_outputs, current_parameters, targets, initial_parameters, name, msmt_type='param'):
  value = extract_named_parameter(current_parameters, name, msmt_type)
  fun = mm.L2_norm
  return apply_correctly(fun, msmt_type, value)


def measure_L1_norm(model_outputs, current_parameters, targets, initial_parameters, name, msmt_type='param'):
  value = extract_named_parameter(current_parameters, name, msmt_type)
  fun = mm.L1_norm
  return apply_correctly(fun, msmt_type, value)

def measure_spectral_norm(model_outputs, current_parameters, targets, initial_parameters, name, msmt_type='param'):
  value = extract_named_parameter(current_parameters, name, msmt_type)
  fun = mm.spectral_norm
  return apply_correctly(fun, msmt_type, value)

def measure_Linfty_norm(model_outputs, current_parameters, targets, initial_parameters, name, msmt_type='param'):
  value = extract_named_parameter(current_parameters, name, msmt_type)
  fun = mm.Linfty_norm
  return apply_correctly(fun, msmt_type, value)

def measure_L2toInit(model_outputs, current_parameters, targets, initial_parameters, name, msmt_type='param'):
  current_value = extract_named_parameter(current_parameters, name, msmt_type)
  init_value = extract_named_parameter(initial_parameters, name, msmt_type)
  fun = mm.Euclidean_distance
  return apply_correctly_2arg(fun, msmt_type, current_value, init_value)

def extract_named_parameter(parameters, name, msmt_type):
  for p in parameters:
    print(p[0])
    if p[0] == name:
      param = p[1]
      break
  else:
    raise Exception('Did not find a match')
  if msmt_type == 'grad':
    return p[1].grad.data
  elif msmt_type == 'param':
    return p[1].data

def apply_correctly(fun, msmt_type, value):
  if msmt_type == 'param':
    return fun(value)
  elif msmt_type == 'grad':
    if len(value.shape) == 1:
      return torch.Tensor([fun(value)])
    else:
      return torch.Tensor([fun(ti) for ti in value])

def apply_correctly_2arg(fun, msmt_type, value1, value2):
  if msmt_type == 'param':
    return fun(value1, value2)
  elif msmt_type == 'grad':
    if len(value1.shape) == 1:
      return torch.Tensor([fun(value1, value2)])
    else:
      return torch.Tensor([fun(value1[i], value2[i]) for i in range(len(value1))])
