import torch
import torch.nn as nn
import collections
from meters import MomentAggregator, get_cumulants
import output_measurements as om

measurement_functions = {'L2Norm': om.measure_L2_norm,
                         'L1Norm': om.measure_L1_norm,
                         'SpectralNorm': om.measure_spectral_norm,
                         'LinftyNorm': om.measure_Linfty_norm,
                         'L2toInit': om.measure_L2toInit,
                         'logit_sum': om.measure_logit_sum,
                         'correct_logit': om.measure_correct_logit,
                         'logit_margin': om.measure_logit_margin,
                         'highest_incorrect_logit': om.measure_highest_incorrect_logit,
                         'accuracy': om.measure_accuracy,
                         'cross_entropy': om.measure_cross_entropy}

def map_nested_dicts(ob, func):
  """ applies func to all the leaves in ob """
  if isinstance(ob, collections.Mapping):
    return {k: map_nested_dicts(v, func) for k, v in ob.items()}
  else:
    return func(ob)

def summarize(sample_msmts, reduction='mean'):
  if reduction == 'mean':
    return torch.mean(sample_msmts).item()
  elif reduction == 'square':
    return torch.mean(torch.mul(sample_msmts, sample_msmts)).item()

def extract_classifier(model):
  """ this assumes that the final element of model.children() is the classifier """
  return list(model.children())[-1]

def classifier_params(model):
  classifier = extract_classifier(model)
  param_types = ['weight', 'bias']

  def valid_attr(a):
    return hasattr(classifier, a) and getattr(classifier,a) is not None

  return {p: getattr(classifier, p).data for p in filter(valid_attr, param_types)}

def zero_model_gradients(model):
  for p in model.parameters():
    if p.grad is not None:
      p.grad.data.zero_()

def measure_model_characteristics(msmts_to_make, model, parameter_names, initial_parameters):

  msmts = {}
  for parameter_name in parameter_names:
    msmts[parameter_name] = {}
    for msmt in msmts_to_make:
      print(f'Measuring {msmt} of {parameter_name}')
      msmts[parameter_name][msmt] = measurement_functions[msmt](None,
                                               model.named_parameters(),
                                               None,
                                               initial_parameters,
                                               parameter_name,
                                               msmt_type='param')

  return msmts

def measure_on_dataset(msmts_to_make, model, dataloader, device, initial_parameters):

  model.eval()
  model = model.to(device)

  measured_cumulants = {msmt: MomentAggregator(max_moment=2) for msmt in msmts_to_make}

  for step, (data, targets) in enumerate(dataloader):

    zero_model_gradients(model)

    data = data.to(device)
    targets = targets.to(device)
    outputs = model(data)

    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    loss = loss_fn(outputs, targets)
    loss.backward()

    for msmt in msmts_to_make:
      print(f'Measuring {msmt}')
      sample_wise_msmts = measurement_functions[msmt](outputs,
                                                      model.parameters(),
                                                      targets,
                                                      initial_parameters,
                                                      None,
                                                      None)
      measured_cumulants[msmt].update(sample_wise_msmts)

  measured_cumulants = map_nested_dicts(measured_cumulants,
                                        get_cumulants)

  return measured_cumulants

def measure_characteristic_on_dataset(msmts_to_make, model, dataloader, device, parameters_to_study, initial_parameters):

  model.eval()
  model = model.to(device)

  measured_cumulants = {c: {m: MomentAggregator(max_moment=2) for m in msmts_to_make}
                        for c in parameters_to_study}

  for step, (data, targets) in enumerate(dataloader):

    zero_model_gradients(model)

    data = data.to(device)
    targets = targets.to(device)
    outputs = model(data)

    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    loss = loss_fn(outputs, targets)
    loss.backward()

    for msmt in msmts_to_make:
      for parameter_name in parameters_to_study:
        print(f'Measuring {msmt} on parameter {parameter_name}')
        sample_wise_msmts = measurement_functions[msmt](None,
                                                 model.named_parameters(),
                                                 None,
                                                 initial_parameters,
                                                 parameter_name,
                                                 msmt_type='grad')
        measured_cumulants[parameter_name][msmt].update(sample_wise_msmts)

  measured_cumulants = map_nested_dicts(measured_cumulants,
                                        get_cumulants)

  return measured_cumulants
