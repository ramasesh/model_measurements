import torch
import torch.nn as nn
import numpy as np

import model_measurements as mm
import output_measurements as om
import measurement_utils as mu

input_dim = 10
n_hidden = 20
n_outputs = 2
dataset_size = 50
batch_size = 5
LinearTest = nn.Sequential(nn.Linear(input_dim, n_hidden),
                           nn.Linear(n_hidden, n_outputs))

module_names = ['0.weight', '0.bias', '1.weight', '1.bias']
outputs_to_measure = om.possible_measurements.keys()

test_data = torch.randn([dataset_size, input_dim])
test_labels = torch.randint(low=0,
                            high=n_outputs,
                            size=[dataset_size])

test_dataset = list(zip(test_data, test_labels))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
grad_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

outputs_to_measure = ['logit_sum', 'correct_logit', 'logit_margin', 'highest_incorrect_logit', 'accuracy', 'cross_entropy']
characteristics_to_measure = ['L2Norm', 'L1Norm', 'SpectralNorm', 'LinftyNorm', 'L2toInit']

measured_cumulants = mu.measure_on_dataset(outputs_to_measure,
                                           LinearTest,
                                           test_loader,
                                           torch.device('cpu'),
                                           LinearTest.parameters())

measured_characteristics = mu.measure_model_characteristics(characteristics_to_measure,
                                                            LinearTest,
                                                            module_names,
                                                            LinearTest.named_parameters())


measured_characteristics_internal = mu.measure_characteristic_on_dataset(characteristics_to_measure,
                                                 LinearTest,
                                                 grad_test_loader,
                                                 torch.device('cpu'),
                                                 module_names,
                                                 list(LinearTest.named_parameters()))
