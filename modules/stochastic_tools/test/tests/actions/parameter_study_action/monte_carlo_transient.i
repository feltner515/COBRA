[ParameterStudy]
  input = sub_transient.i
  parameters = 'BCs/left/value BCs/right/value'
  quantities_of_interest = 'average/value'

  sampling_type = monte-carlo
  num_samples = 3
  distributions = 'uniform uniform'
  uniform_lower_bound = '100 1'
  uniform_upper_bound = '200 2'

  output_type = 'csv'
[]
