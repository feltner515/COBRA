[StochasticTools]
[]

[Distributions]
  [uniform]
    type = Uniform
    lower_bound = 5
    upper_bound = 10
  []
[]

[Samplers]
  [sample]
    type = MonteCarlo
    num_rows = 3
    distributions = 'uniform'
    execute_on = 'PRE_MULTIAPP_SETUP'
  []
[]

[MultiApps]
  [sub]
    type = SamplerFullSolveMultiApp
    sampler = sample
    input_files = 'sub.i'
    mode = batch-reset
  []
[]

[Transfers]
  [data]
    type = SamplerPostprocessorTransfer
    from_multi_app = sub
    sampler = sample
    to_vector_postprocessor = storage
    from_postprocessor = size
  []
[]

[VectorPostprocessors]
  [storage]
    type = StochasticResults
  []
[]

[Controls]
  [cmdline]
    type = MultiAppSamplerControl
    multi_app = sub
    sampler = sample
    param_names = 'Mesh/xmax'
  []
[]

[Outputs]
  [out]
    type = CSV
    execute_on = FINAL
  []
[]
