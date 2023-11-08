[StochasticTools]
[]

[Distributions]
  [k_dist]
    type = Normal
    mean = 5
    standard_deviation = 2
  []
  [L_dist]
    type = Normal
    mean = 0.03
    standard_deviation = 0.01
  []
[]

[Samplers]
  [sample]
    type = LatinHypercube
    num_rows = 10
    distributions = 'k_dist L_dist'
    execute_on = PRE_MULTIAPP_SETUP
    min_procs_per_row = 2
  []
[]

[GlobalParams]
  sampler = sample
[]

[MultiApps]
  [sub]
    type = SamplerFullSolveMultiApp
    input_files = sub_vector.i
    mode = batch-reset
    execute_on = initial
    min_procs_per_app = 2
  []
[]

[Controls]
  [cmdline]
    type = MultiAppSamplerControl
    multi_app = sub
    param_names = 'Materials/conductivity/prop_values L'
  []
[]

[Transfers]
  [data]
    type = SamplerReporterTransfer
    from_multi_app = sub
    stochastic_reporter = results
    from_reporter = 'T_vec/T T_vec/x'
  []
[]

[Reporters]
  [results]
    type = StochasticReporter
    outputs = none
  []
  [eval]
    type = EvaluateSurrogate
    model = pr_surrogate
    response_type = vector_real
    parallel_type = ROOT
    execute_on = timestep_end
  []
[]

[Trainers]
  [pr]
    type = PolynomialRegressionTrainer
    regression_type = ols
    max_degree = 2
    response = results/data:T_vec:T
    response_type = vector_real
    execute_on = initial
  []
[]

[Surrogates]
  [pr_surrogate]
    type = PolynomialRegressionSurrogate
    trainer = pr
  []
[]

[Outputs]
  [out]
    type = JSON
    execute_on = timestep_end
  []
[]
