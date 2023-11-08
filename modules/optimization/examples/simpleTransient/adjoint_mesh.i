[Mesh]
  [gmg]
    type = GeneratedMeshGenerator
    dim = 2
    nx = 10
    ny = 10
    xmin = -1
    xmax = 1
    ymin = -1
    ymax = 1
  []
[]

[Variables]
  [u]
  []
[]

[Kernels]
  [dt]
    type = TimeDerivative
    variable = u
  []
  [diff]
    type = Diffusion
    variable = u
  []
[]

[BCs]
  [dirichlet]
    type = DirichletBC
    variable = u
    boundary = 'left right top bottom'
    value = 0
  []
[]

[Reporters]
  [measured_data]
    type = OptimizationData
    measurement_file = mms_data.csv
    file_xcoord = x
    file_ycoord = y
    file_zcoord = z
    file_time = t
    file_value = u
  []
  [src_values]
    type = ConstantReporter
    real_vector_names = 'time values'
    real_vector_values = '0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0;
                          0' # dummy
  []
[]

[DiracKernels]
  [misfit]
    type = ReporterTimePointSource
    variable = u
    value_name = measured_data/misfit_values
    x_coord_name = measured_data/measurement_xcoord
    y_coord_name = measured_data/measurement_ycoord
    z_coord_name = measured_data/measurement_zcoord
    time_name = measured_data/measurement_time
    reverse_time_end = 1
  []
[]

[Functions]
  [source]
    type = ParameterMeshFunction
    exodus_mesh = source_mesh_in.e
    time_name = src_values/time
    parameter_name = src_values/values
  []
[]

[VectorPostprocessors]
  [adjoint]
    type = ElementOptimizationSourceFunctionInnerProduct
    variable = u
    function = source
    reverse_time_end = 1
  []
[]

[Executioner]
  type = Transient

  num_steps = 100
  end_time = 1

  solve_type = NEWTON
  petsc_options_iname = '-pc_type -pc_hypre_type'
  petsc_options_value = 'hypre boomeramg'
[]

[Outputs]
  console = false
[]
