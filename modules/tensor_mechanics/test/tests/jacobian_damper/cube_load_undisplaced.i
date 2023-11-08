[Mesh]
  type = GeneratedMesh
  dim = 3
  nx = 1
  ny = 1
  nz = 1
[]

[Functions]
  [top_pull]
    type = PiecewiseLinear
    x = '0 1     2'
    y = '0 0.025 0.05'
  []
[]

[Modules]
  [TensorMechanics]
    [Master]
      displacements = 'disp_x disp_y disp_z'
      [all]
        displacements = 'disp_x disp_y disp_z'
        strain = FINITE
        add_variables = true
        new_system = true
        formulation = TOTAL
      []
    []
  []
[]

[BCs]
  [y_pull_function]
    type = FunctionDirichletBC
    variable = disp_y
    boundary = 3
    function = top_pull
    preset = true
  []
  [x_bot]
    type = DirichletBC
    variable = disp_x
    boundary = 4
    value = 0.0
  []
  [y_bot]
    type = DirichletBC
    variable = disp_y
    boundary = 1
    value = 0.0
  []
  [z_bot]
    type = DirichletBC
    variable = disp_z
    boundary = 0
    value = 0.0
  []
[]

[Materials]
  [elasticity_tensor]
    type = ComputeIsotropicElasticityTensor
    poissons_ratio = 0.3
    youngs_modulus = 2e5
  []
  [stress]
    type = ComputeLagrangianLinearElasticStress
    large_kinematics = true
  []
[]

[Dampers]
  [ejd]
    type = ReferenceElementJacobianDamper
    max_increment = 0.002
    displacements = 'disp_x disp_y disp_z'
  []
[]

[Executioner]
  type = Transient

  solve_type = NEWTON

  petsc_options_iname = '-pc_type'
  petsc_options_value = 'lu'

  line_search = 'none'

  nl_rel_tol = 1e-12
  nl_abs_tol = 1e-10
  start_time = 0.0
  end_time = 2
  dt = 1
[]

[Outputs]
  exodus = true
  print_linear_residuals = false
[]
