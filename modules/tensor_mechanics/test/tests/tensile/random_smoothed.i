# Plasticity models:
# Smoothed tensile with strength = 1MPa
#
# Lame lambda = 1GPa.  Lame mu = 1.3GPa
#
# A line of elements is perturbed randomly, and return to the yield surface at each quadpoint is checked

[Mesh]
  type = GeneratedMesh
  dim = 3
  nx = 1000
  ny = 1234
  nz = 1
  xmin = 0
  xmax = 1000
  ymin = 0
  ymax = 1234
  zmin = 0
  zmax = 1
[]

[GlobalParams]
  displacements = 'disp_x disp_y disp_z'
[]

[Modules/TensorMechanics/Master]
  [./all]
    add_variables = true
    incremental = true
    strain = finite
    generate_output = 'stress_xx stress_xy stress_xz stress_yy stress_yz stress_zz'
  [../]
[]

[ICs]
  [./x]
    type = RandomIC
    min = -0.1
    max = 0.1
    variable = disp_x
  [../]
  [./y]
    type = RandomIC
    min = -0.1
    max = 0.1
    variable = disp_y
  [../]
  [./z]
    type = RandomIC
    min = -0.1
    max = 0.1
    variable = disp_z
  [../]
[]

[BCs]
  [./x]
    type = FunctionDirichletBC
    variable = disp_x
    boundary = 'front back'
    function = '0'
  [../]
  [./y]
    type = FunctionDirichletBC
    variable = disp_y
    boundary = 'front back'
    function = '0'
  [../]
  [./z]
    type = FunctionDirichletBC
    variable = disp_z
    boundary = 'front back'
    function = '0'
  [../]
[]

[AuxVariables]
  [./f0]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./int0]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./iter]
    order = CONSTANT
    family = MONOMIAL
  [../]
[]

[AuxKernels]
  [./f0]
    type = MaterialStdVectorAux
    property = plastic_yield_function
    index = 0
    variable = f0
  [../]
  [./int0]
    type = MaterialStdVectorAux
    property = plastic_internal_parameter
    factor = 1E6
    index = 0
    variable = int0
  [../]
  [./iter]
    type = MaterialRealAux
    property = plastic_NR_iterations
    variable = iter
  [../]
[]

[Postprocessors]
  [./tot_iters]
    type = ElementIntegralMaterialProperty
    mat_prop = plastic_NR_iterations
    outputs = console
  [../]
  [./raw_f0]
    type = ElementExtremeValue
    variable = f0
    outputs = console
  [../]
  [./iter]
    type = ElementExtremeValue
    variable = iter
    outputs = console
  [../]
  [./f0]
    type = FunctionValuePostprocessor
    function = should_be_zero0_fcn
  [../]
[]

[Functions]
  [./should_be_zero0_fcn]
    type = ParsedFunction
    expression = 'if(a<1E-1,0,a)'
    symbol_names = 'a'
    symbol_values = 'raw_f0'
  [../]
[]

[UserObjects]
  [./ts]
    type = TensorMechanicsHardeningConstant
    value = 1E6
  [../]
  [./tensile]
    type = TensorMechanicsPlasticTensile
    tensile_strength = ts
    tensile_tip_smoother = 1E5
    yield_function_tolerance = 1.0E-1
    internal_constraint_tolerance = 1.0E-7
  [../]
[]

[Materials]
  [./elasticity_tensor]
    type = ComputeElasticityTensor
    block = 0
    fill_method = symmetric_isotropic
    C_ijkl = '1E9 1.3E9'
  [../]
  [./multi]
    type = ComputeMultiPlasticityStress
    block = 0
    ep_plastic_tolerance = 1E-7
    plastic_models = 'tensile'
    max_NR_iterations = 20
    min_stepsize = 1E-4
    max_stepsize_for_dumb = 1E-3
    debug_fspb = crash
    debug_jac_at_stress = '10 0 0 0 10 0 0 0 10'
    debug_jac_at_pm = '1 1 1 1'
    debug_jac_at_intnl = '1 1 1 1'
    debug_stress_change = 1E1
    debug_pm_change = '1E-6 1E-6 1E-6 1E-6'
    debug_intnl_change = '1E-6 1E-6 1E-6 1E-6'
  [../]
[]


[Executioner]
  end_time = 1
  dt = 1
  type = Transient
[]


[Outputs]
  file_base = random_smoothed
  exodus = false
  [./csv]
    type = CSV
    [../]
[]
