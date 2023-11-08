# Test for central difference integration for a 1D element
# Consistent mass matrix

[Mesh]
  type = GeneratedMesh
  xmin = 0
  xmax = 10
  nx = 5
  dim = 1
[]

[Variables]
  [./disp_x]
    order = FIRST
    family = LAGRANGE
  [../]
[]

[AuxVariables]
  [./accel_x]
  [../]
  [./vel_x]
  [../]
[]

[AuxKernels]
  [./accel_x]
    type = TestNewmarkTI
    variable = accel_x
    displacement = disp_x
    first = false
  [../]
  [./vel_x]
    type = TestNewmarkTI
    variable = vel_x
    displacement = disp_x
  [../]
[]

[Kernels]
  [./DynamicTensorMechanics]
    displacements = 'disp_x'
  [../]
  [./inertia_x]
    type = InertialForce
    variable = disp_x
  [../]
[]

[NodalKernels]
  [./force_x]
    type = UserForcingFunctionNodalKernel
    variable = disp_x
    boundary = right
    function = force_x
  [../]
[]

[Functions]
  [./force_x]
    type = PiecewiseLinear
    x = '0.0 1.0 2.0 3.0 4.0' # time
    y = '0.0 1.0 0.0 -1.0 0.0'  # force
    scale_factor = 1e3
  [../]
[]

[BCs]
  [./fixx1]
    type = DirichletBC
    variable = disp_x
    boundary = left
    value = 0.0
  [../]
[]

[Materials]
  [./elasticity_tensor_block]
    type = ComputeIsotropicElasticityTensor
    youngs_modulus = 1e6
    poissons_ratio = 0.25
    block = 0
  [../]
  [./strain_block]
    type = ComputeIncrementalSmallStrain
    block = 0
    displacements = 'disp_x'
    implicit = false
  [../]
  [./stress_block]
    type = ComputeFiniteStrainElasticStress
    block = 0
  [../]
  [./density]
    type = GenericConstantMaterial
    block = 0
    prop_names = density
    prop_values = 2500
  [../]
[]

[Executioner]
  type = Transient
  start_time = -0.005
  end_time = 0.1
  dt = 0.005
  timestep_tolerance = 1e-6
  l_tol = 1e-10
  [./TimeIntegrator]
    type = CentralDifference
  [../]
[]

[Postprocessors]
  [./disp_x]
    type = NodalVariableValue
    nodeid = 1
    variable = disp_x
  [../]
  [./vel_x]
    type = NodalVariableValue
    nodeid = 1
    variable = vel_x
  [../]
  [./accel_x]
    type = NodalVariableValue
    nodeid = 1
    variable = accel_x
  [../]
[]

[Outputs]
  exodus = false
  csv = true
  perf_graph = false
[]
