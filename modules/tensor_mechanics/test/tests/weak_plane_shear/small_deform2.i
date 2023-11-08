# apply a pure tension, then some shear with compression
# the BCs are designed to map out the yield function, showing
# the affect of the small_smoother parameter
[GlobalParams]
  displacements = 'x_disp y_disp z_disp'
[]

[Mesh]
  type = GeneratedMesh
  dim = 3
  xmin = -0.5
  xmax = 0.5
  ymin = -0.5
  ymax = 0.5
  zmin = -0.5
  zmax = 0.5
[]

[Modules/TensorMechanics/Master/all]
  strain = FINITE
  add_variables = true
  generate_output = 'stress_xz stress_zx stress_yz stress_zz'
[]

[BCs]
  [bottomx]
    type = DirichletBC
    variable = x_disp
    boundary = back
    value = 0.0
  []
  [bottomy]
    type = DirichletBC
    variable = y_disp
    boundary = back
    value = 0.0
  []
  [bottomz]
    type = DirichletBC
    variable = z_disp
    boundary = back
    value = 0.0
  []

  [topx]
    type = FunctionDirichletBC
    variable = x_disp
    boundary = front
    function = 'if(t<1E-6,0,3*t)'
  []
  [topy]
    type = FunctionDirichletBC
    variable = y_disp
    boundary = front
    function = 'if(t<1E-6,0,5*(t-0.01E-6))'
  []
  [topz]
    type = FunctionDirichletBC
    variable = z_disp
    boundary = front
    function = 'if(t<1E-6,t,2E-6-t)'
  []
[]

[AuxVariables]
  [yield_fcn]
    order = CONSTANT
    family = MONOMIAL
  []
[]

[AuxKernels]
  [yield_fcn_auxk]
    type = MaterialStdVectorAux
    property = plastic_yield_function
    index = 0
    variable = yield_fcn
  []
[]

[Postprocessors]
  [s_xz]
    type = PointValue
    point = '0 0 0'
    variable = stress_xz
  []
  [s_yz]
    type = PointValue
    point = '0 0 0'
    variable = stress_yz
  []
  [s_zz]
    type = PointValue
    point = '0 0 0'
    variable = stress_zz
  []
  [f]
    type = PointValue
    point = '0 0 0'
    variable = yield_fcn
  []
[]

[UserObjects]
  [coh]
    type = TensorMechanicsHardeningConstant
    value = 1E3
  []
  [tanphi]
    type = TensorMechanicsHardeningConstant
    value = 1
  []
  [tanpsi]
    type = TensorMechanicsHardeningConstant
    value = 0.01745506
  []
  [wps]
    type = TensorMechanicsPlasticWeakPlaneShear
    cohesion = coh
    tan_friction_angle = tanphi
    tan_dilation_angle = tanpsi
    smoother = 500
    yield_function_tolerance = 1E-3
    internal_constraint_tolerance = 1E-6
  []
[]

[Materials]
  [elasticity_tensor]
    type = ComputeElasticityTensor
    fill_method = symmetric_isotropic
    C_ijkl = '1E9 0.5E9'
  []
  [mc]
    type = ComputeMultiPlasticityStress
    ep_plastic_tolerance = 1E-4
    plastic_models = wps
    transverse_direction = '0 0 1'
    debug_fspb = crash
  []
[]

[Executioner]
  end_time = 2E-6
  dt = 1E-7
  type = Transient
[]

[Outputs]
  csv = true
[]
