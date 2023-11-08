[GlobalParams]
  order = FIRST
  family = LAGRANGE
  displacements = 'disp_x disp_y disp_z'
  volumetric_locking_correction = true
[]

[Mesh]
  file = ellip_crack_4sym_norad_mm.e
  partitioner = centroid
  centroid_partitioner_direction = z
[]

[AuxVariables]
  [SED]
    order = CONSTANT
    family = MONOMIAL
  []
  [resid_z]
  []
[]

[Functions]
  [rampConstantUp]
    type = PiecewiseLinear
    x = '0. 1.'
    y = '0. 0.1'
    scale_factor = -689.5 #MPa
  []
[]

[DomainIntegral]
  integrals = JIntegral
  boundary = 1001
  crack_direction_method = CrackMouth
  crack_mouth_boundary = 11
  crack_end_direction_method = CrackDirectionVector
  crack_direction_vector_end_1 = '0.0 1.0 0.0'
  crack_direction_vector_end_2 = '1.0 0.0 0.0'
  radius_inner = '12.5 25.0 37.5'
  radius_outer = '25.0 37.5 50.0'
  intersecting_boundary = '1 2'
  symmetry_plane = 2
  position_type = angle
  incremental = true
  use_automatic_differentiation = true
[]

[Modules/TensorMechanics/Master]
  [master]
    strain = FINITE
    add_variables = true
    incremental = true
    generate_output = 'stress_xx stress_yy stress_zz vonmises_stress'
    use_automatic_differentiation = true
  []
[]

[AuxKernels]
  [SED]
    type = MaterialRealAux
    variable = SED
    property = strain_energy_density
    execute_on = timestep_end
  []
[]

[BCs]
  [crack_y]
    type = ADDirichletBC
    variable = disp_z
    boundary = 6
    value = 0.0
  []
  [no_y]
    type = ADDirichletBC
    variable = disp_y
    boundary = 12
    value = 0.0
  []
  [no_x]
    type = ADDirichletBC
    variable = disp_x
    boundary = 1
    value = 0.0
  []
  [Pressure]
    [Side1]
      boundary = 5
      function = rampConstantUp
    []
  [] # BCs
[]

[Materials]
  [elasticity_tensor]
    type = ADComputeIsotropicElasticityTensor
    youngs_modulus = 206800
    poissons_ratio = 0.3
  []
  [elastic_stress]
    type = ADComputeFiniteStrainElasticStress
  []
[]

[Executioner]

  type = Transient
  petsc_options = '-snes_converged_reason -ksp_converged_reason -pc_svd_monitor '
                  '-snes_linesearch_monitor -snes_ksp_ew'
  petsc_options_iname = '-pc_type -pc_factor_shift_type'
  petsc_options_value = 'lu       NONZERO'

  nl_max_its = 20
  nl_abs_tol = 1e-5
  nl_rel_tol = 1e-11

  start_time = 0.0
  dt = 1

  end_time = 1
  num_steps = 1
[]

[Postprocessors]
  [_dt]
    type = TimestepSize
  []
  [nl_its]
    type = NumNonlinearIterations
  []
  [lin_its]
    type = NumLinearIterations
  []
  [react_z]
    type = NodalSum
    variable = resid_z
    boundary = 5
  []
[]

[Outputs]
  execute_on = 'timestep_end'
  file_base = j_int_surfbreak_ellip_crack_sym_mm_cm_ad_out
  csv = true
[]
