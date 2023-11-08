[GlobalParams]
  displacements = 'disp_x disp_y disp_z'
[]

[Mesh]
  type = GeneratedMesh
  dim = 3
  xmin = -0.5
  xmax = 0.5
  nx = 5
  ny = 5
  nz = 5
  elem_type = HEX8
[]

[Modules/TensorMechanics/Master]
  [all]
    strain = SMALL
    incremental = true
    add_variables = true
    generate_output = 'stress_xx strain_xx'
    use_automatic_differentiation = true
  []
[]

[BCs]
  [symmy]
    type = ADDirichletBC
    variable = disp_y
    boundary = bottom
    value = 0
  []
  [symmx]
    type = ADDirichletBC
    variable = disp_x
    boundary = left
    value = 0
  []
  [symmz]
    type = ADDirichletBC
    variable = disp_z
    boundary = back
    value = 0
  []
  [axial_load]
    type = ADDirichletBC
    variable = disp_x
    boundary = right
    value = 0.01
  []
[]

[Functions]
  [func]
    type = ParsedFunction
    expression = 'if(x>=0,0.5*t, t)'
  []
[]

[UserObjects]
  [ele_avg]
    type = RadialAverage
    prop_name = local_damage_reg
    weights = constant
    execute_on = "INITIAL timestep_end"
    radius = 0.55
  []
[]

[Materials]
  [non_ad_local_damage]
    type = MaterialADConverter
    ad_props_in = local_damage
    reg_props_out = local_damage_reg
  []
  [local_damage_index]
    type = ADGenericFunctionMaterial
    prop_names = local_damage_index
    prop_values = func
  []
  [local_damage]
    type = ADScalarMaterialDamage
    damage_index = local_damage_index
    damage_index_name = local_damage
  []
  [damage]
    type = ADNonlocalDamage
    average_UO = ele_avg
    local_damage_model = local_damage
    damage_index_name = nonlocal_damage
  []
  [elasticity]
    type = ADComputeIsotropicElasticityTensor
    poissons_ratio = 0.2
    youngs_modulus = 10e9
  []
  [stress]
    type = ADComputeDamageStress
    damage_model = damage
  []
[]

[Postprocessors]
  [stress_xx]
    type = ElementAverageValue
    variable = stress_xx
  []
  [strain_xx]
    type = ElementAverageValue
    variable = strain_xx
  []
  [nonlocal_damage]
    type = ADElementAverageMaterialProperty
    mat_prop = nonlocal_damage
  []
  [local_damage]
    type = ADElementAverageMaterialProperty
    mat_prop = local_damage
  []
[]

[Executioner]
  type = Transient

  l_max_its = 50
  l_tol = 1e-8
  nl_max_its = 20
  nl_rel_tol = 1e-12
  nl_abs_tol = 1e-8

  dt = 0.2
  dtmin = 0.1
  end_time = 1
[]

[Outputs]
  csv = true
[]
