[Mesh]
  type = GeneratedMesh
  dim = 2
  nx = 1
  ny = 1
[]

[AuxVariables]
  [./dummy]
  [../]
[]

[Materials]
  [./provider]
    type = ADDerivativeMaterialInterfaceTestProvider
    block = 0
  [../]
  [./client]
    type = ADDerivativeMaterialInterfaceTestClient
    prop_name = prop
    block = 0
    outputs = exodus
  [../]
  [./client2]
    type = ADDerivativeMaterialInterfaceTestClient
    prop_name = 1.0
    block = 0
    outputs = exodus
  [../]

  [./dummy]
    type = ADGenericConstantMaterial
    prop_names = prop
    block = 0
    prop_values = 0
  [../]
[]

[Executioner]
  type = Steady
[]

[Problem]
  solve = false
[]

[Outputs]
  exodus = true
[]
