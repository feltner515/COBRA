[Mesh]
  [gmg]
    type = GeneratedMeshGenerator
    dim = 3
    nx = 3
    ny = 3
    nz = 4
    bias_x = 2
    bias_z = 0.5
  []
[]

[Outputs]
  exodus = true
[]
