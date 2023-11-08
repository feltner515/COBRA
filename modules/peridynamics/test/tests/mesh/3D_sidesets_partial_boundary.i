[Mesh]
  type = PeridynamicsMesh
  horizon_number = 3

  [./fmg]
    type = FileMeshGenerator
    file = cylinder.e
  [../]
  [./mgpd]
    type = MeshGeneratorPD
    input = fmg
    retain_fe_mesh = false
    construct_pd_sidesets = true
  [../]
[]
