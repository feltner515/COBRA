[Mesh]
  type = PeridynamicsMesh
  horizon_number = 3

  [./fmg]
    type = FileMeshGenerator
    file = 2D_2blocks.e
  [../]
  [./gpd]
    type = MeshGeneratorPD
    input = fmg
    retain_fe_mesh = true
    blocks_to_pd = 2
    #blocks_as_fe = 1
  [../]
[]

# This input file is intended to be run with the "--mesh-only" option so
# no other sections are required
