# PatternedCartesianPeripheralModifier

!syntax description /Mesh/PatternedCartesianPeripheralModifier

## Overview

This `PatternedCartesianPeripheralModifier` class utilizes [`FillBetweenPointVectorsTools`](/FillBetweenPointVectorsTools.md) to replace the outmost layer of the quad elements of the 2D cartesian assembly mesh generated by [`PatternedCartesianMeshGenerator`](PatternedCartesianMeshGenerator.md) with a transition layer consisting of triangular elements so that the assembly mesh can have nodes on designated positions on the external boundary. This boundary modification facilitates the stitching of cartesian assemblies which have different node numbers on their outer periphery due to differing numbers of interior pins and/or different azimuthal discretization.

##  Motivation

The motivation of developing this mesh generator is similar to that of the [`PatternedHexPeripheralModifier`](/PatternedHexPeripheralModifier.md). Please refer to that documentation for details.

## Modification of Peripheral Boundary to Allow Stitching

The `PatternedCartesianPeripheralModifier` class modifies assembly meshes so that assemblies with different number of pins can be stitched together without increasing the mesh fidelity to an impractically fine fidelity. This mesh generator only works with the [!param](/Mesh/PatternedCartesianPeripheralModifier/input) mesh created by [`PatternedCartesianMeshGenerator`](PatternedCartesianMeshGenerator.md). Users must specify the external boundary of the input assembly mesh through [!param](/Mesh/PatternedCartesianPeripheralModifier/input_mesh_external_boundary). Given this input, the mesh generator identifies and deletes the outmost layer of elements and uses the newly formed external boundary as one of the two vectors of boundary nodes needed by [`FillBetweenPointVectorsTools`](/FillBetweenPointVectorsTools.md) after symmetry reduction. In addition, uniformly distributed nodes are placed along the original external boundary of the mesh and defined as the second vector of boundary nodes needed by [`FillBetweenPointVectorsTools`](/FillBetweenPointVectorsTools.md). The number of new boundary nodes is specified using [!param](/Mesh/PatternedHexPeripheralModifier/new_num_sector). Thus, the outmost layer of the assembly mesh can be replaced with a triangular element transition layer mesh that can be easily stitched with another transition layer mesh. An example of the assembly mesh modified by this mesh generator is shown in [assembly_example]

!media reactor/meshgenerators/cart_mod.png
      style=display: block;margin-left:auto;margin-right:auto;width:60%;
      id=assembly_example
      caption=A schematic drawing of an example cartesian assembly mesh with transition layer as its outmost mesh layer.

!alert note
The extra element IDs from the original peripheral region are conserved. They may be modified using the [!param](/Mesh/PatternedCartesianPeripheralModifier/extra_id_names_to_modify) and [!param](/Mesh/PatternedCartesianPeripheralModifier/new_extra_id_values_to_assign) parameters.

## Advantages

This mesh generator forces the number of nodes on a cartesian mesh to match a user-specified input. This allows assemblies with different number of pins or azimuthal discretizations (and consequently different numbers of boundary nodes) to be stitched together without increasing the mesh density to an unreasonable level.

!media reactor/meshgenerators/cart_mod_pattern.png
      style=display: block;margin-left:auto;margin-right:auto;width:60%;
      id=pattern_adv
      caption=A schematic drawing showing a virtual core design with assemblies including 9, 16, 25 and 36 pins.

[pattern_adv] illustrates a core comprising four types of assemblies. This mesh generator's functionality was leveraged to force a common mesh density on each cartesian assembly side (10 nodes on each assembly side) so that the assemblies can be easily stitched.

## Example Syntax

!listing modules/reactor/test/tests/meshgenerators/patterned_cartesian_peripheral_modifier/patterned.i block=Mesh/pmg_1

!syntax parameters /Mesh/PatternedCartesianPeripheralModifier

!syntax inputs /Mesh/PatternedCartesianPeripheralModifier

!syntax children /Mesh/PatternedCartesianPeripheralModifier