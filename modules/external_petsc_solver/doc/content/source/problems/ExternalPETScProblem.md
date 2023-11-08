# ExternalPETScProblem

This is an interface to call a pure PETSc solver. We also sync the PETSc solution to moose variables, and then these variables can be coupled to other moose applications

!syntax parameters /Problem/ExternalPETScProblem

!syntax inputs /Problem/ExternalPETScProblem

!syntax children /Problem/ExternalPETScProblem
