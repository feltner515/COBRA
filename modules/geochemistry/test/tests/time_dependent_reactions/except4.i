#Exception: cannot remove fixed activity of a species that does not have an activity or fugacity constraint
[TimeDependentReactionSolver]
  model_definition = definition
  geochemistry_reactor_name = reactor
  charge_balance_species = "Cl-"
  constraint_species = "H2O H+ Na+ K+ Ca++ Mg++ Al+++ SiO2(aq) Cl- SO4-- HCO3-"
  constraint_value = "  1.0 1E-5 1E-5 1E-5 1E-5 1E-5 1E-5 1E-5 1E-5 1E-5 1E-5"
  constraint_meaning = "kg_solvent_water activity bulk_composition bulk_composition bulk_composition bulk_composition bulk_composition bulk_composition bulk_composition bulk_composition bulk_composition"
  constraint_unit = "kg dimensionless moles moles moles moles moles moles moles moles moles"
  remove_fixed_activity_name = "Na+"
  remove_fixed_activity_time = "0"
[]

[Executioner]
  type = Steady
[]

[UserObjects]
  [definition]
    type = GeochemicalModelDefinition
    database_file = "../../../database/moose_geochemdb.json"
    basis_species = "H2O H+ Na+ K+ Ca++ Mg++ Al+++ SiO2(aq) Cl- SO4-- HCO3-"
  []
[]

