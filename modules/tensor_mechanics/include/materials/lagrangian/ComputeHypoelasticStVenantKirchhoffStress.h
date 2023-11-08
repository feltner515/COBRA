//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "ComputeLagrangianObjectiveStress.h"

// Calculate a small strain elastic stress that is equivalent to the hyperelastic St.
// Venant-Kirchhoff model if integrated using the Truesdell rate.
//
// S_{n+1} = S_n + C : dD
// with C the current elasticity tensor, and dD the strain increment
class ComputeHypoelasticStVenantKirchhoffStress : public ComputeLagrangianObjectiveStress
{
public:
  static InputParameters validParams();
  ComputeHypoelasticStVenantKirchhoffStress(const InputParameters & parameters);

protected:
  /// Implement the elastic small stress update
  virtual void computeQpSmallStress();

protected:
  /// The elasticity tensor
  const MaterialProperty<RankFourTensor> & _elasticity_tensor;

  /// The deformation gradient
  const MaterialProperty<RankTwoTensor> & _def_grad;
};
