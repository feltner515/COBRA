//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "HeatConductionTimeDerivative.h"
#include "Material.h"

class TrussHeatConductionTimeDerivative : public HeatConductionTimeDerivative
{
public:
  static InputParameters validParams();

  TrussHeatConductionTimeDerivative(const InputParameters & parameters);

protected:
  virtual Real computeQpResidual() override;

  virtual Real computeQpJacobian() override;

  /// Coupled variable for the cross-sectional area of truss element
  const VariableValue & _area;
};
