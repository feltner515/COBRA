//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "ADScalarKernel.h"

class ADShaftConnectableUserObjectInterface;

/**
 * Torque contributed by a component connected to a shaft
 */
class ADShaftComponentTorqueScalarKernel : public ADScalarKernel
{
public:
  ADShaftComponentTorqueScalarKernel(const InputParameters & parameters);

protected:
  virtual ADReal computeQpResidual() override;

  /// Shaft connected component user object
  const ADShaftConnectableUserObjectInterface & _shaft_connected_component_uo;

public:
  static InputParameters validParams();
};
