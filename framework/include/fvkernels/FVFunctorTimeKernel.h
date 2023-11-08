//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "FVElementalKernel.h"

class FVFunctorTimeKernel : public FVElementalKernel
{
public:
  static InputParameters validParams();
  FVFunctorTimeKernel(const InputParameters & parameters);

protected:
  ADReal computeQpResidual() override;

  /// The functor to query for the time derivative
  const Moose::FunctorBase<ADReal> & _functor;
};
