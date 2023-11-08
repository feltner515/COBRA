//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

// MOOSE includes
#include "ElementIntegralPostprocessor.h"

/**
 * An object for testing that the specified quadrature order is used.  It
 * counts the number of quadrature points.
 */
class NumElemQPs : public ElementIntegralPostprocessor
{
public:
  static InputParameters validParams();

  NumElemQPs(const InputParameters & parameters);
  virtual ~NumElemQPs();
  virtual Real computeIntegral() override;
  virtual Real computeQpIntegral() override;
};
