//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "AuxKernel.h"
#include "TagAuxBase.h"

/**
 * For visualization or other purposes, the diagnal of the matrix of a tag
 * is extracted, and nodal values are assigned by using the matrix diagnal values.
 */
class TagMatrixAux : public TagAuxBase<AuxKernel>
{
public:
  static InputParameters validParams();

  TagMatrixAux(const InputParameters & parameters);

protected:
  virtual Real computeValue() override;

  const VariableValue & _v;
  const MooseVariableBase & _v_var;
};
