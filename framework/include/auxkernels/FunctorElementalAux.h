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

/**
 * Evaluate a functor (functor material property, function or variable) with either the cell-center
 * or quadrature point as the functor argument
 */
template <bool is_ad>
class FunctorElementalAuxTempl : public AuxKernel
{
public:
  static InputParameters validParams();

  FunctorElementalAuxTempl(const InputParameters & parameters);

protected:
  virtual Real computeValue() override;

  /// Functor to evaluate with the element argument
  const Moose::Functor<GenericReal<is_ad>> & _functor;

  /// Factor to multiply the functor with
  const Moose::Functor<GenericReal<is_ad>> & _factor;

  /// Whether to use a quadrature-based functor argument, appropriate for finite element
  /// evaluations. If false, use a cell-center functor argument appropriate for finite volume
  /// calculations
  const bool _use_qp_arg;
};

typedef FunctorElementalAuxTempl<false> FunctorElementalAux;
typedef FunctorElementalAuxTempl<true> ADFunctorElementalAux;
