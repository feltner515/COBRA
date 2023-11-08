//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "FunctorMaterial.h"

/**
 * This is the material class used to compute phase averaged properties of mixtures
 */
class NSFVMixtureMaterial : public FunctorMaterial
{
public:
  static InputParameters validParams();

  NSFVMixtureMaterial(const InputParameters & parameters);

protected:
  /// Vector of phase 1 properties
  std::vector<const Moose::Functor<ADReal> *> _phase_1_properties;

  /// Vector of phase 2 properties
  std::vector<const Moose::Functor<ADReal> *> _phase_2_properties;

  /// Vector of phase 1 properties names
  std::vector<MooseFunctorName> _phase_1_names;

  /// Vector of phase 2 properties names
  std::vector<MooseFunctorName> _phase_2_names;

  /// Vector of mixture properties names
  std::vector<MooseFunctorName> _mixture_names;

  /// Phase 1 fraction
  const Moose::Functor<ADReal> & _phase_1_fraction;
};
