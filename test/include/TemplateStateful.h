//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "Material.h"

template <bool is_ad>
class TemplateStatefulTempl : public Material
{
public:
  static InputParameters validParams();

  TemplateStatefulTempl(const InputParameters & parameters);

protected:
  virtual void computeQpProperties() override;
  virtual void initQpStatefulProperties() override;

  /// Property value
  GenericMaterialProperty<Real, is_ad> & _property;

  /// Old property value
  const MaterialProperty<Real> & _property_old;
};

typedef TemplateStatefulTempl<false> TemplateStateful;
typedef TemplateStatefulTempl<true> ADTemplateStateful;
