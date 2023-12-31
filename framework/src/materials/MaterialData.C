//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "MaterialData.h"
#include "Material.h"

MaterialData::MaterialData(MaterialPropertyStorage & storage)
  : _storage(storage), _n_qpoints(0), _swapped(false), _resize_only_if_smaller(false)
{
}

void
MaterialData::resize(unsigned int n_qpoints)
{
  if (n_qpoints == nQPoints())
    return;

  if (_resize_only_if_smaller && n_qpoints < nQPoints())
    return;

  for (const auto state : _storage.stateIndexRange())
    props(state).resizeItems(n_qpoints, {});
  _n_qpoints = n_qpoints;
}

void
MaterialData::copy(const Elem & elem_to, const Elem & elem_from, unsigned int side)
{
  _storage.copy(*this, &elem_to, &elem_from, side, nQPoints());
}

void
MaterialData::swap(const Elem & elem, unsigned int side /* = 0*/)
{
  if (!_storage.hasStatefulProperties() || isSwapped())
    return;

  _storage.swap(*this, elem, side);
  _swapped = true;
}

void
MaterialData::reset(const std::vector<std::shared_ptr<MaterialBase>> & mats)
{
  for (const auto & mat : mats)
    mat->resetProperties();
}

void
MaterialData::swapBack(const Elem & elem, unsigned int side /* = 0*/)
{
  if (isSwapped() && _storage.hasStatefulProperties())
  {
    _storage.swapBack(*this, elem, side);
    _swapped = false;
  }
}

void
MaterialData::mooseErrorHelper(const MooseObject & object, const std::string_view & error)
{
  object.mooseError(error);
}
