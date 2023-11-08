//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "Closures1PhaseBase.h"

/**
 * Sets up no 1-phase closures
 *
 * Users have to implement the closure materials and supply them in the input file
 */
class Closures1PhaseNone : public Closures1PhaseBase
{
public:
  Closures1PhaseNone(const InputParameters & params);

  virtual void checkFlowChannel(const FlowChannelBase & flow_channel) const override;
  virtual void checkHeatTransfer(const HeatTransferBase & heat_transfer,
                                 const FlowChannelBase & flow_channel) const override;
  virtual void addMooseObjectsFlowChannel(const FlowChannelBase & flow_channel) override;
  virtual void addMooseObjectsHeatTransfer(const HeatTransferBase & heat_transfer,
                                           const FlowChannelBase & flow_channel) override;

public:
  static InputParameters validParams();
};
