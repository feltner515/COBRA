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
#include "MooseEnum.h"
#include "MoosePartitioner.h"

class MooseMesh;

/**
 * Partitions a mesh into sub-partitions for each computational node
 * then into partitions within that node.  All partitions are made
 * using a regular grid.
 */
class HierarchicalGridPartitioner : public MoosePartitioner
{
public:
  HierarchicalGridPartitioner(const InputParameters & params);
  virtual ~HierarchicalGridPartitioner();

  static InputParameters validParams();

  virtual std::unique_ptr<Partitioner> clone() const override;

protected:
  virtual void _do_partition(MeshBase & mesh, const unsigned int n) override;

  MooseMesh & _mesh;

  const unsigned int _nx_nodes;
  const unsigned int _ny_nodes;
  const unsigned int _nz_nodes;
  const unsigned int _nx_procs;
  const unsigned int _ny_procs;
  const unsigned int _nz_procs;
};
