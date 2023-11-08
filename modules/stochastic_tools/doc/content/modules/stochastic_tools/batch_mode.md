# Stochastic Tools Batch Mode

The [SamplerFullSolveMultiApp.md] and [SamplerTransientMultiApp.md] are capable of running sub-applications
in one of three different modes:

1. +normal+: One sub-application is created for each row of data (`num_rows`) supplied by the Sampler object.
1. +batch-reset+: $N$ sub-applications are created, where the sub-applications are
                  destroyed and re-created (on the same existing MPI communicator)
                  for each row of data supplied by the Sampler object.
1. +batch-restore+: $N$ sub-applications are created, where the sub-application is
                    backed up after initialization. Then for each row of data supplied by the
                    Sampler object the sub-application is restored to the initial state prior to
                    execution.

For the two "batch" options, $N$ indicates the number of applications created, which in the
most general expression is given by

\begin{equation}
N=\text{min}(n_{rows}, \text{floor}\left(\frac{n_{proc}}{n_{min}}\right))
\end{equation}

where $n_{rows}$ is the `num_rows` parameter on the Sampler object, $n_{proc}$ is the number
of processors (launched to the `mpiexec` command), and $n_{min}$ is the `min_procs_per_app`,
or the minimum number of processors which should be used to run each sub-application. If you
launch your Stochastic Tools main application with fewer than `min_procs_per_app`, the simulation
will still proceed, but will just use the maximum number of ranks you have provided as the
"effective" `min_procs_per_app`. We can
also illustrate this with an example.
[batch_n_table] shows the number of applications created ($N$)
for two different choices of `num_rows`, `min_procs_per_app`, and number of processors.

!table id=batch_n_table caption=Number of sub-apps launched for the "batch" modes for different example choices of `num_rows` and `min_procs_per_app`
| `num_rows` | `min_procs_per_app` | Processors | Number of Sub-Apps |
| :- | :- | :- | :- |
| 4 | 2 | 9 | 4 |
| 20 | 2 | 9 | 4 |
| 4 | 2 | 3 | 1 |
| 4 | 2 | 1 | 1 |

All three modes are available when using SamplerFullSolveMultiApp, the "batch-reset" mode is not
available for SamplerTransientMultiApp because the sub-application has state that must be
maintained as simulation time progresses.

The primary benefit to using a batch mode is to improve performance of a simulation by reducing the
memory of the running application. The performance gains depend on the type of sub-application being
executed as well as the number of samples being evaluated. The following sections highlight the
the performance improvements that may be expected for full solve and transient sub-applications.

## Example 1: Full Solve Sub-Application

The first example demonstrates the performance improvements to expect when using
SamplerFullSolveMultiApp with sub-applications. In this case, the sub-application
solves steady-state diffusion on a unit cube domain with Dirichlet boundary conditions on the
left, $x=0$, and right, $y=0$, sides of the domain, the complete input file for this problem
is given in [steady-sub].

!listing stochastic_tools/examples/batch/sub.i id=steady-sub caption=Complete input file
         for steady-state diffusion problem.

The master application does not perform a solve, it performs a stochastic analysis using the
[MonteCarlo](source/samplers/MonteCarloSampler.md) object to perturb the values of the two Dirichlet conditions on the sub-applications
to vary with a uniform distribution. The complete input file for the master application is given
in [steady-sub].

!listing stochastic_tools/examples/batch/full_solve.i id=steady-master caption=Complete input file
         for master application that performs stochastic simulations of the
         steady-state diffusion problem in [steady-sub] using Monte Carlo sampling.

The example is executed to demonstrate memory performance of the various modes of operation:
"normal", "batch-reset", and "batch-restore". Each mode is executed with increasing
number of Monte Carlo samples by setting the "n_samples" parameter of the MonteCarloSampler object.
[full-serial-memory] and [full-mpi-memory] show the resulting memory use at the end of the
simulation for each mode of operation with increasing sample numbers in serial and in parallel,
respectively.

!media full_solve_memory_serial.svg id=full-serial-memory
       caption=Total memory at the end of the simulation using a SamplerFullSolveMultiApp with
               increasing number of Monte Carlo samples for the three available modes of operation
               running on a single processor.

!media full_solve_memory_mpi.svg id=full-mpi-memory
       caption=Total memory and maximum memory per processor at the end of the simulation using a
               SamplerFullSolveMultiApp with increasing number of Monte Carlo samples for the three
               available modes of operation running on 56 processors.

An important feature of the various modes of operation is that run-time is not negatively
impacted by changing the mode, in some cases using a batch mode can actually decrease total
simulation run time.

The total run time results for the full solve problem in serial and parallel
are shown in [full-serial-time] and [full-mpi-time], respectively. The time shown in these plots
is the total simulation time, which encompasses both the simulation initialization and solve. The
differences in speed are mainly due to the installation and destruction of the sub-application.
When running in 'batch-reset' mode, each data sample causes the sub-application to be created and
destroyed during the solve, causing the slowest performance. The 'normal' mode creates all
sub-applications up front, and the 'batch-restore' method uses the backup-restore capability to
save the state of the sub-applications, thus does not require as many instantiations and has the
lowest run-time. For this example, the solve portion is minimal as such the sub-application
creation time plays a large role. As the solve time increases time gains can be expected to be
minimal.

!media full_solve_memory_serial_time.svg id=full-serial-time
       caption=Total execution time of a simulation using SamplerFullSolveMultiApp with increasing
               number of Monte Carlo samples for the available modes of operation on a single
               processor.

!media full_solve_memory_mpi_time.svg id=full-mpi-time
       caption=Total execution time of a simulation using SamplerFullSolveMultiApp with increasing
               number of Monte Carlo samples for the available modes of operation on 56
               processors.

## Example 2: Transient Sub-Application

The second example is nearly identical to the first, except the master application is a transient
solve that sets the boundary conditions at the end of each time step. The only difference occurs
in the master input file, in the Executioner and MultiApps block, as shown in [transient-master].

!listing stochastic_tools/examples/batch/transient.i id=transient-master block=Executioner MultiApps
         caption=Complete input file for a transient master application that performs stochastic
         simulations of a diffusion problem with time varying boundary conditions using using Monte
         Carol sampling.

The results shown in [transient-serial-memory] and [transient-mpi-memory] include the memory use at
the end of the simulation (10 time steps) for each mode of operation within increasing number of
samples in serial and parallel. Recall, as mentioned above, that the "batch-reset" mode is not
available in the SamplerTransientMultiApp.

!media transient_memory_serial.svg id=transient-serial-memory
       caption=Total memory at the end of the simulation using a SamplerTransientMultiApp with
               increasing number of Monte Carlo samples for the two available modes of operation
               running on a single processor.

!media transient_memory_mpi.svg id=transient-mpi-memory
       caption=Total memory and maximum memory per processor at the end of the simulation using a
               SamplerTransientMultiApp with increasing number of Monte Carlo samples for the two
               available modes of operation running on 56 processors.

Again, an important feature of the various modes of operation is that run-time is not negatively
impacted by changing the mode as seen in [transient-serial-time] and [transient-mpi-time]. The
solve portion of this example is significantly longer than the steady-state example. As such the
differences in execution time due to the instantiating of objects is diminished and both modes behave
similarly.

!media transient_memory_serial_time.svg id=transient-serial-time
       caption=Total execution time of a simulation using SamplerTransientMultiApp with increasing
               number of Monte Carlo samples for the available modes of operation on a single
               processor.

!media transient_memory_mpi_time.svg id=transient-mpi-time
       caption=Total execution time of a simulation using SamplerTransientMultiApp with increasing
               number of Monte Carlo samples for the available modes of operation on 56
               processors.
