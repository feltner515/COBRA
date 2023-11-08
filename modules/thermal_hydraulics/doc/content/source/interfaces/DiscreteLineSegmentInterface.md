# DiscreteLineSegmentInterface

This interface is used to define mesh along a line segment in 3D space.
The line segment is defined with a "start" point $\mathbf{x}_\text{start}$,
corresponding to either end, the direction $\mathbf{d}$ to the other end, and
the distance in that direction, $L$. Thus the other end of the line segment is

!equation
\mathbf{x}_\text{end} = \mathbf{x}_\text{start} + L \mathbf{d} \eqp

These quantities are defined using the following parameters:

- `position`: the "start" point $\mathbf{x}_\text{start}$,
- `orientation`: the direction $\mathbf{d}$ (which gets automatically normalized), and
- `length`: the length(s) that sum to $L$.

The most basic mesh specification is given by a single value for the parameters
`length` and `n_elems`, which correspond to the length of the
component and number of uniformly-sized elements to use. For example, the
following parameters would specify a total length $L = 50$ m, divided
into 100 elements (each with width 0.5 m):

```
length = 50
n_elems = 100
```

The `length` and `n_elems` parameters can also be supplied with
multiple values. Multiple values correspond to splitting the length into
segments that can have different element sizes. However, within each segment,
the discretization is assumed uniform. The numbers of elements in each segment
are specified with the parameter `n_elems`, with
entries corresponding to the entries in `length`.
For example, the following would also specify a total length $L = 50$
m with 100 total elements, but in this case the first 10 m have 40 elements of
size 0.25 m, whereas the last 40 m have 60 elements of size $0.\bar{6}$ m.

```
length = '10 40'
n_elems = '40 60'
```
