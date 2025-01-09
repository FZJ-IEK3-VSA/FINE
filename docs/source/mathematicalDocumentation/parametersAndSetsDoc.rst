General Parameters and Sets
###########################

The energy system’s basic framework is constituted by a number of
parameters and sets that hold information about the components,
commodities and the spatial and temporal resolution with which the
energy system is modeled. The description of the parameters and sets is based on the description in `Welder (2022) <https://publications.rwth-aachen.de/record/861215/files/861215.pdf>`_ 
and has been updated with the changes of the latest ETHOS.FINE version.

Component Sets
**************

The model of the energy system is based on several components with different attributes. 
A component can be assigned to either the nodes or the edges of the model. 
The components assigned to nodes are described by the set :math:`\mathcal{C}^\text{node}`.
The components assigned to edges are described by the set :math:`\mathcal{C}^\text{edge}`. 
The set that contains all components :math:`\mathcal{C}` with which the
energy system is modeled is given by

.. math::

   \begin{aligned}
       \mathcal{C} &~=~& \mathcal{C}^\text{node} \cup \mathcal{C}^\text{edge}.
   \end{aligned}

Commodity Sets
**************

The set of all commodities (goods) that
are considered in the energy system is given by the set :math:`\mathcal{G}`.
The set :math:`\mathcal{G}_\text{c} \subseteq \mathcal{G}` contains all commodities associated to component :math:`c \in \mathcal{C}`.

Location Sets
*************

The set of all locations that
are considered in the energy system is given by :math:`\mathcal{L}`.
A location is modeled as a *node* in the framework. 

If component :math:`c \in\mathcal{C}^\text{node}`, the set of
locations at which the component is modeled is defined as

.. math::

   \begin{aligned}
       \mathcal{L}_\text{c} = \left\{ \text{\small l} ~\vert~ \forall~\text{\small l}\in\mathcal{L}: \text{\small E}_\text{c,l}=1 \right\}.
   \end{aligned}

The parameter :math:`\text{E}_\text{c,l}` with :math:`c \in\mathcal{C}^\text{node}`
equals 1 if a component is eligible at that location.

Arc Sets
********

A connection between two locations is modeled as an *edge* called arc. 
The set of all arcs :math:`\mathcal{A}` that
are considered in the energy system is given by

.. math::

   \begin{aligned}
       \mathcal{A} = \{ a = (l_\text{1}, l_\text{2} ) \in \mathcal{L} \times \mathcal{L} \}
   \end{aligned}

If component :math:`c \in\mathcal{C}^\text{edge}`, the set
of arcs at which the component is modeled is defined as

.. math::

   \begin{aligned}
       \mathcal{A}_\text{c} = \big\{ \text{a} ~\vert~ \forall~\text{a}\in\mathcal{A}: \text{\small E}_\text{c,a}=1 \big\}.\ 
   \end{aligned}

The parameter :math:`\text{E}_\text{c,a}` with :math:`c \in\mathcal{C}^\text{edge}`
equals 1 if a component is eligible at that arc.

*Edge*-based components are modeled as being
bidirectional by default. This implies that if a connection between
:math:`\text{l}_\text{1}` and :math:`\text{l}_\text{2}` is
eligible, also the connection between :math:`\text{l}_\text{2}` and
:math:`\text{l}_\text{1}` is eligible.

Time Sets
*********

The parameter :math:`\text{T}^\text{total}\in\mathbb{N}`, by
default 8760 (i.e. 24 h/day :math:`\cdot` 365 day :math:`=` 8760 h),
specifies the total number of time steps with which the energy system is
modeled. The corresponding index set that encompasses all of these time
steps is

.. math::

   \begin{aligned}
       \mathcal{T}&~=~&\left\{0,\dots,\text{\small T}^\text{total}-1\right\}.
   \end{aligned}

The parameter :math:`\text{T}^\text{hours}\in\mathbb{R}^{+}` defines the
number of hours per time step, by default 1 h. The number of years
:math:`\text{T}^\text{years}` which the energy system covers is determined
by

.. math::

   \begin{aligned}
       \text{T}^\text{years}&~=~&\frac{\text{\small T}^\text{total} \cdot \text{\small T}^\text{hours}}{8760~\text{\small h}}~\text{\small a}.
   \end{aligned}

Thus, the default value represents one year (1 a). 


**Additional Time Sets for time series aggregation**

ETHOS.FINE provides support to use time series aggregation by using built-in methods integrating 
the python package `tsam <https://github.com/FZJ-IEK3-VSA/tsam>`_ into the code. Therefor, additional sets and parameters are introduced:
The parameter
:math:`\text{T}^\text{per period}{}\in\mathbb{Z}^{+}`
specifies the number of time steps per period. Thereby,
:math:`\text{T}^\text{total}` must be a multiple of
:math:`\text{T}^\text{per period}`,
i.e. :math:`\text{T}^\text{total} \pmod{\text{T}^\text{per\, period}} = 0`. If
the energy system is investigated with its full temporal resolution,
:math:`\text{T}^\text{per period}` is set equal to
:math:`\text{T}^\text{total}`, i.e. the energy system is investigated with only one period. If the energy system
is modeled with typical periods,
:math:`\text{T}^\text{per period}` is set smaller or
equal to :math:`\text{T}^\text{total}`. The
corresponding set that contains all time steps within one period is
given by

.. math::

   \begin{aligned}
       \mathcal{T}^\text{per period}&~=~&\left\{0,\dots,\text{\small T}^\text{per period}-1\right\}.
   \end{aligned}

An additional time set is required to keep track of storage
inventories. Storage inventories are defined right at the beginning and
at the end of the regular time steps. The set

.. math::

   \begin{aligned}
       \mathcal{T}^\text{per period}_\text{inter}&~=~&\left\{0,\dots,\text{\small T}^\text{per period}\right\} 
   \end{aligned}

gives these momentary points in time. Here, index 0 corresponds to the
beginning of time step 0 and the index
:math:`\text{T}^\text{per period}` corresponds to the end of time step
:math:`\text{T}^\text{per period}-1` within that period,
respectively. The total number of periods :math:`\text{P}^\text{total}`
results from the total number of time steps and the time steps per
period by

.. math::

   \begin{aligned}
       \text{\small P}^\text{total} &~=~& \text{\small T}^\text{total}~/~\text{\small T}^\text{per period}.
   \end{aligned}

The corresponding set that encompasses all of these periods is

.. math::

   \begin{aligned}
       \mathcal{P}^\text{total} &~=~& \left\{0,\dots,\text{\small P}^\text{total}-1\right\}.
   \end{aligned}

Thus, :math:`\vert\mathcal{P}^\text{total}\vert=1` if the energy system
is modeled with the full temporal resolution and
:math:`\vert\mathcal{P}^\text{total}\vert\geq1` if typical periods are
considered. In analogy to the set
:math:`\mathcal{T}^\text{per period}_{\text{inter}}`, the momentary points at the
beginning and at the end of a period are encompassed in the set

.. math::

   \begin{aligned}
       \mathcal{P}^\text{total}_\text{inter} &~=~& \left\{0,\dots,\text{\small P}^\text{total}\right\}.
   \end{aligned}

If typical periods are considered, each regular period *p* is assigned one
of :math:`\text{P}^\text{typical}\in\mathbb{Z}^{+}` typical periods :math:`\text{p}^\text{typical}`. The set encompassing all
typical periods is

.. math::

   \begin{aligned}
       \mathcal{P}^\text{typical} &~=~& \left\{0,\dots,\text{P}^\text{typical}-1\right\},~~\text{with}~\mathcal{P}^\text{typical}\subseteq\mathcal{P}^\text{total}.
   \end{aligned}

The function which maps the regular periods to a typical period is
labeled

.. math::

   \begin{aligned}
       map:\mathcal{P}^\text{total}\rightarrow\mathcal{P}^\text{typical}.\ \label{eqMap}
   \end{aligned}

The frequency *f* with which each period occurs during the total
investigated time is defined as

.. math::

   \begin{aligned}
       f:
       \begin{cases}
           \left\{0\right\} \rightarrow \left\{1\right\} &\text{\small , with full temporal resolution, or}\\
           \mathcal{P}^\text{typical}\rightarrow\mathbb{Z}^{+} &\text{\small , with time series aggregation.}
       \end{cases} 
   \end{aligned}

In the following, all basic operation variables are declared for all
periods or typical periods, depending on whether time series aggregation
is considered, and all time steps within these periods. The
cross-product of these sets is given by

.. math::

   \begin{aligned}
   \mathcal{P}\times\mathcal{T} =
   \begin{cases}
       \mathcal{P}^\text{total}\hspace{0.25cm}\times\mathcal{T}^\text{per period} &\text{\small , for a full temporal resolution, or}\\
       \mathcal{P}^\text{typical}\times\mathcal{T}^\text{per period} &\text{\small , with time series aggregation.}
   \end{cases}
   \end{aligned}

Similarly, the cross-product for keeping track of storage inventories is
defined by

.. math::

   \begin{aligned}
   \mathcal{P}\times\mathcal{T}_\text{inter} =
   \begin{cases}
       \mathcal{P}^\text{total}\hspace{0.25cm}\times\mathcal{T}^\text{per period}_\text{inter} &\text{\small , for a full temporal resolution, and}\\
       \mathcal{P}^\text{typical}\times\mathcal{T}^\text{per period}_\text{inter} &\text{\small , with time series aggregation.}
   \end{cases}
   \end{aligned}

General Parameters
******************
The general parameters are summarized in following table: 

.. list-table:: General parameters

 * - **Parameter**
   - **Domain**
   - **Description**
 * - :math:`\text{T}^\text{total}`
   - :math:`\mathbb{N}`   
   - total number of time steps
 * - :math:`\text{T}^\text{hours}`
   - :math:`\mathbb{R}^{+}`   
   - number of hours per time step
 * - :math:`\text{T}^\text{per period}`
   - :math:`\mathbb{Z}^{+}`   
   - number of time steps per period

Other parameters, e.g., elegibilities of components, are described in :ref:`Basic Component Model`.

Compound Index Sets
*******************

As several variables and parameters are depending on a temporal and operational level, two more compound index sets are added.
To describe the temporal position, the compound index set :math:`\Theta` is introduced: 

.. math::

    \begin{aligned}
    \Theta = \left\{(p,t) ~\vert~ p \in \mathcal{P}, t \in \mathcal{T} \right\}
    \end{aligned}

The operation of the different modeled components can be described by several modes of operation. 
Those are summarized in the set :math:`\mathcal{M}`. All eligible modes for component :math:`c` are described by :math:`M_\text{c}=1`. 
For each component the set of modes of operation is described by 

.. math:: 

    \begin{aligned}
    \mathcal{M}_\text{c}&~=~&\left\{ \text{\small m} ~\vert~ \forall~\text{\small m}\in\mathcal{M}: \text{\small M}_\text{c}=1 \right\}.
    \end{aligned}

The compound index set :math:`\Omega` which describes the modes of operation of a certain component is given by: 

.. math::

    \begin{aligned}
    \Omega = \left\{(c,m) ~\vert~ c \in \mathcal{C}, m \in \mathcal{M}_\text{c} \right\}
    \end{aligned}


