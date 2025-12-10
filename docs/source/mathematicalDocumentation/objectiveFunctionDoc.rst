Objective Function
##################

In the framework, the objective of the optimization is to minimize the
net present value of the specified energy system. The objective function
is defined as

.. math::
    z^* = \min \underset{\text{c} \in \mathcal{C}}{\sum} \ \underset{\text{l} \in \mathcal{L}_\text{c}}{\sum}
    \left( NPV^\text{K}_\text{c,l} ~+~ NPV^\text{B}_\text{c,l} ~+~ NPV^\text{O}_\text{c,l} \right) 

In detail, the objective functions equals

.. math::
    z^* = \min \underset{\text{c} \in \mathcal{C}}{\sum}  \underset{\text{loc} \in \mathcal{L}^\text{comp}}{\sum}  \underset{ip \in \mathcal{IP}}{\sum}  \ design^\text{comp}_\text{loc,ip} + \ design^\text{comp}_{bin, \ loc,ip} + \ op^\text{comp}_\text{loc,ip}

The design variable :math:`design^\text{comp}_\text{loc,ip}` contributes to the objective function with 

.. math::
        design^\text{comp}_\text{loc,ip} =
        \sum\limits_{year=ip-\text{ipEconomicLifetime}}^{ip}
        \text{F}^\text{comp,bin}_\text{loc,year}
        \cdot \left( \frac{\text{investPerCap}^\text{comp}_\text{loc,year}}{\text{CCF}^\text{comp}_\text{loc,year}}
        + \text{opexPerCap}^\text{comp}_\text{loc,year} \right) \cdot commis^\text{comp}_\text{loc,year}
        \cdot  \text{APVF}^\text{comp}_\text{loc} \cdot \text{discFactor}^\text{comp}_\text{loc,ip}

The binary design variables :math:`design^\text{comp}_\text{bin\ loc,ip}` contribute to the objective function with 

.. math::
        design^\text{comp}_\text{bin, \ loc,ip} =
        \sum\limits_{year=ip-\text{ipEconomicLifetime}}^{ip}
        \text{F}^\text{comp,bin}_\text{loc,year} \cdot \left( \frac{\text{investIfBuilt}^\text{comp}_\text{loc,year}} {\text{CCF}^\text{comp}_\text{loc,year}}
        + \text{opexIfBuilt}^\text{comp}_\text{loc,year} \right)  \cdot  bin^\text{comp}_\text{loc,year}
        \cdot  \text{APVF}^\text{comp}_\text{loc} \cdot discFactor^\text{comp}_\text{loc,ip}

The operation variables :math:`op^\text{comp}_\text{loc,ip}` contribute to the objective function with

.. math::
        op^\text{comp}_\text{loc,ip} =
        \underset{(p,t) \in \mathcal{P} \times \mathcal{T}}{\sum} \ \underset{\text{opType} \in \mathcal{O}^{comp}}{\sum}
        \text{factorPerOp}^{comp,opType}_{loc,ip} \cdot op^\text{comp,opType}_\text{loc,ip,p,t} \cdot  \frac{\text{freq(p)}}{\tau^{years}}
        \cdot  \text{APVF}^{comp}_{loc} \cdot \text{discFactor}^\text{comp}_\text{loc,ip}

With the annuity present value factor (Rentenbarwertfaktor):

.. math::
    \text{APVF}^{comp}_{loc} = \frac{(1 + \text{interestRate}^{comp}_{loc})^{interval} - 1}{\text{interestRate}^{comp}_{loc} \cdot
    (1 + \text{interestRate}^{comp}_{loc})^{interval}} \ if \text{interestRate}^{comp}_{loc} != 0 \  else \  1

and the discount factor.

.. math::
    \text{discFactor}^{comp}_{loc,ip} = \frac{1+\text{interestRate}^{comp}_{loc}}{(1+\text{interestRate}^{comp}_{loc})^{ip \cdot
    \text{interval}}}

The general definition of the :math:`NPV^\text{c}` is given in the :ref:`Basic Component Model`. 
Specifications of the objective functions in the model extensions are given in the different sections.
