# Objective Function

In the framework, the objective of the optimization is to minimize the net present value of the specified
energy system. The objective function is defined as

$$
z^* = \min \underset{\text{c} \in \mathcal{C}}{\sum} \ \underset{\text{l} \in \mathcal{L}_\text{c}}{\sum}
\left( NPV^\text{K}_\text{c,l} ~+~ NPV^\text{B}_\text{c,l} ~+~ NPV^\text{O}_\text{c,l} \right)
$$

In detail, the objective function equals

$$
z^* = \min \underset{\text{c} \in \mathcal{C}}{\sum}  \underset{\text{loc} \in \mathcal{L}^\text{comp}}{\sum}  \underset{ip \in \mathcal{IP}}{\sum}  \ design^\text{comp}_\text{loc,ip} + \ design^\text{comp}_{bin, \ loc,ip} + \ op^\text{comp}_\text{loc,ip}
$$

The design variable $design^\text{comp}_\text{loc,ip}$ contributes to the objective function with

$$
design^\text{comp}_\text{loc,ip} =
\sum\limits_{year=ip-\text{ipEconomicLifetime}}^{ip}
\text{F}^\text{comp,bin}_\text{loc,year}
\cdot \left( \frac{\text{investPerCap}^\text{comp}_\text{loc,year}}{\text{CCF}^\text{comp}_\text{loc,year}}
+ \text{opexPerCap}^\text{comp}_\text{loc,year} \right) \cdot \text{commis}^\text{comp}_\text{loc,year}
\cdot  \text{APVF}^\text{comp}_\text{loc} \cdot \text{discFactor}^\text{comp}_\text{loc,ip}
$$

The binary design variables $design^\text{comp}_{\text{bin, loc,ip}}$ contribute to the objective function with

$$
design^\text{comp}_{\text{bin, loc,ip}} =
\sum\limits_{year=ip-\text{ipEconomicLifetime}}^{ip}
\text{F}^\text{comp,bin}_\text{loc,year} \cdot \left( \frac{\text{investIfBuilt}^\text{comp}_\text{loc,year}} {\text{CCF}^\text{comp}_\text{loc,year}}
+ \text{opexIfBuilt}^\text{comp}_\text{loc,year} \right)  \cdot  \text{bin}^\text{comp}_\text{loc,year}
\cdot  \text{APVF}^\text{comp}_\text{loc} \cdot \text{discFactor}^\text{comp}_\text{loc,ip}
$$

The operation variables $op^\text{comp}_\text{loc,ip}$ contribute to the objective function with

$$
op^\text{comp}_\text{loc,ip} =
\underset{(p,t) \in \mathcal{P} \times \mathcal{T}}{\sum} \ \underset{\text{opType} \in \mathcal{O}^{\text{comp}}}{\sum}
\text{factorPerOp}^{\text{comp,opType}}_{\text{loc,ip}} \cdot op^\text{comp,opType}_\text{loc,ip,p,t} \cdot  \frac{\text{freq(p)}}{\tau^{\text{years}}}
\cdot  \text{APVF}^{comp}_{loc} \cdot \text{discFactor}^\text{comp}_\text{loc,ip}
$$

With the annuity present value factor (Rentenbarwertfaktor):

$$
\text{APVF}^\text{comp}_\text{loc} = \begin{cases}
    \dfrac{(1 + \text{interestRate}^\text{comp}_\text{loc})^\text{interval} - 1}{\text{interestRate}^\text{comp}_\text{loc} \cdot (1 + \text{interestRate}^\text{comp}_\text{loc})^\text{interval}} & \text{if } \text{interestRate}^\text{comp}_\text{loc} \neq 0 \\
    1 & \text{else}
\end{cases}
$$

and the discount factor

$$
\text{discFactor}^\text{comp}_\text{loc,ip} = \frac{1+\text{interestRate}^\text{comp}_\text{loc}}{(1+\text{interestRate}^\text{comp}_\text{loc})^{ip \cdot
\text{interval}}}
$$

The general definition of the $NPV^\text{c}$ is given in the [Basic Component Model](basic_component.md).
Specifications of the objective functions in the model extensions are given in the different sections.
