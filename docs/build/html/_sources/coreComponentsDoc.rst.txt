**********************
Core component classes
**********************

.. |br| raw:: html

   <br />

The core component classes of this framework are the

* Source and Sink (inherits from Source) classes
* Conversion class
* Transmission class
* Storage class

Each of these classes inherits from the Component class. Instances (with concrete values) of these classes can be added
to an EnergySystemModel class to specify a concrete energy system.

Component class
###############

**Class description:**


.. automodule:: component
.. autoclass:: Component
   :members:
   :member-order: bysource

   .. automethod:: __init__

**Inheritance diagram:**

.. inheritance-diagram:: Component
   :parts: 1

Source and Sink class
#####################

**Class description:**

.. automodule:: sourceSink
.. autoclass:: Source
   :members:
   :member-order: bysource

   .. automethod:: __init__

**Inheritance diagram:**

.. inheritance-diagram:: Source
   :parts: 1

**Class description:**

.. autoclass:: Sink
   :members:
   :member-order: bysource

   .. automethod:: __init__

**Inheritance diagram:**

.. inheritance-diagram:: Sink
   :parts: 1


Conversion class
################

**Class description:**

.. automodule:: conversion
.. autoclass:: Conversion
   :members:
   :member-order: bysource

   .. automethod:: __init__

**Inheritance diagram:**

.. inheritance-diagram:: Conversion
   :parts: 1

Transmission class
##################

**Class description:**

.. automodule:: transmission
.. autoclass:: Transmission
   :members:
   :member-order: bysource

   .. automethod:: __init__

**Inheritance diagram:**

.. inheritance-diagram:: Transmission
   :parts: 1

Storage class
#############

**Class description:**

.. automodule:: storage
.. autoclass:: Storage
   :members:
   :member-order: bysource

   .. automethod:: __init__

**Inheritance diagram:**

.. inheritance-diagram:: Storage
   :parts: 1

