##########
Components
##########

Components can be added to an EnergySystemModel class to model the behavior of the energy system. The data of these
components are stored in so called component classes. All components have to inherit from the "Component" class.
The behavior of the components in the energy system is modeled in respective component modeling classes. All component
modeling classes have to inherit from the "ComponentModel" class. There are five basic component and component modeling
classes in FINE. These are

* Source and Sink (inherits from Source) classes + the SourceSinkModel class,
* Conversion class + ConversionModel class,
* Transmission class + TransmissionModel class, and
* Storage class + StorageModel class.

Form these basic component and component modeling classes, further subclasses can be defined. For example, a
LinearOptimalPowerFlow + LOPFModel class inherit from the Transmission and TransmissionModel class.

**Component and ComponentModeling class**

.. toctree::
   :maxdepth: 1

   components/componentClassDoc

**Basic component and component modeling classes**

.. toctree::
   :maxdepth: 1

   components/sourceSinkClassDoc
   components/conversionClassDoc
   components/transmissionClassDoc
   components/storageClassDoc

**Extended subclasses**

.. toctree::
   :maxdepth: 1

   components/subclasses/conversionPartLoadClassDoc
   components/subclasses/conversionDynamicClassDoc
   components/subclasses/lopfClassDoc
   components/subclasses/DSMClassDoc
   components/subclasses/storageExtClassDoc