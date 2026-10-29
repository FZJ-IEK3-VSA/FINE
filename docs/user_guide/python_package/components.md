# Components

Components can be added to an `EnergySystemModel` class to model the behavior of the energy system.
All components have to inherit from the `Component` class. There are five basic component classes:

- `Source` and `Sink` (inherits from Source) classes + the `SourceSinkModel` class
- `Conversion` class + `ConversionModel` class
- `Transmission` class + `TransmissionModel` class
- `Storage` class + `StorageModel` class

From these basic component classes, further subclasses can be defined.

## Component Base Class

::: fine.component
    options:
        show_root_heading: true
        show_source: false

## Source and Sink

::: fine.sourceSink
    options:
        show_root_heading: true
        show_source: false

## Conversion

::: fine.conversion
    options:
        show_root_heading: true
        show_source: false

## Transmission

::: fine.transmission
    options:
        show_root_heading: true
        show_source: false

## Storage

::: fine.storage
    options:
        show_root_heading: true
        show_source: false
