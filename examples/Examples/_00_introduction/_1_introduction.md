# Basic Example: Building an Energy System Model with FINE

This example provides a step-by-step introduction to energy system modeling with **FINE**. It guides users through the complete workflow, from initializing an energy system model to adding components and analyzing optimization results.

The objective of this example is to demonstrate the core concepts of FINE and to illustrate how different model components interact within an energy system. Each notebook focuses on a specific step of the modeling process and can be explored either individually or as part of the full workflow.

---

## Workflow Overview

The example is organized into the following notebooks:

### 1. Initializing an Energy System Model

This notebook introduces the `EnergySystemModel`, which serves as the central container of every FINE model. It explains how to define locations, commodities, temporal resolution, and other fundamental settings that establish the structure of the energy system.

### 2. Adding Sources and Sinks

This notebook demonstrates how to represent commodity imports, exports, demands, and other interactions between the energy system and its environment using Source and Sink components.

### 3. Adding Conversion Components

This notebook demonstrates how to represent processes that can transform one commodity into another (e.g., converting natural gas into electricity) using Conversion components.

### 4. Adding Storage Components

This notebook demonstrates how to represent energy storage technologies using Storage components.

### 5. Adding Transmission Components

This notebook demonstrates how to represent the transportation of commodities between locations using Transmission components.

### 6. Optimizing and Analyzing Results

This notebook demonstrates how to run the optimization, inspect results, and interpret the behavior of the modeled energy system.

---

## Energy System Structure

The figure below illustrates the components used in this basic example and their interactions within the energy system.

put the grafik here

The energy system consists of:

- **Sources**: provide commodities to the system  
- **Sinks**: represent demands or exports  
- **Conversions**: transform commodities  
- **Storages**: shift commodities over time  
- **Transmissions**: transport commodities between locations  
- The **EnergySystemModel**: connects all components and defines the optimization problem  

---

By completing this example, users will gain a solid understanding of the core modeling principles in FINE and will be able to build their own energy system models.