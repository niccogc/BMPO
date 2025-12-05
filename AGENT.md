# Project Guidelines for BMPO

This document outlines the essential guidelines for working within the BMPO project. Adherence to these principles is mandatory to ensure consistency, efficiency, and maintainability.

## Core Principles

*   **Clarity & Consistency:** All code and documentation must be clear, concise, and consistent with existing project patterns.
*   **Efficiency:** Leverage established tools and libraries to optimize development workflows.

## Enforced Tooling and Libraries

### 1. Project Navigation and Information Retrieval (mcps)

It is **imperative** to utilize the following tools for project navigation and information retrieval:

*   **`docs-mcp`:** This tool **must** be used for querying documentation related to external libraries, especially the `quimb` library.
*   **`serena`:** This tool **must** be used for navigating the project structure, searching for code patterns, and understanding code symbols within the BMPO codebase.

### 2. Tensor Network Operations and Simulations

It is **absolutely critical and mandatory** that the `quimb` library is used for **all** tensor network operations and simulations within this project. Prioritize and exclusively use native `quimb` functions and functionalities. **Under no circumstances** should `quimb`'s core features be reimplemented or bypassed.

Tensor operations **MUST** be done through proper labeling using native quimb.Tensors, that ensures the operations are sound. Keep the tensor form in order to use the labels and use quimb operations.

### 3. General Programming Practices

Where applicable and appropriate, always prefer native Python functions and data structures for general programming tasks. Avoid introducing unnecessary external dependencies.

By strictly following these guidelines, we ensure a robust, efficient, and standardized development environment for the BMPO project.
