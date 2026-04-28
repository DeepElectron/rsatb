### Coding Style & Architecture Directives

**Core Standard:** All generated code must strictly prioritize functionality, maintainability, and readability. 

**Rule 1: Concrete over Abstract**
* **Directive:** Favor concrete, straightforward implementations. Do not introduce high-level abstractions speculatively or prematurely to avoid unnecessary cognitive load.
* **Exception:** You may only introduce abstraction if the code has stabilized and the abstraction demonstrably improves design, reusability, and readability.

**Rule 2: Flat Hierarchy**
* **Directive:** Maintain a flat codebase architecture. Do not create deep vertical slices or complex layer designs. 
* **Approach:** Default to a function-based division of logic. Use hybrid structural designs only when they explicitly simplify the specific problem at hand.

**Rule 3: Pragmatic API Design**
* **Directive:** Balance semantic correctness with practical readability. Do not over-engineer APIs for the sake of strict semantic purity.
* **Constraint:** Avoid breaking unified, easily understood functions into heavily fragmented, rarely-used micro-APIs. Keep cohesive logic grouped together.

**Rule 4: The "Nearest Neighbor" Principle (Colocation)**
* **Directive:** Code with similar logical, functional, or design structures must be grouped close to each other (e.g., in the same file or adjacent files). Optimize for easy context gathering and maintenance.

**Rule 5: Balanced Modularization**
* **Directive:** Create modules that are entirely self-contained, complete, and independent. 
* **Constraint:** Avoid both extremes: do not create monolithic "god modules" that house unrelated files, and do not over-fragment into micro-submodules that separate concrete, related functions. Group files logically based on the Nearest Neighbor principle.