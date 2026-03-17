# urdf2ontology

# "Extracting Semantics: LLM-Guided Automatic Population of Robot Ontology from URDF" [under review]

This repository contains the implementation of a partially-closed loop pipeline designed to automate the abstraction of robotic hardware. By leveraging the reasoning capabilities of Large Language Models (LLMs) grounded by a domain-expert ontology, the system transforms URDF data into a structured, machine-readable knowledge base (populated ontology).

# Methodology:
The following summarizes the pipeline's methodology for abstracting a given component type (Sensor, End-effector).
```
Step 1. Identification Task:
- Input all URDF leaf links -> Output only links relevant to the component type.
Step 2. Attachment Root Detection Task:
- Input selected kinematic chains (root to leaf) -> Output the component's monting point.
Step 3. Pre-Classification Task:
- Input each trimmed kinematic chain along with component subparts taxonomy from the ontology -> Output a classification for each chain.
Step 4. Group Merging Task:
- Input all trimmed kinematic chains -> Output clustered links in component groups.
Step 5. Post-Classification Task:
- Input each group with their trimmed kinematic chain along with component types taxonomy from the ontology -> Output a classification for each group.
Step 6. Abstraction Verification Task:
- Verify that each component group can be abstracted from the URDF structure without forgetting unidentified links.
Step 7. Ontology Population Task:
- Populate the ontology based on the functional groups identified, with their upper-type, their sub-components and their lower-level types.
```

# Project Structure
```
├── urdf2ontology/
│   ├── urdf/          # Source URDF files for TIAGo, PR2, Pepper
│   ├── results/       # Results with logs and OWL files
│   ├── scripts/       # Scripts to run the abstraction process
│       ├── tasks/     # folder containing each of the python file for the tasks
│       └── main.py   # main python file to run the abstraction
│   └── ontology/      # OWL/TTL files for the domain-expert ontology
└── README.md
```

# Quick Start (Local Setup)
```
1. pip install ollama (or create a virtual environment with ollama)
2. install ontologenius from the dedicated github repository ([link](https://github.com/sarthou/ontologenius))
3. Load the URDF files in urdf2ontology/urdf2ontology/urdf/ folder
4. Add their paths to the main.py
5. Run ontologenius with ros2 launch ontologenius ontologeniu_empty.xml
6. Run the main loop with python3 main.py
```
