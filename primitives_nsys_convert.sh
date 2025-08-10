#!/bin/bash

primitives=("icosahedron" "octahedron" "trihexa" "trisurfel" "tetrahedron" "sphere" "diamond")

for primitive in "${primitives[@]}"; do
    nsys export --type json --output traces/primitive_comparison/${primitive}/profile_report_${primitive}.json traces/primitive_comparison/${primitive}/nsys_trace.nsys-rep --force-overwrite=true
done