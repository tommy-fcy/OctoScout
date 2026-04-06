---
description: Query the OctoScout compatibility matrix for known issues between ML package versions.
argument-hint: "<pkg1==ver1 pkg2==ver2> or 'check' or 'stats'"
allowed-tools: "Bash(octoscout *), Read"
---

# OctoScout Compatibility Matrix

Query the compatibility matrix: $ARGUMENTS

## Instructions

Based on what the user wants:

### Query specific versions
If the user provides package==version pairs:
```
octoscout matrix query transformers==4.55.0 torch==2.3.0
```

### Check current environment
If the user says "check" or wants to scan their environment:
```
octoscout matrix check --auto-env
```

### Show statistics
If the user says "stats" or wants an overview:
```
octoscout matrix stats
```

### Generate heatmap
If the user wants a visual:
```
octoscout matrix heatmap
```
Then tell them the heatmap has been opened in their browser.

Present results clearly with risk levels and recommended actions.
