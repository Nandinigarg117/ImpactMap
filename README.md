# ImpactMap

**ImpactMap** is a developer tool and analytics platform that combines static Python dependency analysis with machine learning to estimate change risk, trace component relationships, and prioritize regression testing for Git commits.

## Problem Statement
In large software projects, a small code change can affect multiple connected components, making it difficult for developers to identify all the areas that need testing. Manually tracing these dependencies is time-consuming and can lead to important test cases being missed. ImpactMap aims to analyze code changes, map relationships between software components, and use machine learning to estimate change risk and prioritize potentially affected files and tests. This helps developers make more informed testing decisions and reduces the effort required for code review and regression testing.

---
*Work in progress. See `docs/architecture.md` and `docs/evaluation.md` as modules are built.*
