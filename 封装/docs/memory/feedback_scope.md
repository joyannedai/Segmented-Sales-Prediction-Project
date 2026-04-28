---
name: feedback_scope
description: Only integrate code files within the current working folder, not from sibling or parent folders.
type: feedback
originSessionId: c949e9f2-8959-40ef-b750-25d5e4d966ee
---
**Rule:** When asked to "integrate code in this folder," only process files located within the current working directory (`D:/4080/claude/`). Do not pull in code from sibling folders like `D:/4080/modeling/` unless explicitly requested.

**Why:** The user corrected me mid-task when I initially started integrating code from `D:/4080/modeling/02_代码脚本/`. They said: "是在这个文件夹下的所有代码，不是其他文件夹."

**How to apply:** Before integrating or refactoring, always verify the file paths. If the user refers to "this folder" or "this document," assume they mean the current working directory or the specific file they mentioned. Ask for clarification if ambiguous.
