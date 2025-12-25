# Documentation for `nnfs`
> **Note:** Analyzed via CLI command `nnfs`.
**File Path:** `/usr/local/lib/python3.12/dist-packages/nnfs/__init__.py`

## 🚦 Navigator: How to Drive
This section helps you understand how to run this library from the command line or entry points.

### 💻 Installed CLI Commands
This library installs the following system commands (accessible from terminal):
| Command | Entry Point (Function) |
| :--- | :--- |
| `nnfs` | `nnfs.console.nnfs:main` |

- ✅ **Target Match**: You are analyzing the package backing the command `nnfs`.

### 🐍 Python API Usage (Inferred)
Since no CLI entry point was found, here are the likely **Python API entry points** for your script:

#### 🚀 Top 20 Recommended Entry Points
| Type | API | Description |
| :--- | :--- | :--- |
| `ƒ` | **nnfs.init**(dot_precision_workaround, default_dtype, random_seed) | No description. |

> **Note:** Bold parameters are required. Others are optional.

#### 🧩 Code Snippets (Auto-Generated)
```python
import nnfs

# --- Top 20 Ranked Functions ---
# 1. init
result_1 = nnfs.init()

```

_No explicit `argparse` configuration detected in the main module._


## 📊 Network & Architecture Analysis
### 🌍 Top 20 External Dependencies
| Library | Usage Count |
| :--- | :--- |
| **_frozen_importlib_external** | 8 |
| **_frozen_importlib** | 8 |


### 🕸️ Network Metrics (Advanced)
#### 👑 Top 20 Modules by PageRank (Authority)
| Rank | Module | Score | Type | Role |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `_frozen_importlib_external` | 0.2484 | External | External Lib |
| 2 | `_frozen_importlib` | 0.2484 | External | External Lib |
| 3 | `core` | 0.0734 | Internal | Data Processing |
| 4 | `datasets.sine` | 0.0670 | Internal | Data Processing |
| 5 | `datasets.spiral` | 0.0670 | Internal | Data Processing |
| 6 | `datasets.vertical` | 0.0670 | Internal | Data Processing |
| 7 | `nnfs` | 0.0572 | Internal | Utility / Core |
| 8 | `console` | 0.0572 | Internal | Unknown |
| 9 | `console.nnfs` | 0.0572 | Internal | Utility / Core |
| 10 | `datasets` | 0.0572 | Internal | Utility / Core |


### 🗺️ Dependency & Architecture Map
```mermaid
graph TD
    classDef core fill:#f96,stroke:#333,stroke-width:2px;
    classDef external fill:#9cf,stroke:#333,stroke-width:1px;
    id_8["nnfs"] -.-> id_9["_frozen_importlib_external"]
    class id_8 core;
    class id_9 external;
    id_8["nnfs"] -.-> id_2["_frozen_importlib"]
    class id_8 core;
    class id_2 external;
    id_8["nnfs"] --> id_5["core"]
    class id_8 core;
    class id_5 core;
    id_5["core"] -.-> id_9["_frozen_importlib_external"]
    class id_5 core;
    class id_9 external;
    id_5["core"] -.-> id_2["_frozen_importlib"]
    class id_5 core;
    class id_2 external;
    id_6["console"] -.-> id_9["_frozen_importlib_external"]
    class id_6 core;
    class id_9 external;
    id_6["console"] -.-> id_2["_frozen_importlib"]
    class id_6 core;
    class id_2 external;
    id_3["nnfs"] -.-> id_9["_frozen_importlib_external"]
    class id_3 core;
    class id_9 external;
    id_3["nnfs"] -.-> id_2["_frozen_importlib"]
    class id_3 core;
    class id_2 external;
    id_0["datasets"] -.-> id_9["_frozen_importlib_external"]
    class id_0 core;
    class id_9 external;
    id_0["datasets"] -.-> id_2["_frozen_importlib"]
    class id_0 core;
    class id_2 external;
    id_0["datasets"] --> id_7["sine"]
    class id_0 core;
    class id_7 core;
    id_0["datasets"] --> id_1["spiral"]
    class id_0 core;
    class id_1 core;
    id_0["datasets"] --> id_4["vertical"]
    class id_0 core;
    class id_4 core;
    id_7["sine"] -.-> id_9["_frozen_importlib_external"]
    class id_7 core;
    class id_9 external;
    id_7["sine"] -.-> id_2["_frozen_importlib"]
    class id_7 core;
    class id_2 external;
    id_1["spiral"] -.-> id_9["_frozen_importlib_external"]
    class id_1 core;
    class id_9 external;
    id_1["spiral"] -.-> id_2["_frozen_importlib"]
    class id_1 core;
    class id_2 external;
    id_4["vertical"] -.-> id_9["_frozen_importlib_external"]
    class id_4 core;
    class id_9 external;
    id_4["vertical"] -.-> id_2["_frozen_importlib"]
    class id_4 core;
    class id_2 external;
```

## 🚀 Global Execution Flow & Extraction Guide
This graph visualizes how data flows between functions across the entire project.
```mermaid
graph TD
    classDef main fill:#f9f,stroke:#333,stroke-width:2px;
    classDef func fill:#fff,stroke:#333,stroke-width:1px;
    f_4["main"] --> f_5["print_usage"]
    class f_4 main;
    class f_5 func;
    f_4["main"] --> f_2["info"]
    class f_4 main;
    class f_2 func;
    f_4["main"] --> f_0["code"]
    class f_4 main;
    class f_0 func;
    f_3["init"] -->|method<br>default_dtype| f_1["enclose"]
    class f_3 func;
    class f_1 func;
```

### ✂️ Navigator: Snippet Extractor
Want to use a specific function without the whole library? Here is the **Dependency Closure** for **Top 20** key functions.
#### To extract `main`:
> You need these **4** components:
`code, info, main, print_usage`

#### To extract `init`:
> You need these **2** components:
`enclose, init`

## 📑 Top-Level API Contents & Logic Flow
### 🔧 Functions
#### `init(dot_precision_workaround=True, default_dtype='float32', random_seed=0)`
> No documentation available.
<details><summary>Full Docstring</summary>

```text
No documentation available.
```
</details>