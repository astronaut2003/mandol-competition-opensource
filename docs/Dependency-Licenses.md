# Dependency license inventory

This inventory covers the application dependencies in `uv.lock` and the local
inference bridge package. Upstream license metadata describes each distribution;
it does not grant rights to unrelated assets. Third-party wheels and model
weights are installed separately and are not bundled with this project.

| Distribution | Application lock / local package version | Declared license |
|---|---|---|
| annotated-doc | 0.0.4 | MIT |
| annotated-types | 0.7.0 | MIT |
| anyio | 4.13.0 | MIT |
| certifi | 2026.4.22 | MPL-2.0 |
| click | 8.3.3 | BSD-3-Clause |
| cloudpickle | 3.1.2 | BSD-3-Clause |
| colorama | 0.4.6 | BSD-3-Clause |
| distro | 1.9.0 | Apache License, Version 2.0 |
| fastapi | 0.136.3 | MIT |
| grpcio | 1.80.0 | Apache-2.0 |
| h11 | 0.16.0 | MIT |
| h2 | 4.4.1 | MIT |
| hpack | 4.2.0 | MIT |
| httpcore | 1.0.9 | BSD-3-Clause |
| httpx | 0.28.1 | BSD-3-Clause |
| hyperframe | 6.1.0 | MIT |
| idna | 3.13 | BSD-3-Clause |
| iniconfig | 2.3.0 | MIT |
| jiter | 0.14.0 | MIT |
| joblib | 1.6.0 | BSD-3-Clause |
| mandol-competition | 0.1.0 | Apache-2.0 |
| mandol-inference-worker | 0.1.0 | Apache-2.0 |
| mandol-substrate | 0.1.0 | Apache-2.0 |
| networkx | 3.6.1 | BSD-3-Clause |
| nltk | 3.9.1 | Apache-2.0; benchmark extra |
| numpy | 1.26.4 | BSD-3-Clause; wheel bundles additional licenses |
| openai | 2.32.0 | Apache-2.0 |
| packaging | 26.1 | Apache-2.0 OR BSD-2-Clause |
| pillow | 12.2.0 | MIT-CMU |
| pluggy | 1.6.0 | MIT |
| portalocker | 3.2.0 | BSD-3-Clause |
| protobuf | 7.34.1 | 3-Clause BSD License |
| pydantic | 2.13.3 | MIT |
| pydantic_core | 2.46.3 | MIT |
| Pygments | 2.20.0 | BSD-2-Clause |
| pytest | 9.0.2 | MIT |
| pywin32 | 312 | PSF and bundled component notices; Windows only |
| qdrant-client | 1.19.0 | Apache-2.0 |
| regex | 2026.9.29 | Apache-2.0 AND CNRI-Python |
| rustworkx | 0.17.1 | Apache-2.0 |
| sniffio | 1.3.1 | MIT OR Apache-2.0 |
| starlette | 1.3.1 | BSD-3-Clause |
| tqdm | 4.67.3 | MPL-2.0 AND MIT |
| tree-sitter | 0.25.2 | MIT |
| tree-sitter-language-pack | 1.14.3 | MIT |
| typing-inspection | 0.4.2 | MIT |
| typing_extensions | 4.15.0 | PSF-2.0 |
| urllib3 | 2.6.3 | MIT |
| uvicorn | 0.49.0 | BSD-3-Clause |

NumPy binary wheels include OpenBLAS, LAPACK, GCC runtime and quadmath notices,
including runtime-exception and LGPL terms; its primary BSD label does not
replace those notices. Preserve the upstream wheel notices if separately
redistributing binary dependencies. certifi's MPL-2.0 and tqdm's MPL/MIT terms
remain with those distributions. Tree-sitter grammar downloads, optional model
workers and separate vLLM environments require their own artifact reviews.
The application installation does not install Torch or Transformers. The
inference package's optional `models` extra uses Torch 2.8.0 (BSD-3-Clause),
Transformers 4.57.6 (Apache-2.0) and Sentence Transformers 5.4.1 (Apache-2.0).
Their transitive dependencies and separate serving environments retain their
own terms. Model and provider terms are described in [Attribution](ATTRIBUTION.md).
