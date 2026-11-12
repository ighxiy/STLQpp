# Third party components

## hnswlib

`third_party/hnswlib/` contains the header-only hnswlib implementation and its
Apache License 2.0 text (`third_party/hnswlib/LICENSE`). These files were originally
copied from the read-only Rayuela DEV reference:

- Repository branch: `dev`
- Commit: `4a5dbbc493166238810863012a964554de6dad92`
- Source: `chainq/STLQ/third_party/hnswlib/`

Local portability change: initialize the deleted-element replacement ID in
`hnswalg.h` to avoid MSVC's possible-uninitialized-variable warning.

HNSW is compiled only at shared cluster-candidate and IVF-routing boundaries.
It is independent of the reconstruction code length M. See README for the
optional approximate routes and their configuration.
