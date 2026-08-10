# Paper Dataset Audit

Archive SHA-256:
`a9c0f446ed4d99bcb7e00cbc3cd686d9fe19149ad3a1015a379e05569992f404`.

| Dataset | Exact author images | Canonical queries | Query provenance | Paper statement | Observed |
|---|---:|---:|---|---:|---:|
| GQA | 1,000 | 15,334 | All released rows associated with exact image IDs | 3/image | 15.334/image |
| VQAv1 | 1,000 | 3,000 | Recovered pre-refactor author manifest | 4/image | 3/image |
| VQAv2 | 1,000 | 5,180 | Released author manifest; one exact duplicate removed | 4/image | 5.18/image |
| RefCOCOg | 1,000 | 1,000 grouped prompts, 1,938 targets | Official annotations joined to exact image IDs | 1/image | 1 grouped prompt/image |

The ZIP contains exactly 1,000 image files under each dataset prefix and no annotation or
query files. The installer
hash-locks the archive, checks every image basename against its manifest, extracts only
those basenames, rejects extra files in the destination, and writes collection hashes to
`dataset_provenance.json`.

The image split is exact for all four datasets. Only VQAv1 and VQAv2 have recoverable
author query manifests. GQA and RefCOCOg query construction is explicit and deterministic,
but it must be reported as reconstructed rather than as an exact released query list.
No additional GQA or RefCOCOg query manifest was found in any pre-refactor archive.
