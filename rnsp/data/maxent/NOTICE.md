# Third-party splice-site models

The files `score5_matrix.txt` and `score3_matrix.txt` are the donor (5′) and
acceptor (3′) maximum-entropy score matrices used by MaxEntScan.

- Scientific source: Yeo G, Burge CB. Maximum entropy modeling of short
  sequence motifs with applications to RNA splicing signals. *Journal of
  Computational Biology* 11:377–394 (2004).
- Software source of this copy: the `maxentpy` package
  (https://github.com/kepbod/maxentpy), which is distributed under the MIT
  License. The MIT text is in `LICENSE` beside these files.
- `rnsp.maxent` is a small Python port of that scoring code. It is included so
  donor and acceptor delta scores can be reproduced without a network call.
  The matrices are not a reimplementation of Google DeepMind AlphaGenome.
