# Release validation

Validation on 10 November 2026 used the recorded Python 3.9 and PyTorch 2.8 environment. The release checks used CPU execution. They did not repeat full training, all attacks, or the full bootstrap report.

| Check | Result |
| --- | --- |
| Automated tests | 22 passed |
| Shell script syntax | All launchers passed `bash -n` |
| Raw annotation reconstruction | Both datasets matched ordered identifiers and label checksums |
| Existing metadata relocation | Both datasets matched the same references |
| BDD100K validation repair | 6,986 retained validation images |
| nuImages partition audit | Training, validation, and test partitions disjoint |
| Fixed evaluation manifests | 1,000 images per dataset |
| Released checkpoint tensors | Every tensor matched its evaluated source checkpoint |
| Checkpoint installation | Archive and member checksums verified; repeated installation accepted |
| Model checks | Four models loaded and completed four Square queries on one image each |
| Controlled inference | CC and AA matched native inference for all four models |
| Queue construction | Six queues, 480 rows, 400 distinct runs |
| tmux worker check | Three dry-run workers returned exit code zero |
| Paper displays | Six tables and three figures generated from aggregate results |

The automated tests cover metrics, perturbation and query bounds, controlled inference, result integrity, report inputs, data reconstruction, checkpoint installation, and configuration precedence. Library deprecation warnings occurred during testing. No test failed.

The bounded model checks establish that the packaged architectures and weights execute together. They do not establish identical full-run numerical results across devices or acquisition-bank histories. The README describes those reproduction limits.

`SHA256SUMS` records the files in the prepared source release. From the repository root, run `sha256sum -c SHA256SUMS` before editing files to check their integrity. The checkpoint archive has a separate checksum in `artifacts/checkpoints.json`.
