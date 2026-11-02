---
description: How to cite RSATB, its license, and who develops it.
---

<p class="rs-kicker">About</p>

# Citing RSATB

<div class="rs-lead" markdown>
Until the accompanying paper is published, please cite the software itself.
The paper citation will be added here and to
[`CITATION.cff`](https://github.com/DeepElectron/rsatb/blob/main/CITATION.cff)
on publication.
</div>

```bibtex
@software{rsatb,
  author  = {Zhouyin, Zhanghao},
  title   = {{RSATB}: Random State Ab-initio Tight-Binding},
  version = {0.1.0},
  url     = {https://github.com/DeepElectron/rsatb},
  license = {Apache-2.0}
}
```

When reporting results, record the RSATB version and archive the run's
`summary.json`, which records the resolved input.

## Author

RSATB is developed by [Zhanghao Zhouyin](https://floatingcatty.github.io/),
Department of Physics, McGill University ·
[ORCID](https://orcid.org/0000-0002-0954-1650) ·
[Google Scholar](https://scholar.google.com/citations?user=vSMmfvUAAAAJ).

## License

RSATB is available under the
[Apache License 2.0](https://github.com/DeepElectron/rsatb/blob/main/LICENSE).

!!! note "Research software"
    RSATB is under active development. For production science, record the
    version and input, check stochastic convergence with `nsamples`, and check
    the spectral or time resolution of every reported observable.
