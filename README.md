# ProVia

A collection of jupyter notebooks to analyze cell proliferation and viability measurements. Cell proliferation data is modeled using the Gompertz function:

```math
N = k e^{-e^{a-bt}}
```

<center><img src="./docs/gompertz-demo.png" height="300"></center>

ProVia determines the inflection points during proliferation and identifies their sensitivity to cell viability assays (AO/DAPI, ATP, and Annexin).


The documnetation is in ./docs/build/.
