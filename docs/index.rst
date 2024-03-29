.. ProVia documentation master file, created by
   sphinx-quickstart on Wed Jan  5 11:09:49 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ProVia's documentation
=================================

ProVia contains jupyter notebooks and modules for the following sets of data analysis using cell proliferation and cell viability data.

#. Fit cell proliferation data with Gompertz model to extract the inflection points during proliferation.
#. Determine the sensitivity of cell viability assays to the proliferation inflection points.
#. Area Under the Curve (AUC) analysis for each viability assay for a changing interval of inflection points.

The current version of ProVia uses the data from the following cell viability assays:

#. AO/DAPI % viability for membrane integrity.
#. LDH cytotoxicity.
#. Annexin and PI staining for apoptosis.
#. ATP metabolic assay.

.. image:: ./gompertz-demo.png
  :width: 550
  :align: center
  :alt: Gompertz model demo

:Download:
 `github.com/sarkar-s/ProVia <https://github.com/sarkar-s/ProVia.git>`_

:References:
 Please cite the following reference if you are using parts of this package:

 *Laura Pierce, Hidayah Anderson, Swarnavo Sarkar, Steven Bauer, Sumona Sarkar. Approach for establishing fit-for-purpose cell viability measurements that are sensitive to proliferative capacity in a model system (2022).*

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   Create_Gompertz_fits.ipynb
   Plot_inflection_points.ipynb
   AODAPI_vs_T2-from-proliferation.ipynb
   cytotoxicity-vs-T2-from-AODAPI.ipynb
   Annexin-vs-T2-from-AODAPI.ipynb
   ATP-vs-T2-from-AODAPI.ipynb
   functions

.. toctree::
   :maxdepth: 2
   :caption: Contents:
