.. InsideAirBnB documentation master file, created by
   sphinx-quickstart on Sun Jan 19 22:00:56 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Results
=======

XGBoost Regressor Berlin
------------------------

The parameters of the model:

.. literalinclude:: ../results/test_run/test_run_pipeline.log
   :caption: Log
   :lines: 1-15


First the performance of the model:

.. figure:: ../results/test_run/test_run_actual_vs_predicted.png
   :alt: Image 1

   The actual vs predicted values of the model. With the Pearson correlation coefficient and p-value.

.. figure:: ../results/test_run/test_run_shap_aggregated_beeswarm.png
   :alt: Image 2

   The SHAP values of the model. The effect ofs of the individual features can be read from this plot.

.. figure:: ../results/test_run/test_run_shap_aggregated_bar.png
   :alt: Image 3

   The absolute SHAP values of the model. The features importances ofs of the individual features can be read from this plot.

