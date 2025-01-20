Results
=======

XGBoost Regressor Munich_prediction
-----------------------------------


The parameters of the model:


.. literalinclude:: ../results/Munich_prediction/Munich_prediction_pipeline.log
   :caption: Log
   :lines: 1-15


.. figure:: ../results/Munich_prediction/Munich_prediction_actual_vs_predicted.png
   :alt: Actual Vs Predicted

   The actual vs predicted values of the model. With the Pearson correlation coefficient and p-value.


.. figure:: ../results/Munich_prediction/Munich_prediction_shap_aggregated_beeswarm.png
   :alt: Shap Aggregated Beeswarm

   The SHAP values of the model. The effects of the individual features can be read from this plot.


.. figure:: ../results/Munich_prediction/Munich_prediction_shap_aggregated_bar.png
   :alt: Shap Aggregated Bar

   The absolute SHAP values of the model. The feature importances can be read from this plot.


XGBoost Regressor Test_run
--------------------------


The parameters of the model:


.. literalinclude:: ../results/test_run/test_run_pipeline.log
   :caption: Log
   :lines: 1-15


.. figure:: ../results/test_run/test_run_actual_vs_predicted.png
   :alt: Actual Vs Predicted

   The actual vs predicted values of the model. With the Pearson correlation coefficient and p-value.


.. figure:: ../results/test_run/test_run_shap_aggregated_beeswarm.png
   :alt: Shap Aggregated Beeswarm

   The SHAP values of the model. The effects of the individual features can be read from this plot.


.. figure:: ../results/test_run/test_run_shap_aggregated_bar.png
   :alt: Shap Aggregated Bar

   The absolute SHAP values of the model. The feature importances can be read from this plot.

