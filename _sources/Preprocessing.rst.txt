.. InsideAirBnB documentation master file, created by
   sphinx-quickstart on Sun Jan 19 22:00:56 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Preprocessing
=============

The data was automatically downloaded from the InsideAirBnB website:
`InsideAirBnB <http://insideairbnb.com/get-the-data.html>`_

Then the data was aggregated into a single csv file for each city by the following script:

.. literalinclude:: ../preprocessing/preprocessing.py
   :language: python
   :caption: preprocessing.py