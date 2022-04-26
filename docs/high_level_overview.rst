High Level Overview
~~~~~~~~~~~~~~~~~~~~~~

This is intended to be a high-level documentation of how the model is structured. UML diagrams are used when necessary.

The overall workflow of using Neural Lifetimes is represented by the following diagram:

.. image :: _static/workflow.png
    :align: center
    :width: 600px

Library Functionalities
------------------------

This library contains several parts, each with different functionalities:

- neural_lifetimes: This package contains the source code, including the model, model settings, trainers, and data handlers.
- ml_utils: This package contains the utilities for the event model, including loss functions, encoders, and decoders.
- clickhouse_utils: This package contains the utilities for the clickhouse database.

Details of the utilisation of the user interface can be found in Quickstart.

Model
------

The model is structured as follows:

.. image :: _static/model.png
    :align: center
    :width: 600px


