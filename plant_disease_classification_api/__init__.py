#!/usr/bin/env python

"""ML API for plant disease classification."""

from __future__ import absolute_import

__author__ = "Abdullah Selek"
__email__ = "abdullahselek.os@gmail.com"
__copyright__ = "Copyright (c) 2020 Abdullah Selek"
__license__ = "MIT License"
__version__ = "0.1"
__url__ = "https://github.com/abdullahselek/plant-disease-classification-api"
__download_url__ = "https://github.com/abdullahselek/plant-disease-classification-api"
__description__ = "ML API for plant disease classification."

from models import ClassficationRequestItem
from ml import constant
from ml.network import CNN
from ml.plant_disease_classifier import (
    PlantDiseaseClassifier,
)
from plant_disease_classification_api import main
