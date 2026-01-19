"""Tests for the EmergencyIntentModel class."""

import pytest
import numpy as np
import os
import tempfile
from unittest.mock import patch, MagicMock

from model import EmergencyIntentModel
from config import MODEL_INPUT_SHAPE, NUM_CLASSES


class TestEmergencyIntentModel:
    """Test cases for EmergencyIntentModel."""

    def test_model_initialization(self):
        """Test that model initializes correctly."""
        model = EmergencyIntentModel()
        assert model.input_shape == MODEL_INPUT_SHAPE
        assert model.num_classes == NUM_CLASSES
        assert model.model is None
        assert model.interpreter is None

    def test_build_model(self):
        """Test model building."""
        model = EmergencyIntentModel()
        tf_model = model.build_model()

        assert tf_model is not None
        assert model.model is tf_model

        # Check input shape
        assert tf_model.input_shape == (None, *MODEL_INPUT_SHAPE)

        # Check output shape
        assert tf_model.output_shape == (None, NUM_CLASSES)

    @patch('tensorflow.lite.TFLiteConverter.from_keras_model')
    def test_quantize_model(self, mock_converter):
        """Test model quantization."""
        # Mock the converter and its methods
        mock_tflite_model = b'fake_quantized_model'
        mock_converter.return_value.convert.return_value = mock_tflite_model

        model = EmergencyIntentModel()
        model.model = MagicMock()  # Mock the built model

        with tempfile.TemporaryDirectory() as temp_dir:
            model.model_path = os.path.join(temp_dir, 'test_model.tflite')
            quantized_model = model.quantize_model()

            assert quantized_model == mock_tflite_model
            assert os.path.exists(model.model_path)

            # Verify the file was written
            with open(model.model_path, 'rb') as f:
                assert f.read() == mock_tflite_model

    def test_predict_without_loaded_model(self):
        """Test that predict raises error when model not loaded."""
        model = EmergencyIntentModel()

        with pytest.raises(RuntimeError, match="Model not loaded"):
            model.predict(np.random.rand(*MODEL_INPUT_SHAPE))

    def test_benchmark_latency_without_loaded_model(self):
        """Test that benchmark raises error when model not loaded."""
        model = EmergencyIntentModel()

        with pytest.raises(RuntimeError, match="Model not loaded"):
            model.benchmark_latency()

    def test_get_model_info_without_loaded_model(self):
        """Test get_model_info when model not loaded."""
        model = EmergencyIntentModel()
        info = model.get_model_info()

        assert info == {"status": "Model not loaded"}