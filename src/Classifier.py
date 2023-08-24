import os
import warnings
import pandas as pd
import h2o
from h2o.model import ModelBase
from h2o.automl import H2OAutoML
from sklearn.exceptions import NotFittedError
from schema.data_schema import MulticlassClassificationSchema

warnings.filterwarnings("ignore")

PREDICTOR_FILE_NAME = 'predictor.joblib'


class Classifier:
    """A wrapper class for the multiclass classifier.

        This class provides a consistent interface that can be used with other
        classifier models.
    """

    model_name = 'h2o_multiclass_classifier'

    def __init__(self, train_input: h2o.H2OFrame, schema: MulticlassClassificationSchema):
        """Construct a new Multiclass Classifier."""
        self._is_trained = False
        self.schema = schema
        self.training_df = train_input
        x = train_input.columns
        x.remove(schema.id)
        x.remove(schema.target)
        self.y = schema.target
        self.aml = H2OAutoML(max_models=5, seed=10, nfolds=10, verbosity='info')
        self.x = x
        self.training_df[schema.target] = self.training_df[schema.target].asfactor()

    def train(self):
        self.aml.train(x=self.x, y=self.y, training_frame=self.training_df)
        self._is_trained = True

    def predict_proba(self, inputs: pd.DataFrame) -> h2o.H2OFrame:
        """Predict class probabilities for the given data.

        Args:
            inputs (h2o.H2OFrame): The input data.
        Returns:
            h2o.H2OFrame: The predicted class probabilities.
        """
        return self.aml.leader.predict(inputs)

    def save(self, model_dir_path: str) -> None:
        """Save the classifier to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """

        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        h2o.save_model(self.aml.leader, path=model_dir_path, filename=PREDICTOR_FILE_NAME, force=True)

    @classmethod
    def load(cls, model_dir_path: str) -> ModelBase:
        """Load the classifier from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            ModelBase: A new instance of the loaded classifier.
        """
        return h2o.load_model(path=os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def predict_with_model(cls, model: ModelBase, data: h2o.H2OFrame) -> h2o.H2OFrame:
        """
        Predict class probabilities for the given data.

        Args:
            model (ModelBase): The classifier model.
            data (h2o.H2OFrame): The input data.

        Returns:
            h2o.H2OFrame: The predicted classes or class probabilities.
        """
        return model.predict(data)

    @classmethod
    def save_predictor_model(cls, model: ModelBase, predictor_dir_path: str) -> None:

        """
        Save the classifier model to disk.

        Args:
            model (ModelBase): The classifier model to save.
            predictor_dir_path (str): Dir path to which to save the model.
        """
        if not os.path.exists(predictor_dir_path):
            os.makedirs(predictor_dir_path)
        model.save(predictor_dir_path)

    @classmethod
    def load_predictor_model(cls, predictor_dir_path: str) -> ModelBase:
        """
        Load the classifier model from disk.

        Args:
            predictor_dir_path (str): Dir path where model is saved.

        Returns:
            Classifier: A new instance of the loaded classifier model.
        """
        return Classifier.load(predictor_dir_path)
