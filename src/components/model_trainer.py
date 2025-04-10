import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting Train and Test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                #"AdaBoost Classifier": AdaBoostClassifier(),
                #"Gradient Boosting": GradientBoostingClassifier(),
                #"XGBClassifier": XGBClassifier(),
                #"CatBoosting Classifier": CatBoostClassifier(verbose=False),
            }
            
            params = {
                "Decision Tree": {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [3, 5, 10, None],
                    'min_samples_split': [2, 5, 10]
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128],
                    'max_depth': [3, 5, 10, None],
                    'min_samples_split': [2, 5, 10]
                },
                # "Gradient Boosting": {
                #     'learning_rate': [0.1, 0.05, 0.01],
                #     'n_estimators': [50, 100, 200],
                #     'max_depth': [3, 5, 10]
                # },
                "Logistic Regression": {
                    'penalty': ['l2'],
                    'solver': ['lbfgs'],  # Ensure compatibility with 'l2' penalty
                },
                # "XGBClassifier": {
                #     'learning_rate': [0.1, 0.05, 0.01],
                #     'n_estimators': [50, 100, 200]
                # },
                # "CatBoosting Classifier": {
                #     'depth': [6, 8, 10],
                #     'learning_rate': [0.01, 0.05, 0.1],
                #     'iterations': [30, 50, 100]
                # },
                # "AdaBoost Classifier": {
                #     'learning_rate': [0.1, 0.01, 0.5],
                #     'n_estimators': [50, 100, 200]
                # }
            }

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                 models=models, param=params)

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            MINIMUM_SCORE_THRESHOLD = 0.6
            if best_model_score < MINIMUM_SCORE_THRESHOLD:
                raise CustomException(f"Best model score is {best_model_score}, which is below the threshold of {MINIMUM_SCORE_THRESHOLD}")

            logging.info(f"Best model found: {best_model_name} with score {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info(f"Model saved at {self.model_trainer_config.trained_model_file_path}")

            predicted = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, predicted)
            return accuracy

        except Exception as e:
            logging.error(f"Error occurred: {str(e)}", exc_info=True)
            raise CustomException(e, sys)
