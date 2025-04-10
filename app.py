from flask import Flask, request, render_template
import logging
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Initialize Flask application
application = Flask(__name__)
app = application

# Setup logger
logger = logging.getLogger("Iris_Classification")
logger.setLevel(logging.DEBUG)

# Route for Home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    try:
        if request.method == 'GET':
            return render_template('home.html')
        else:
            # Collect input data from the form for Iris dataset
            try:
                sepal_length = float(request.form.get('sepal_length'))
                sepal_width = float(request.form.get('sepal_width'))
                petal_length = float(request.form.get('petal_length'))
                petal_width = float(request.form.get('petal_width'))
            except ValueError:
                return render_template('home.html', results="Please enter valid numeric values for all fields.")

            # Log the received input
            logger.info(f"Received input: Sepal Length={sepal_length}, Sepal Width={sepal_width}, "
                        f"Petal Length={petal_length}, Petal Width={petal_width}")

            # Create a CustomData instance for prediction
            data = CustomData(
                sepal_length=sepal_length,
                sepal_width=sepal_width,
                petal_length=petal_length,
                petal_width=petal_width
            )

            # Convert data to DataFrame for prediction
            pred_df = data.get_data_as_data_frame()
            logger.info(f"Prepared data for prediction: {pred_df}")

            # Initialize the prediction pipeline and predict
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            logger.info(f"Prediction result: {results}")

            # Return the result to the frontend
            return render_template('home.html', results=results[0])  # Assuming the result is list-like

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        return render_template('home.html', results="There was an error with the input data or prediction.")

# Main driver function
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
