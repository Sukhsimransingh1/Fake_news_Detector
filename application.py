from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    news_text = ""

    if request.method == "POST":
        news_text = request.form.get("news")
        predictor = PredictPipeline()
        prediction = predictor.predict(news_text)

    return render_template(
        "index.html",
        prediction=prediction,
        news_text=news_text
    )


if __name__ == "__main__":
    app.run(debug=True)
