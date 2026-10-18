# Reference for future: -
# make install -> create .venv and install the dependencies
# make train   -> train + evaluate, writes the model and the figures to model_output/
# make serve   -> run the API locally (no docker) on port 8080
# make build   -> train, then build the docker image telco-churn:latest
# make run     -> run the docker image on port 8080
# make clean   -> delete the trained model, the figures and the python cache

# following are the commands
.PHONY: install train serve build run clean

install:
	uv venv --allow-existing .venv
	uv pip install --python .venv/bin/python -r requirements.txt

train:
	.venv/bin/python -m model.train

serve:
	.venv/bin/uvicorn api:app --host 0.0.0.0 --port 8080

# retrain first, so the image always has a model made from the current code
build: train
	docker build -t telco-churn:latest .

run:
	docker run --rm -p 8080:8080 telco-churn:latest

clean:
	rm -rf model_output __pycache__ model/__pycache__
