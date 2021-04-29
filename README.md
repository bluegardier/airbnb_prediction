# Rental Price Prediction

### Project Description
The main purpose of this project is just to practice good practices of traditional software, from data's ingestion to model deployment.

This project is a regression type that estimate the rental price of a local of interest aimed only for practicing.

### Data Dependencies
We are using the following Data Source:

| Source | Description | Year |
|--------|-------------|------|
|Airbnb Listings RJ| Prices from RJ's rental places. It's advertisements contains other informations like Description, Room Type, Neighnorhood and so on.| 2020|

### Usage
To run this project locally:

```
git clone https://github.com/bluegardier/airbnb_prediction.git
cd airbnb_prediction
pip install .
``` 

### Repository Structure
- `airbnb_prediction`: Project modules.
- `app`: API configuration for requests. Uses FastAPI.
- `data`: Project's data containing both raw and clean data for modelling and the model's binary.
- `notebooks`: Jupyter Notebooks containing both feature engineering proccess and the model stage.
- `requirements.txt`: contains python dependencies to reproduce the experiments.

### Running the Project
- `python main.py --help`: Shows usage information.
- `python main.py features`: Generate features
- `python main.py deploy_model`: Deploy model
- `python main.py evaluate_model`: Evaluate model
- `python main.py run`: Run all model pipeline steps sequentially