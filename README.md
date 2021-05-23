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
pip install requirements.txt
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

### Serving the Model
- Execute `sudo docker build -t <TAG> . && sudo docker run -p<PORT>:80 <TAG>` to run the server in a container.
- You can use Postman to send a payload to get a prediction.

Payload example:
```python
{
    "host_response_time": "within an hour",
     "host_is_superhost": "t",
     "room_type": "Entire home/apt",
     "accommodates": 5,
     "bathrooms": 1.0,
     "bedrooms": 8.0,
     "beds": 2.0,
     "minimum_nights": 345,
     "maximum_nights": 9,
     "number_of_reviews": 3000,
     "instant_bookable": "t",
     "days_since_host": 4117.0,
     "half_bath": "no",
     "delta_nights": 175,
     "mean_reviews": 0.9961685823754789,
     "regiao": "zona_sul",
     "property_type_refactor": "other",
     "is_host_rj": "yes",
     "count_name": 41,
     "count_description": 835,
     "count_neighborhood_overview": 121,
     "count_host_about": 144
     }
