from pathlib import Path

data_dir_raw = "../data/raw/"
data_dir_processed = Path("../data/processed/")

model_path = "../data/model/"

zona_sul = [
    "Botafogo",
    "Catete",
    "Copacabana",
    "Cosme Velho",
    "Flamengo",
    "Gávea",
    "Glória",
    "Humaitá",
    "Ipanema",
    "Jardim Botânico",
    "Jardin Botânico",
    "Lagoa",
    "Laranjeiras",
    "Leblon",
    "Leme",
    "São Conrado",
    "Urca",
    "Vidigal",
    "Rocinha",
]

zona_norte = [
    "Alto da Boa Vista",
    "Andaraí",
    "Grajaú",
    "Maracanã",
    "Praça da Bandeira",
    "Tijuca",
    "Vila Isabel",
    "Abolição",
    "Água Santa",
    "Cachambi",
    "Del Castilho",
    "Encantado",
    "Engenho de Dentro",
    "Engenho Novo",
    "Higienópolis",
    "Jacaré",
    "Jacarezinho",
    "Lins de Vasconcelos",
    "Manguinhos",
    "Maria da Graça",
    "Méier",
    "Piedade",
    "Pilares",
    "Riachuelo",
    "Rocha",
    "Sampaio",
    "São Francisco Xavier",
    "Todos os Santos",
    "Bonsucesso",
    "Bancários",
    "Cacuia",
    "Cidade Universitária",
    "Cocotá",
    "Freguesia (Ilha)",
    "Galeão",
    "Jardim Carioca",
    "Jardim Guanabara",
    "Maré",
    "Moneró",
    "Olaria",
    "Pitangueiras",
    "Portuguesa",
    "Praia da Bandeira",
    "Ramos",
    "Ribeira",
    "Tauá",
    "Zumbi",
    "Acari",
    "Anchieta",
    "Barros Filho",
    "Bento Ribeiro",
    "Brás de Pina",
    "Campinho",
    "Cavalcanti",
    "Cascadura",
    "Coelho Neto",
    "Colégio",
    "Complexo do Alemão",
    "Cordovil",
    "Costa Barros",
    "Engenheiro Leal",
    "Engenho da Rainha",
    "Guadalupe",
    "Honório Gurgel",
    "Inhaúma",
    "Irajá",
    "Jardim América",
    "Madureira",
    "Marechal Hermes",
    "Osvaldo Cruz",
    "Parada de Lucas",
    "Parque Anchieta",
    "Parque Colúmbia",
    "Pavuna",
    "Penha",
    "Penha Circular",
    "Quintino Bocaiúva",
    "Ricardo de Albuquerque",
    "Rocha Miranda",
    "Tomás Coelho",
    "Turiaçu",
    "Vaz Lobo",
    "Vicente de Carvalho",
    "Vigário Geral",
    "Vila da Penha",
    "Vila Kosmos",
    "Vista Alegre",
]

zona_oeste = [
    "Anil",
    "Barra da Tijuca",
    "Camorim",
    "Cidade de Deus",
    "Curicica",
    "Freguesia (Jacarepaguá)",
    "Gardênia Azul",
    "Grumari",
    "Itanhangá",
    "Jacarepaguá",
    "Joá",
    "Praça Seca",
    "Pechincha",
    "Rio das Pedras",
    "Recreio dos Bandeirantes",
    "Tanque",
    "Taquara",
    "Vargem Grande",
    "Vargem Pequena",
    "Vila Valqueire",
    "Jardim Sulacap",
    "Bangu",
    "Campo dos Afonsos",
    "Deodoro",
    "Gericinó",
    "Jabour",
    "Magalhães Bastos",
    "Padre Miguel",
    "Realengo",
    "Santíssimo",
    "Senador Camará",
    "Vila Kennedy",
    "Vila Militar",
    "Barra de Guaratiba",
    "Campo Grande",
    "Cosmos",
    "Guaratiba",
    "Inhoaíba",
    "Paciência",
    "Pedra de Guaratiba",
    "Santa Cruz",
    "Senador Vasconcelos",
    "Sepetiba",
]

centro = [
    "São Cristóvão",
    "Benfica",
    "Caju",
    "Catumbi",
    "Centro",
    "Cidade Nova",
    "Estácio",
    "Estacio",
    "Gamboa",
    "Lapa",
    "Mangueira",
    "Paquetá",
    "Rio Comprido",
    "Santa Teresa",
    "Santo Cristo",
    "Saúde",
    "Vasco da Gama",
]

to_drop = [
    "listing_url",
    "scrape_id",
    "last_scraped",
    "picture_url",
    "host_id",
    "host_url",
    "host_name",
    "host_location",
    "host_thumbnail_url",
    "host_picture_url",
    "neighbourhood",
    "bathroom_text_clean",
    "neighbourhood_cleansed",
    "host_response_rate",
    "host_acceptance_rate",
    "host_has_profile_pic",
    "has_availability",
    "name",
    "host_about",
    "description",
    "neighborhood_overview",
    "host_since",
    "host_neighbourhood",
    "host_listings_count",
    "host_total_listings_count",
    "host_verifications",
    "host_identity_verified",
    "latitude",
    "longitude",
    "property_type",
    "bathrooms_text",
    "amenities",
    "minimum_minimum_nights",
    "maximum_minimum_nights",
    "minimum_maximum_nights",
    "maximum_maximum_nights",
    "minimum_nights_avg_ntm",
    "maximum_nights_avg_ntm",
    "availability_30",
    "availability_60",
    "availability_90",
    "availability_365",
    "calendar_last_scraped",
    "number_of_reviews_ltm",
    "number_of_reviews_ltm",
    "number_of_reviews_l30d",
    "first_review",
    "last_review",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
    "review_scores_rating",
    "reviews_per_month",
    "delta_date_reviews",
    "calculated_host_listings_count",
    "calculated_host_listings_count_entire_homes",
    "calculated_host_listings_count_private_rooms",
    "calculated_host_listings_count_shared_rooms",
]

string_variables = ["name", "description", "neighborhood_overview", "host_about",]

model_variables = [
    "host_response_time",
    "host_is_superhost",
    "room_type",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "minimum_nights",
    "maximum_nights",
    "number_of_reviews",
    "instant_bookable",
    "days_since_host",
    "half_bath",
    "delta_nights",
    "mean_reviews",
    "regiao",
    "property_type_refactor",
    "is_host_rj",
    "count_name",
    "count_description",
    "count_neighborhood_overview",
    "count_host_about",
]

payload_example = [
    {
        "host_response_time": "within an hour",
        "host_is_superhost": "t",
        "room_type": "Entire home/apt",
        "accommodates": 5,
        "bathrooms": 1.0,
        "bedrooms": 2.0,
        "beds": 2.0,
        "minimum_nights": 5,
        "maximum_nights": 180,
        "number_of_reviews": 260,
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
        "count_host_about": 144,
    }
]

pycaret_numerical_features = [
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "minimum_nights",
    "maximum_nights",
    "number_of_reviews",
    "days_since_host",
    "delta_nights",
    "mean_reviews",
    "count_name",
    "count_description",
    "count_neighborhood_overview",
    "count_host_about",
]

pycaret_categorical_features = [
    "host_response_time",
    "host_is_superhost",
    "room_type",
    "instant_bookable",
    "half_bath",
    "regiao",
    "property_type_refactor",
    "is_host_rj",
]

metric_list = ["MAE", "MSE", "RMSE", "R2", "RMSLE", "MAPE",]