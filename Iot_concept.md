# Task 2 — AI-Driven IoT Concept: Smart Agriculture Simulation System

## 1. Required IoT Sensors
To build an effective AI model for predicting crop yield, collect comprehensive data on environment, soil, and plant health.

- Soil Moisture Sensor  
  - Purpose: Measures volumetric water content.  
  - Why: Critical for irrigation management; prevents under- and over-watering.

- Soil Temperature Sensor  
  - Purpose: Measures soil temperature at various depths.  
  - Why: Influences germination and root development; helps optimize planting times.

- Air Temperature & Humidity Sensor (e.g., DHT22)  
  - Purpose: Measures ambient air temperature and relative humidity.  
  - Why: High humidity promotes fungal disease; high temperature causes heat stress.

- Soil pH Sensor  
  - Purpose: Measures soil acidity/alkalinity.  
  - Why: pH affects nutrient availability; informs soil amendments.

- NPK Sensor (Nutrient Sensor)  
  - Purpose: Measures Nitrogen (N), Phosphorus (P), Potassium (K) concentrations.  
  - Why: Enables precision fertilization, reducing cost and runoff.

- PAR (Photosynthetically Active Radiation) Sensor  
  - Purpose: Measures light wavelengths used for photosynthesis.  
  - Why: Correlates with plant growth potential better than ambient light.

- Weather Station (Anemometer, Rain Gauge)  
  - Purpose: Measures wind speed and precipitation.  
  - Why: Wind affects evaporation and can damage crops; rain gauge calibrates irrigation.

## 2. Proposed AI Model for Crop Yield Prediction
The collected data is a multivariate time series. A Long Short-Term Memory (LSTM) network is recommended.

- Model: Long Short-Term Memory (LSTM)
- Rationale: LSTMs capture long-range temporal dependencies—important because yield depends on cumulative conditions across the growing season.
- Task: Regression — predict yield (e.g., tons/ha or bushels/acre).

Model inputs (features)
- Time-series:
  - soil_moisture (daily average)
  - soil_temp (daily min/max)
  - air_temp (daily min/max)
  - humidity (daily average)
  - soil_ph (weekly)
  - soil_npk (weekly)
  - par_light (daily total)
  - rainfall (daily total)
- Static:
  - crop_type (corn, wheat, soy, etc.)
  - soil_type (sandy, clay, loam)
  - planting_date

Model output
- predicted_yield (numeric regression)

## 3. Data Flow Diagram (Sketch)
(Field) -> (Cloud) -> (End user)

1. IoT Sensors (soil, temp, humidity, etc.)
   - Wireless: LoRaWAN, 5G, or Wi‑Fi
2. IoT Gateway / Edge Device
   - Aggregates sensor data
   - Performs preprocessing (averaging, filtering)
3. Cloud IoT Hub / Message Broker
   - Receives and routes sensor data securely and at scale
4. Raw Data Lake / Time-Series Database
   - Stores historical data for training and analysis
5. Data Preprocessing & Feature Engineering
   - Cleans missing values, normalizes, aggregates daily/weekly features
6. AI Model (LSTM) — Prediction Service
   - Runs inference on processed data to generate yield predictions
7. User Dashboard / Mobile App
   - Displays current readings, trends, and predicted yield for the farmer
