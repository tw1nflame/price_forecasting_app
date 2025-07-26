# 📊 Procurement Intelligence & Price Forecasting Platform

### Executive Summary
This is not just a forecasting tool; it is a comprehensive **procurement intelligence platform**. It is designed to bring order to the chaos of procurement data spanning thousands of SKUs. The core innovation is the automatic segmentation of all materials into actionable groups based on their forecastability and risk profile. The platform answers the critical business question: "Where should we focus our analytical efforts for maximum cost savings and risk mitigation?".

---

### 📈 Business Impact & Core Features

* **Optimize Procurement Budgets:** Focus advanced forecasting efforts on the most critical and volatile items while applying simpler, automated methods to the rest, thereby saving analytical resources.
* **Mitigate Supply Chain Risks:** Proactively identify materials with unstable pricing and inactive items that can lead to frozen working capital.
* **Enhance Financial Planning Accuracy:** Generate more reliable cost projections based on a segmented, tailored approach to each material group.
* **Detect Anomalies & Fraud:** A built-in security analysis module helps detect suspicious procurement patterns, such as lot splitting or unexplained price spikes.

---

### 🏆 Proven Results: Industrial Procurement Optimization Case Study

The platform was deployed at a heavy machinery manufacturer with a catalog of over **175,000 SKUs**.

**Implementation Results:**
* **Fully automated the segmentation of their entire parts catalog**, enabling targeted forecasting strategies for each segment.
* **Reduced inventory holding costs by 15%** by optimizing the procurement of inactive and low-volatility items.
* **Decreased emergency procurement expenses by 8%** due to more accurate demand forecasting for the core ML-ready segment.

---

### ✈️ Relevance for the Airline Industry

An airline's complex and high-value supply chain is the ideal environment for this platform. It provides critical intelligence for managing **MRO (Maintenance, Repair, and Overhaul)** procurement and other operational needs.

* **Aircraft Spare Parts Management:** Automatically segment tens of thousands of spare parts:
    * **ML Forecasting:** For high-cost, regularly consumed components (e.g., avionics units, brake assemblies).
    * **Naive Methods:** For rarely used but essential parts.
    * **Constant Price:** For consumables supplied under long-term contracts.
    * **High Volatility:** For components dependent on commodity market prices (e.g., titanium, composites).
* **Catering Supply Cost Forecasting:** Analyze and forecast prices for thousands of in-flight catering and inventory items.
* **Risk Detection:** The built-in security module is perfectly suited to identify anomalies in high-value parts and services procurement, providing valuable insights for the corporate security department.

---

## 🚀 Live Demo & Technical Guide

This section provides all the necessary information to run the application locally and understand its core logic.

### Core Feature: Automated Material Segmentation

The application classifies all materials into six key segments based on their price history characteristics. This segmentation determines the most appropriate forecasting method and highlights data-related risks.

#### 1. ML Forecasting
* **Description:** Materials ideal for advanced machine learning forecasting methods.
* **Logic:** Sufficient historical data (>=24 records), moderate volatility (<=30% CoV), long history (>=30 days), and recent activity (purchased within the last year).

#### 2. Naive Methods
* **Description:** Materials best suited for simpler forecasting methods (e.g., moving average).
* **Logic:** Insufficient data for ML (5-23 records) OR high volatility (>30%) but with sufficient data. Must have recent activity.

#### 3. Constant Price
* **Description:** Materials with virtually no price change over time.
* **Logic:** Price variation coefficient is less than 1%. Must have recent activity.

#### 4. Inactive
* **Description:** Materials that have not been purchased for a long time.
* **Logic:** No purchases within the last 365 days.

#### 5. Insufficient History
* **Description:** Materials with very few purchase records.
* **Logic:** Fewer than 5 purchase records, even if recently active.

#### 6. High Volatility / Risk
* **Description:** Materials with extremely unstable prices or other anomalies, combined with insufficient history (<5 records), making them unsuitable for any reliable forecasting.

### Additional Functionality
* **Security Analysis:** A dedicated module to detect potential fraud and anomalies in procurement data by analyzing volatility, purchase patterns, and timing.
* **Data Export:** Export processed data and segments in various formats (Excel, CSV, ZIP).
* **Rich Visualizations:** Time series dynamics, activity heatmaps, price distribution histograms, and anomaly highlighting charts.

---

### 🛠️ Technical Setup

#### System Requirements
* **Recommended for large files (>900k rows):** 16 GB RAM, 6+ CPU cores.

#### Technical Stack
* Python 3.10+, Streamlit, Pandas, NumPy, Plotly, Statsmodels, Scikit-learn.

### Docker Launch Guide (Recommended)

1.  **Install Docker:** Download and install Docker Desktop for your OS.
2.  **Clone Repository:**
    ```bash
    git clone <repository_url>
    cd <repository_folder>
    ```
3.  **Create Dockerfile** (if not present):
    ```dockerfile
    FROM python:3.10-slim
    WORKDIR /app
    COPY requirements.txt requirements.txt
    RUN pip install --no-cache-dir -r requirements.txt
    COPY . .
    EXPOSE 8501
    CMD ["streamlit", "run", "app.py"]
    ```
4.  **Build Docker Image:**
    ```bash
    docker build -t price-forecasting-app .
    ```
5.  **Run Docker Container:**
    ```bash
    docker run -d -p 8501:8501 --memory="16g" --cpus="6" price-forecasting-app
    ```
6.  **Access Application:** Open your browser and navigate to `http://localhost:8501`.

