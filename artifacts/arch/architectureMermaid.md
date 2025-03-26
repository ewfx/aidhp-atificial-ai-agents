flowchart LR
    %% === DATA SOURCES ===
    subgraph Data Sources
    A[Customer Profiles\n(Excel / DB)]
    B[Transaction History\n(Excel / DB)]
    C[Social Media Sentiment\n(Excel / API)]
    end

    %% === DATA PROCESSING & MODELING ===
    subgraph Data Pipeline & Modeling
    D[Data Cleaning & Integration]
    E[Feature Engineering\n(e.g., Normalization,\nOne-Hot Encoding)]
    F[K-Means Clustering\n(User Segmentation)]
    G[Sentiment Analysis\n(DistilBERT or similar)]
    H[Recommendation Engine]
    end

    %% === UI / FRONT-END ===
    subgraph Streamlit UI
    I[User Interface\n(Dashboard)]
    J[User Input:\nText / Audio / Image]
    K[ASR (Whisper)\nImage Captioning (BLIP)]
    L[Goal-based\nRecommendation Updates]
    end

    %% === FLOW ===
    A --> D
    B --> D
    C --> D
    D --> E
    E --> F
    E --> G
    F --> H
    G --> H

    H --> I
    I --> J
    J --> K
    K --> L
    L --> H
