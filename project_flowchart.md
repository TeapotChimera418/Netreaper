# Netreaper Full Project Flowchart

Use this Mermaid diagram directly in your report.

```mermaid
flowchart TD
    A[KDDTrain_with_headers.csv\nRaw NSL-KDD Data]

    A --> B[Stage A: Baseline Training\nstage_a_baseline.py]
    B --> B1[Preprocess\nlabel -> binary\nencode + impute]
    B1 --> B2[Train RF/XGB Classifier]
    B2 --> B3[Evaluate Metrics + Plots]
    B2 --> BA1[stage_a_model.pkl]
    B1 --> BA2[X_test.npy]
    B1 --> BA3[y_test.npy]
    B1 --> BA4[feature_names.pkl]

    B1 --> C[Stage B: Adversarial Simulation\nrun_stage_b_attack.py + attack_simulation.py]
    C --> C1[Generate Gaussian-noise attacks\non X_test]
    C1 --> C2[Robustness Metrics\nclean/adv accuracy, attack success rate, L2/Linf]
    C1 --> CA1[X_adversarial_samples.csv]

    A --> D[Stage C: Safety Net Training\nstage_c_anomaly.py]
    D --> D1[Filter normal traffic only]
    D1 --> D2[Train IsolationForest]
    D2 --> DA1[stage_c_isolation_forest.pkl]
    D2 --> D3[SHAP Explainability]
    D3 --> DA2[stage_c_shap_summary.png]

    CA1 --> E[Stage C Verification\nstage_c_test.py]
    DA1 --> E
    E --> E1[Anomaly Prediction\n-1 anomaly, 1 normal]

    B2 --> F[Final Integration\nfinal_pipeline.py / app.py]
    E1 --> F
    F --> F1[Decision Rule:\nAttack if classifier predicts attack or anomaly model flags anomaly]
    F1 --> FA1[final_output_with_labels.csv]

    F --> G[Streamlit Demo\napp.py]
    G --> G1[Dashboard: baseline vs attacked vs final accuracy\n+ confusion matrix + sample predictions]
```

## Notes for Report

- Stage A is supervised attack classification.
- Stage B stress-tests Stage A with adversarial perturbations.
- Stage C is an unsupervised safety net trained on normal behavior.
- Final system combines Stage A + Stage C for stronger adversarial resilience.