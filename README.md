# RNN-Transformer-DDoS-Detection-Comparison

A systematic comparison of RNN and Transformer architectures for detecting transport-layer DDoS attacks in resource-constrained IoT/5G environments.

## Overview

This project provides a unified framework for comparing different deep learning architectures in detecting transport-layer DDoS attacks, with a focus on resource-constrained environments such as IoT gateways. The framework includes implementations of various model architectures:

- Threshold Detector (rule-based baseline)
- Linear Model
- Shallow DNN
- Deep Neural Network (DNN)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)
- Transformer

Each model is evaluated along four dimensions:
- Detection accuracy (precision, recall, F1 score)
- Detection latency
- Peak memory/CPU footprint
- Early-warning lead time

## Repository Structure

```
RNN-Transformer-DDoS-Detection-Comparison/
├── align_dataset.py               # Ensures consistent data alignment
├── analyze_attack_types.py        # Analyzes model performance by attack type
├── analyze_model_interpretability.py # Extracts feature importances and attentions
├── analyze_scalability.py         # Tests models under different traffic rates
├── captures/                      # Directory for PCAP files (real and synthetic)
│   ├── normal.pcap                # Generated normal traffic
│   ├── syn_flood.pcap             # Generated SYN flood attack traffic
│   └── mixed.pcap                 # Generated mixed attack traffic
├── collect_predictions.py         # Gathers raw predictions from all models
├── data/                          # Dataset loaders and processors
│   ├── loader.py                  # Functions for loading and preparing datasets
├── evaluation_config.py           # Configuration for evaluation
├── fixed_generate_visualizations.py # Generates visualizations for paper
├── gen_synth_nsld_kdd.py          # Generates synthetic attack samples
├── generate_advanced_models_preds.py # Generates predictions for neural models
├── generate_baseline_preds.py      # Generates predictions for baseline models
├── improved_run_evaluation.py      # Enhanced evaluation pipeline
├── inference/                      # Inference package for real-time detection
│   ├── __init__.py                 # Package initialization
│   ├── inference_engine.py         # Core inference functionality
│   ├── pcap_generator.py           # Synthetic PCAP file generator
│   ├── pcap_processor.py           # PCAP to NSL-KDD feature converter
│   └── report_generator.py         # Detection report generator
├── main.py                         # Main entry point for training and inference
├── measure_performance.py          # Measures execution time and memory usage
├── models/                         # Model implementations
│   ├── dnn.py                      # Deep Neural Network
│   ├── gru.py                      # Gated Recurrent Unit
│   ├── linear_regressor.py         # Linear model
│   ├── lstm.py                     # Long Short-Term Memory
│   ├── metrics.py                  # Custom metrics for evaluation
│   ├── shallow_dnn.py              # Shallow Neural Network
│   ├── threshold_detector.py       # Rule-based threshold detector
│   └── transformer.py              # Transformer model
├── preprocess_hard.py              # Preprocesses hard attack dataset
├── process_threshold_detector.py   # Processes threshold detector model
├── reports/                        # Directory for detection reports
├── saved_models/                   # Directory for trained models
│   ├── dnn_1746776936/             # Saved DNN model
│   ├── gru_1746776936/             # Saved GRU model
│   ├── linear_model_1746776936/    # Saved linear model
│   ├── lstm_1746776936/            # Saved LSTM model
│   ├── shallow_dnn_1746776936/     # Saved shallow DNN model
│   ├── threshold_detector_1746776936/ # Saved threshold detector
│   └── transformer_1746776936/     # Saved transformer model
├── test_significance.py            # Performs statistical significance testing
├── threshold_detector_evaluation.py # Evaluates threshold detector models
└── trainer.py                      # Trains all model architectures
```

## Key Features

- **Modular Design**: Easily swap or extend model architectures
- **Comprehensive Evaluation**: Assess models on accuracy, latency, memory, and CPU usage
- **Realistic Testing**: Uses both public datasets and synthetic attacks
- **Resource-Constrained Focus**: Optimized for edge devices like Raspberry Pi
- **Reproducibility**: All data processing pipelines and training code included
- **Real-time Detection**: Process PCAP files for live DDoS attack detection

## Datasets

The framework works with the NSL-KDD dataset and includes a synthetically enhanced NSL-KDD-Hard dataset for evaluating models against stealthy attacks. The data processing pipeline handles:

- Feature extraction and normalization
- Sequence windowing for RNN/Transformer models
- Train/validation/test splitting

## Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.10+
- NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn
- Scapy (for PCAP generation and processing)
- dpkt (for PCAP processing)
- psutil (for memory measurement)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/kafetzis/RNN-Transformer-DDoS-Detection-Comparison.git
cd RNN-Transformer-DDoS-Comparison
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Download and prepare the dataset:
```bash
# The NSL-KDD dataset will be automatically downloaded
# Generate synthetic hard dataset
python gen_synth_nsld_kdd.py
```

### Training Models

Train all model architectures:
```bash
python trainer.py
```

### Running Evaluations

Run the full evaluation pipeline:
```bash
python improved_run_evaluation.py --test-file data/nsl_kdd_dataset/NSL-KDD-Hard.csv
```

Generate visualizations:
```bash
python fixed_generate_visualizations.py
```

### Using Inference Mode

The framework provides real-time DDoS detection capabilities using the trained models. It can process network traffic captures in PCAP format and generate detailed attack reports.

#### Generating Synthetic PCAP Files

Generate synthetic PCAP files with various attack patterns for testing:

```bash
# Generate a SYN flood attack PCAP
python main.py generate --output captures/syn_flood.pcap --attack-type syn_flood

# Generate normal traffic without attacks
python main.py generate --output captures/normal.pcap --attack-type mixed --attack-duration 0 --normal-ratio 1.0

# Generate mixed attack traffic
python main.py generate --output captures/mixed.pcap --attack-type mixed --duration 600 --attack-duration 300
```

#### Running Inference on PCAP Files

Process PCAP files to detect DDoS attacks:

```bash
# Run inference on a PCAP file using GRU model
python main.py infer --pcap-file captures/syn_flood.pcap --model gru

# Save detailed JSON report
python main.py infer --pcap-file captures/syn_flood.pcap --model lstm --save-json

# Use a specific model directory and detection threshold
python main.py infer --pcap-file captures/syn_flood.pcap --model-dir saved_models/gru_1746776936 --threshold 0.3
```

The inference process will:
1. Extract NSL-KDD features from the PCAP file
2. Apply the selected model for attack detection
3. Generate a detailed report showing:
   - Attack windows with timestamps
   - Confidence levels
   - Suspected attack types
   - Overall attack statistics

## Results

The framework evaluates each model along multiple dimensions:

- **F1 Score**: Recurrent architectures (GRU, LSTM) achieve the highest F1 scores (~0.73)
- **Latency**: All models exhibit 45-54 ms inference times, dominated by feature extraction costs
- **Memory Usage**: Models consume between 1650-2708 MB
- **CPU Usage**: The Transformer requires significantly more CPU (22.98%) than other models (<0.33%)

## Deployment

For edge deployment, consider:

1. The feature extraction process is the primary bottleneck (993 ms)
2. Memory constraints are universal across all architectures
3. GRU offers the best balance of accuracy and efficiency
4. Use threshold detector when extreme resource constraints exist

## Citation

If you use this code in your research, please cite:

```
@article{kafetzis2025real,
  title={Real-Time, Resource-Efficient Detection of Transport-Layer DDoS Attacks in IoT/5G Networks: A Comparative Study of RNNs and Transformers},
  author={Kafetzis, Dimitrios},
  journal={},
  year={2025},
  url={https://github.com/kafetzis/RNN-Transformer-DDoS-Detection-Comparison}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Dimitrios Kafetzis - kafetzis@aueb.gr

Project Link: [https://github.com/kafetzis/RNN-Transformer-DDoS-Detection-Comparison](https://github.com/kafetzis/RNN-Transformer-DDoS-Detection-Comparison)