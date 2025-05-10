"""
Inference package for DDoS detection.

This package provides functionality for running inference on PCAP files,
generating detection reports, generating synthetic PCAP files for testing,
and visualizing results.
"""

from inference.pcap_processor import pcap_to_nslkdd_features
from inference.inference_engine import run_inference
from inference.report_generator import generate_text_report, save_json_report, print_summary
from inference.pcap_generator import PcapGenerator

__all__ = [
    'pcap_to_nslkdd_features',
    'run_inference',
    'generate_text_report', 
    'save_json_report',
    'print_summary',
    'PcapGenerator'
]