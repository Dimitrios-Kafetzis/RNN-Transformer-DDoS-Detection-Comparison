#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: inference/pcap_processor.py
Author: Dimitrios Kafetzis (kafetzis@aueb.gr)
Created: May 2025
License: MIT

Description:
    Simple PCAP file processor that converts captured network traffic
    to the NSL-KDD feature format for DDoS detection. Extracts basic 
    flow statistics and connection information from packets.

Usage:
    This module is imported by other scripts and not meant to be run directly.
    
    Import example:
    from inference.pcap_processor import process_pcap
    
    Usage example:
    features, timestamps = process_pcap("traffic.pcap")
"""

import os
import numpy as np
import pandas as pd
import dpkt
import socket
import logging
from collections import defaultdict
from typing import Tuple, List, Dict, Any

logger = logging.getLogger(__name__)

# Features present in NSL-KDD
NSL_KDD_FEATURES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate'
]

def ip_to_str(ip_address: bytes) -> str:
    """Convert IP address bytes to string."""
    try:
        return socket.inet_ntop(socket.AF_INET, ip_address)
    except ValueError:
        # Try IPv6
        try:
            return socket.inet_ntop(socket.AF_INET6, ip_address)
        except ValueError:
            return "Unknown"

def protocol_to_str(protocol: int) -> str:
    """Convert protocol number to string."""
    protocols = {
        1: "icmp",
        6: "tcp",
        17: "udp"
    }
    return protocols.get(protocol, "other")

def process_pcap(pcap_file: str, window_size: int = 10) -> Tuple[np.ndarray, List[float]]:
    """
    Process PCAP file and extract NSL-KDD features.
    
    Args:
        pcap_file: Path to the PCAP file
        window_size: Number of packets to group into a single flow
        
    Returns:
        Tuple of (features, timestamps) where features is a numpy array of shape
        (n_windows, n_features) and timestamps is a list of window end times
    """
    if not os.path.exists(pcap_file):
        raise FileNotFoundError(f"PCAP file not found: {pcap_file}")
    
    logger.info(f"Processing PCAP file: {pcap_file}")
    
    # Flow tracking data structures
    flows = defaultdict(list)
    flow_stats = defaultdict(lambda: {
        'duration': 0,
        'protocol_type': '',
        'src_bytes': 0,
        'dst_bytes': 0,
        'wrong_fragment': 0,
        'urgent': 0,
        'count': 0,
        'srv_count': 0,
        'timestamp': 0
    })
    
    # Read PCAP file
    try:
        with open(pcap_file, 'rb') as f:
            pcap = dpkt.pcap.Reader(f)
            
            for timestamp, buf in pcap:
                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                    
                    # Skip non-IP packets
                    if not isinstance(eth.data, dpkt.ip.IP):
                        continue
                    
                    ip = eth.data
                    protocol = ip.p
                    protocol_str = protocol_to_str(protocol)
                    
                    src_ip = ip_to_str(ip.src)
                    dst_ip = ip_to_str(ip.dst)
                    
                    # Extract port information if TCP or UDP
                    src_port = 0
                    dst_port = 0
                    tcp_flags = 0
                    
                    if protocol == 6 and isinstance(ip.data, dpkt.tcp.TCP):  # TCP
                        tcp = ip.data
                        src_port = tcp.sport
                        dst_port = tcp.dport
                        tcp_flags = tcp.flags
                    elif protocol == 17 and isinstance(ip.data, dpkt.udp.UDP):  # UDP
                        udp = ip.data
                        src_port = udp.sport
                        dst_port = udp.dport
                    
                    # Create flow identifier (5-tuple)
                    flow_id = (src_ip, src_port, dst_ip, dst_port, protocol)
                    
                    # Add packet to flow
                    flows[flow_id].append({
                        'timestamp': timestamp,
                        'size': len(buf),
                        'protocol': protocol_str,
                        'tcp_flags': tcp_flags,
                        'is_src_to_dst': True  # Direction flag
                    })
                    
                    # Update flow statistics
                    flow_stats[flow_id]['protocol_type'] = protocol_str
                    flow_stats[flow_id]['src_bytes'] += len(buf)
                    flow_stats[flow_id]['count'] += 1
                    flow_stats[flow_id]['timestamp'] = timestamp  # Last seen timestamp
                    
                    # Check for fragmentation (simplified)
                    if ip.off & dpkt.ip.IP_MF or ip.off & dpkt.ip.IP_OFFMASK:
                        flow_stats[flow_id]['wrong_fragment'] += 1
                    
                    # TCP urgent pointer
                    if protocol == 6 and isinstance(ip.data, dpkt.tcp.TCP):
                        if ip.data.urp:
                            flow_stats[flow_id]['urgent'] += 1
                
                except Exception as e:
                    logger.warning(f"Error processing packet: {e}")
                    continue
    
    except Exception as e:
        logger.error(f"Error reading PCAP file: {e}")
        raise
    
    # Process flows into windows
    all_features = []
    timestamps = []
    
    # Sort flows by first packet timestamp
    sorted_flows = []
    for flow_id, packets in flows.items():
        if packets:
            first_ts = min(p['timestamp'] for p in packets)
            sorted_flows.append((first_ts, flow_id, packets))
    
    sorted_flows.sort()  # Sort by timestamp
    
    # Create windows of flows
    windows = []
    current_window = []
    
    for _, flow_id, packets in sorted_flows:
        current_window.append((flow_id, packets))
        
        if len(current_window) >= window_size:
            windows.append(current_window)
            current_window = []
    
    # Add the last partial window if it exists
    if current_window:
        windows.append(current_window)
    
    # Create feature vectors for each window
    for window in windows:
        window_features = extract_window_features(window, flow_stats)
        
        # Get timestamp for this window (latest packet in window)
        latest_ts = max(max(p['timestamp'] for p in packets) 
                         for _, packets in window)
        
        all_features.append(window_features)
        timestamps.append(latest_ts)
    
    # Convert to numpy array
    if all_features:
        features_array = np.vstack(all_features)
    else:
        # Create empty array with the right number of features
        features_array = np.zeros((0, len(NSL_KDD_FEATURES)))
    
    logger.info(f"Processed {len(windows)} windows from PCAP file")
    
    return features_array, timestamps

def extract_window_features(window, flow_stats):
    """
    Extract NSL-KDD features from a window of flows.
    
    Args:
        window: List of (flow_id, packets) tuples
        flow_stats: Dictionary of flow statistics
        
    Returns:
        Numpy array of feature values
    """
    # Initialize the feature vector with default values
    features = {
        'duration': 0,
        'protocol_type': 0,  # Will be one-hot encoded
        'service': 0,  # Will be one-hot encoded
        'flag': 0,  # Will be one-hot encoded
        'src_bytes': 0,
        'dst_bytes': 0,
        'land': 0,
        'wrong_fragment': 0,
        'urgent': 0,
        'hot': 0,
        'num_failed_logins': 0,
        'logged_in': 0,
        'num_compromised': 0,
        'root_shell': 0,
        'su_attempted': 0,
        'num_root': 0,
        'num_file_creations': 0,
        'num_shells': 0,
        'num_access_files': 0,
        'num_outbound_cmds': 0,
        'is_host_login': 0,
        'is_guest_login': 0,
        'count': 0,
        'srv_count': 0,
        'serror_rate': 0,
        'srv_serror_rate': 0,
        'rerror_rate': 0,
        'srv_rerror_rate': 0,
        'same_srv_rate': 0,
        'diff_srv_rate': 0,
        'srv_diff_host_rate': 0,
        'dst_host_count': 0,
        'dst_host_srv_count': 0,
        'dst_host_same_srv_rate': 0,
        'dst_host_diff_srv_rate': 0,
        'dst_host_same_src_port_rate': 0,
        'dst_host_srv_diff_host_rate': 0,
        'dst_host_serror_rate': 0,
        'dst_host_srv_serror_rate': 0,
        'dst_host_rerror_rate': 0,
        'dst_host_srv_rerror_rate': 0
    }
    
    # Protocol type counts
    protocol_counts = {"tcp": 0, "udp": 0, "icmp": 0}
    
    # Aggregated statistics across the window
    for flow_id, packets in window:
        stats = flow_stats[flow_id]
        protocol = stats['protocol_type']
        
        # Update protocol counts
        if protocol in protocol_counts:
            protocol_counts[protocol] += 1
        
        # Update basic counts
        features['src_bytes'] += stats['src_bytes']
        features['dst_bytes'] += stats.get('dst_bytes', 0)
        features['wrong_fragment'] += stats['wrong_fragment']
        features['urgent'] += stats['urgent']
        features['count'] += stats['count']
    
    # Set the most common protocol - convert to numeric representation
    most_common_protocol = max(protocol_counts.items(), key=lambda x: x[1])[0]
    
    # Convert protocol type to a numeric value
    protocol_mapping = {"tcp": 1, "udp": 2, "icmp": 3, "other": 0}
    features['protocol_type'] = protocol_mapping.get(most_common_protocol, 0)
    
    # Calculate simple attack-indicative metrics
    # These are simplified approximations of NSL-KDD features
    if features['count'] > 0:
        features['serror_rate'] = features['wrong_fragment'] / features['count']
        features['srv_serror_rate'] = features['urgent'] / features['count']
    
    # Set some reasonable values for the rest of the features
    # In a real implementation, these would be calculated based on historical data
    features['same_srv_rate'] = 0.5  # Assumes half of connections use same service
    features['diff_srv_rate'] = 0.5  # Assumes half of connections use different services
    
    # Convert the dictionary to a list in the correct order
    feature_list = [features[name] for name in NSL_KDD_FEATURES]
    
    return np.array(feature_list, dtype=np.float32)

def preprocess_features(features: np.ndarray, mean: np.ndarray = None, std: np.ndarray = None) -> np.ndarray:
    """
    Apply normalization to features.
    
    Args:
        features: Feature array
        mean: Mean values for normalization (if None, calculated from features)
        std: Standard deviation values for normalization (if None, calculated from features)
        
    Returns:
        Normalized features
    """
    # Handle empty feature array
    if features.shape[0] == 0:
        return features
    
    # Calculate mean and std if not provided
    if mean is None or std is None:
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        # Avoid division by zero
        std[std < 1e-10] = 1.0
    
    # Apply normalization
    normalized = (features - mean) / std
    
    # Handle NaN and Inf values
    normalized = np.nan_to_num(normalized)
    
    return normalized

def pcap_to_nslkdd_features(pcap_file: str, model_dir: str = None) -> Tuple[np.ndarray, List[float]]:
    """
    Convert PCAP file to NSL-KDD features ready for model inference.
    
    Args:
        pcap_file: Path to the PCAP file
        model_dir: Directory containing model normalization parameters
        
    Returns:
        Tuple of (normalized_features, timestamps)
    """
    # Extract raw features from PCAP
    features, timestamps = process_pcap(pcap_file)
    
    if features.shape[0] == 0:
        logger.warning(f"No features extracted from {pcap_file}")
        return features, timestamps
    
    # Apply normalization
    try:
        if model_dir and os.path.exists(model_dir):
            # Try to load normalization parameters from model directory
            mean_path = os.path.join(model_dir, "X_mean.npy")
            std_path = os.path.join(model_dir, "X_std.npy")
            
            if os.path.exists(mean_path) and os.path.exists(std_path):
                mean = np.load(mean_path)
                std = np.load(std_path)
                normalized = preprocess_features(features, mean, std)
                logger.info(f"Applied normalization from {model_dir}")
                return normalized, timestamps
        
        # Fall back to normalizing based on the features themselves
        normalized = preprocess_features(features)
        logger.info("Applied self-normalization to features")
        return normalized, timestamps
        
    except Exception as e:
        logger.error(f"Error applying normalization: {e}")
        # Return unnormalized features as fallback
        return features, timestamps