#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: inference/report_generator.py
Author: Dimitrios Kafetzis (kafetzis@aueb.gr)
Created: May 2025
License: MIT

Description:
    Generates text-based reports from DDoS detection results.
    Creates human-readable output summarizing attack detection,
    timing, confidence, and provides interpretable insights.

Usage:
    This module is imported by other scripts and not meant to be run directly.
    
    Import example:
    from inference.report_generator import generate_text_report
    
    Usage example:
    report_text = generate_text_report(inference_results, output_file="report.txt")
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

def generate_text_report(results: Dict[str, Any], output_file: Optional[str] = None) -> str:
    """
    Generate a text-based report from inference results.
    
    Args:
        results: Dictionary containing inference results
        output_file: Path to output file (if None, only returns text)
        
    Returns:
        Text report as a string
    """
    # Extract key data from results
    model_type = results.get("model_type", "unknown")
    threshold = results.get("threshold", 0.5)
    report = results.get("report", {})
    
    is_attack_detected = report.get("is_attack_detected", False)
    attack_count = report.get("attack_count", 0)
    total_count = report.get("total_count", 0)
    attack_ratio = report.get("attack_ratio", 0)
    attack_windows = report.get("attack_windows", [])
    max_confidence = report.get("max_confidence", 0)
    mean_confidence = report.get("mean_confidence", 0)
    suspected_attack_types = report.get("suspected_attack_types", [])
    
    # Build the report
    lines = []
    lines.append("=" * 80)
    lines.append("DDoS ATTACK DETECTION REPORT".center(80))
    lines.append("=" * 80)
    
    # Add timestamp
    lines.append(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Add model information
    lines.append("MODEL INFORMATION")
    lines.append("-" * 80)
    lines.append(f"Model type: {model_type}")
    lines.append(f"Detection threshold: {threshold}")
    lines.append("")
    
    # Add summary
    lines.append("DETECTION SUMMARY")
    lines.append("-" * 80)
    
    if is_attack_detected:
        lines.append("🚨 ATTACK DETECTED 🚨")
    else:
        lines.append("✓ No attacks detected")
    
    lines.append(f"Windows analyzed: {total_count}")
    lines.append(f"Attack windows: {attack_count} ({attack_ratio*100:.2f}%)")
    
    if attack_count > 0:
        lines.append(f"Maximum detection confidence: {max_confidence:.4f}")
        lines.append(f"Mean detection confidence: {mean_confidence:.4f}")
        
        # Add suspected attack types
        lines.append("")
        lines.append("SUSPECTED ATTACK TYPES")
        lines.append("-" * 80)
        
        for attack_type in suspected_attack_types:
            lines.append(f"- {attack_type}")
        
        # Add attack window details
        lines.append("")
        lines.append("ATTACK DETAILS")
        lines.append("-" * 80)
        
        for i, window in enumerate(attack_windows):
            lines.append(f"Attack Window #{i+1}:")
            lines.append(f"  Time range: {window['start_time']} to {window['end_time']}")
            lines.append(f"  Duration: {window['duration_seconds']:.2f} seconds")
            lines.append(f"  Window indices: {window['start_idx']} to {window['end_idx']}")
            lines.append(f"  Confidence: {window['avg_confidence']:.4f} (avg), {window['max_confidence']:.4f} (max)")
            lines.append("")
    
    # Add footer
    lines.append("=" * 80)
    lines.append("END OF REPORT".center(80))
    lines.append("=" * 80)
    
    # Join lines into a single string
    report_text = "\n".join(lines)
    
    # Write to file if requested
    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        try:
            with open(output_file, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving report to {output_file}: {e}")
    
    return report_text

def save_json_report(results: Dict[str, Any], output_file: str) -> bool:
    """
    Save inference results as a JSON file.
    
    Args:
        results: Dictionary containing inference results
        output_file: Path to output file
        
    Returns:
        True if successful, False otherwise
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"JSON report saved to {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving JSON report to {output_file}: {e}")
        return False

def print_summary(results: Dict[str, Any]) -> None:
    """
    Print a brief summary of inference results to console.
    
    Args:
        results: Dictionary containing inference results
    """
    report = results.get("report", {})
    is_attack_detected = report.get("is_attack_detected", False)
    attack_count = report.get("attack_count", 0)
    total_count = report.get("total_count", 0)
    attack_ratio = report.get("attack_ratio", 0)
    suspected_types = report.get("suspected_attack_types", [])
    
    print("\n" + "=" * 80)
    print("DETECTION SUMMARY".center(80))
    print("=" * 80)
    
    if is_attack_detected:
        print("🚨 ATTACK DETECTED 🚨".center(80))
    else:
        print("✓ No attacks detected".center(80))
    
    print(f"Windows analyzed: {total_count}")
    print(f"Attack windows: {attack_count} ({attack_ratio*100:.2f}%)")
    
    if is_attack_detected and suspected_types:
        print("\nSuspected attack types:")
        for attack_type in suspected_types:
            print(f"- {attack_type}")
    
    print("=" * 80)