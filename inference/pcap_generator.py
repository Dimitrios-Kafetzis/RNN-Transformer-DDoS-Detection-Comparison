#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: inference/pcap_generator.py
Author: Dimitrios Kafetzis (kafetzis@aueb.gr)
Created: May 2025
License: MIT

Description:
    Generates synthetic PCAP files with controlled attack patterns for testing.
    Creates realistic network traffic with embedded DDoS attacks including:
    - TCP SYN Floods
    - UDP Floods
    - HTTP Floods
    - ICMP Floods
    - Low-and-Slow attacks
    
    These generated files can be used to test the inference pipeline
    without requiring real network captures.

Usage:
    $ python -m inference.pcap_generator [options]
    
    Examples:
    $ python -m inference.pcap_generator --output captures/test.pcap
    $ python -m inference.pcap_generator --output captures/syn_flood.pcap --attack-type syn_flood --attack-duration 30
    $ python -m inference.pcap_generator --output captures/mixed.pcap --attack-type mixed --normal-ratio 0.7
"""

import os
import random
import time
import argparse
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta
import socket
import struct
import ipaddress

from scapy.all import (
    Ether, IP, TCP, UDP, ICMP, 
    RandIP, RandMAC, RandShort,
    wrpcap, Raw
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Attack types
ATTACK_TYPES = [
    "syn_flood",
    "udp_flood",
    "http_flood",
    "icmp_flood",
    "low_and_slow",
    "mixed"
]

class PcapGenerator:
    """Generator for synthetic PCAP files with controlled attack patterns."""
    
    def __init__(
        self,
        output_file: str,
        duration: int = 300,
        attack_type: str = "mixed",
        attack_start: int = 60,
        attack_duration: int = 120,
        normal_ratio: float = 0.6,
        packet_rate: int = 100,
        victim_ip: str = "192.168.1.100",
        random_seed: Optional[int] = None
    ):
        """
        Initialize PCAP generator.
        
        Args:
            output_file: Path to output PCAP file
            duration: Total duration of the capture in seconds
            attack_type: Type of attack to generate
            attack_start: When the attack starts (seconds from beginning)
            attack_duration: Duration of the attack in seconds
            normal_ratio: Ratio of normal traffic during non-attack periods
            packet_rate: Average packets per second
            victim_ip: Target IP address for attacks
            random_seed: Seed for random number generator
        """
        self.output_file = output_file
        self.duration = duration
        self.attack_type = attack_type.lower()
        self.attack_start = attack_start
        self.attack_duration = attack_duration
        self.normal_ratio = normal_ratio
        self.packet_rate = packet_rate
        self.victim_ip = victim_ip
        
        # Initialize random seed
        if random_seed is not None:
            random.seed(random_seed)
        
        # Validate parameters
        self._validate_parameters()
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    def _validate_parameters(self):
        """Validate input parameters."""
        if self.attack_type not in ATTACK_TYPES:
            raise ValueError(f"Invalid attack type: {self.attack_type}. Must be one of {ATTACK_TYPES}")
        
        if self.attack_start + self.attack_duration > self.duration:
            logger.warning("Attack extends beyond capture duration. Adjusting attack duration.")
            self.attack_duration = self.duration - self.attack_start
        
        if self.normal_ratio < 0 or self.normal_ratio > 1:
            raise ValueError(f"Normal ratio must be between 0 and 1, got {self.normal_ratio}")
        
        try:
            ipaddress.ip_address(self.victim_ip)
        except ValueError:
            raise ValueError(f"Invalid victim IP address: {self.victim_ip}")
    
    def generate(self):
        """Generate the PCAP file."""
        logger.info(f"Generating PCAP file: {self.output_file}")
        logger.info(f"Duration: {self.duration} seconds, Attack type: {self.attack_type}")
        logger.info(f"Attack starts at {self.attack_start}s and lasts for {self.attack_duration}s")
        
        # Generate packets
        packets = []
        
        # Calculate number of packets
        total_packets = self.duration * self.packet_rate
        logger.info(f"Generating approximately {total_packets} packets")
        
        # Generate traffic for each second
        current_time = time.time()  # Base timestamp
        
        for second in range(self.duration):
            # Determine if we're in the attack period
            is_attack_period = (second >= self.attack_start and 
                               second < self.attack_start + self.attack_duration)
            
            # Determine number of packets for this second (add some randomness)
            second_packets = int(self.packet_rate * (0.8 + 0.4 * random.random()))
            
            # Generate packets for this second
            for _ in range(second_packets):
                timestamp = current_time + second + random.random()  # Add sub-second randomness
                
                if is_attack_period:
                    # During attack period, determine if this should be an attack packet
                    is_attack_packet = random.random() > self.normal_ratio
                    
                    if is_attack_packet:
                        # Generate attack packet
                        packet = self._create_attack_packet(timestamp)
                    else:
                        # Generate normal packet
                        packet = self._create_normal_packet(timestamp)
                else:
                    # Outside attack period, only generate normal packets
                    packet = self._create_normal_packet(timestamp)
                
                packets.append(packet)
        
        # Write packets to file
        wrpcap(self.output_file, packets)
        
        logger.info(f"Generated {len(packets)} packets")
        logger.info(f"PCAP file saved to {self.output_file}")
    
    def _create_normal_packet(self, timestamp):
        """Create a normal traffic packet."""
        # Randomly choose protocol
        protocol = random.choice(["tcp", "udp", "icmp"])
        
        # Create Ethernet and IP layers
        eth = Ether(src=RandMAC(), dst=RandMAC())
        
        # Use realistic IPs (avoid using the victim IP for normal traffic)
        if random.random() < 0.8:  # 80% private IPs
            src_ip = self._generate_private_ip(avoid=self.victim_ip)
            dst_ip = self._generate_private_ip(avoid=self.victim_ip)
        else:  # 20% public IPs
            src_ip = str(RandIP())
            dst_ip = str(RandIP())
        
        ip = IP(src=src_ip, dst=dst_ip)
        
        # Add protocol layer
        if protocol == "tcp":
            sport = random.randint(1024, 65535)
            dport = random.choice([80, 443, 8080, 22, 21, 25, 110, 143, 3306, 5432])
            tcp = TCP(sport=sport, dport=dport, flags=self._random_tcp_flags())
            
            # Add payload for HTTP-like traffic
            if dport in [80, 443, 8080] and random.random() < 0.7:
                payload = self._generate_http_payload()
                packet = eth/ip/tcp/Raw(load=payload)
            else:
                packet = eth/ip/tcp
                
        elif protocol == "udp":
            sport = random.randint(1024, 65535)
            dport = random.choice([53, 67, 68, 69, 123, 161, 162, 514, 1900, 5353])
            udp = UDP(sport=sport, dport=dport)
            
            # Add DNS-like payload for UDP port 53
            if dport == 53 and random.random() < 0.8:
                payload = os.urandom(random.randint(20, 60))  # Random DNS-like payload
                packet = eth/ip/udp/Raw(load=payload)
            else:
                packet = eth/ip/udp
                
        else:  # icmp
            icmp = ICMP(type=8, code=0)  # Echo request
            packet = eth/ip/icmp
        
        # Set timestamp
        packet.time = timestamp
        
        return packet
    
    def _create_attack_packet(self, timestamp):
        """Create an attack packet based on the specified attack type."""
        if self.attack_type == "mixed":
            # Choose a random attack type for each packet
            attack_type = random.choice(["syn_flood", "udp_flood", "http_flood", "icmp_flood", "low_and_slow"])
        else:
            attack_type = self.attack_type
        
        # Create Ethernet layer
        eth = Ether(src=RandMAC(), dst=RandMAC())
        
        if attack_type == "syn_flood":
            # TCP SYN flood: many SYN packets from spoofed IPs
            src_ip = str(RandIP())
            ip = IP(src=src_ip, dst=self.victim_ip)
            sport = random.randint(1024, 65535)
            dport = random.choice([80, 443, 8080, 22, 21])
            tcp = TCP(sport=sport, dport=dport, flags='S')  # SYN flag
            packet = eth/ip/tcp
            
        elif attack_type == "udp_flood":
            # UDP flood: high volume of UDP packets
            src_ip = str(RandIP())
            ip = IP(src=src_ip, dst=self.victim_ip)
            sport = random.randint(1024, 65535)
            dport = random.randint(1, 65535)  # Random destination port
            udp = UDP(sport=sport, dport=dport)
            
            # Add random payload
            payload = os.urandom(random.randint(64, 1024))
            packet = eth/ip/udp/Raw(load=payload)
            
        elif attack_type == "http_flood":
            # HTTP flood: many HTTP requests
            # Use a small set of source IPs to simulate a botnet
            src_ip = self._generate_botnet_ip()
            ip = IP(src=src_ip, dst=self.victim_ip)
            sport = random.randint(1024, 65535)
            dport = random.choice([80, 443, 8080])
            tcp = TCP(sport=sport, dport=dport, flags='PA')  # PSH-ACK flags
            
            # Generate HTTP request
            http_methods = ["GET", "POST", "HEAD", "PUT", "DELETE", "OPTIONS"]
            method = random.choice(http_methods)
            urls = ["/", "/index.html", "/login", "/api/v1/users", "/search", "/assets/js/main.js"]
            url = random.choice(urls)
            
            http_request = f"{method} {url} HTTP/1.1\r\n"
            http_request += f"Host: {self.victim_ip}\r\n"
            http_request += "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64)\r\n"
            http_request += "Accept: */*\r\n"
            http_request += "Connection: keep-alive\r\n\r\n"
            
            packet = eth/ip/tcp/Raw(load=http_request)
            
        elif attack_type == "icmp_flood":
            # ICMP flood: many ping requests
            src_ip = str(RandIP())
            ip = IP(src=src_ip, dst=self.victim_ip)
            icmp = ICMP(type=8, code=0)  # Echo request
            
            # Add payload
            payload = os.urandom(random.randint(32, 1024))
            packet = eth/ip/icmp/Raw(load=payload)
            
        elif attack_type == "low_and_slow":
            # Low and slow: few connections but kept open with minimal traffic
            # Use a consistent source IP for each connection
            src_ip = self._generate_botnet_ip()
            ip = IP(src=src_ip, dst=self.victim_ip)
            sport = random.randint(1024, 65535)
            dport = random.choice([80, 443, 8080])
            
            # Alternate between different TCP flags to keep connection alive
            if random.random() < 0.2:
                # Occasionally send SYN to establish new connections
                tcp = TCP(sport=sport, dport=dport, flags='S')
                packet = eth/ip/tcp
            else:
                # Send minimal data to keep connection alive
                tcp = TCP(sport=sport, dport=dport, flags='PA')
                
                # Generate a minimal HTTP request that keeps connection open
                http_request = "GET / HTTP/1.1\r\n"
                http_request += f"Host: {self.victim_ip}\r\n"
                http_request += "Connection: keep-alive\r\n"
                
                # Add a small amount of data in each request
                fragment_size = random.randint(1, 10)
                http_request += "X-Padding: " + "A" * fragment_size + "\r\n"
                
                packet = eth/ip/tcp/Raw(load=http_request)
        
        else:
            # Fallback to normal packet if attack type is invalid
            packet = self._create_normal_packet(timestamp)
        
        # Set timestamp
        packet.time = timestamp
        
        return packet
    
    def _random_tcp_flags(self):
        """Generate random TCP flags for normal traffic."""
        # Common flag combinations
        flags = ['S', 'SA', 'A', 'FA', 'PA', 'R', 'RA']
        weights = [0.15, 0.25, 0.35, 0.05, 0.15, 0.025, 0.025]
        return random.choices(flags, weights=weights)[0]
    
    def _generate_http_payload(self):
        """Generate a realistic HTTP payload."""
        methods = ["GET", "POST", "HEAD"]
        urls = ["/", "/index.html", "/about", "/contact", "/api/data", "/images/logo.png", "/css/style.css", "/js/script.js"]
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
        ]
        
        method = random.choice(methods)
        url = random.choice(urls)
        user_agent = random.choice(user_agents)
        
        payload = f"{method} {url} HTTP/1.1\r\n"
        payload += "Host: example.com\r\n"
        payload += f"User-Agent: {user_agent}\r\n"
        payload += "Accept: */*\r\n"
        
        if method == "POST":
            content = "param1=value1&param2=value2"
            payload += "Content-Type: application/x-www-form-urlencoded\r\n"
            payload += f"Content-Length: {len(content)}\r\n"
            payload += "\r\n"
            payload += content
        else:
            payload += "\r\n"
        
        return payload
    
    def _generate_private_ip(self, avoid=None):
        """Generate a private IP address avoiding a specific one."""
        private_ranges = [
            ("10.0.0.0", "10.255.255.255"),
            ("172.16.0.0", "172.31.255.255"),
            ("192.168.0.0", "192.168.255.255")
        ]
        
        # Select a range
        start_ip, end_ip = random.choice(private_ranges)
        
        # Convert to integers
        start_int = struct.unpack("!I", socket.inet_aton(start_ip))[0]
        end_int = struct.unpack("!I", socket.inet_aton(end_ip))[0]
        
        # Generate a random IP in the range
        while True:
            ip_int = random.randint(start_int, end_int)
            ip = socket.inet_ntoa(struct.pack("!I", ip_int))
            
            # Make sure it's not the one to avoid
            if ip != avoid:
                return ip
    
    def _generate_botnet_ip(self):
        """Generate an IP from a simulated botnet (small set of IPs)."""
        # Use a limited set of IPs to simulate a botnet
        botnets = [
            self._generate_private_ip() for _ in range(10)
        ]
        return random.choice(botnets)

def main():
    """Main function when run as a script."""
    parser = argparse.ArgumentParser(description="Generate synthetic PCAP files with controlled attack patterns")
    
    parser.add_argument("--output", required=True,
                        help="Path to output PCAP file")
    parser.add_argument("--duration", type=int, default=300,
                        help="Total duration of the capture in seconds (default: 300)")
    parser.add_argument("--attack-type", choices=ATTACK_TYPES, default="mixed",
                        help="Type of attack to generate (default: mixed)")
    parser.add_argument("--attack-start", type=int, default=60,
                        help="When the attack starts in seconds (default: 60)")
    parser.add_argument("--attack-duration", type=int, default=120,
                        help="Duration of the attack in seconds (default: 120)")
    parser.add_argument("--normal-ratio", type=float, default=0.3,
                        help="Ratio of normal traffic during attack periods (default: 0.3)")
    parser.add_argument("--packet-rate", type=int, default=100,
                        help="Average packets per second (default: 100)")
    parser.add_argument("--victim-ip", default="192.168.1.100",
                        help="Target IP address for attacks (default: 192.168.1.100)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Create generator and run
    generator = PcapGenerator(
        output_file=args.output,
        duration=args.duration,
        attack_type=args.attack_type,
        attack_start=args.attack_start,
        attack_duration=args.attack_duration,
        normal_ratio=args.normal_ratio,
        packet_rate=args.packet_rate,
        victim_ip=args.victim_ip,
        random_seed=args.seed
    )
    
    generator.generate()

if __name__ == "__main__":
    main()