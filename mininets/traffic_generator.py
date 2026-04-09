import time
import logging
from typing import List, Tuple
from mininet.net import Mininet

logger = logging.getLogger(__name__)

class TrafficGenerator:
    def __init__(self, net: Mininet):
        self.net = net
        self.active_procs = []

    def add_iperf_flow(self, src_host: str, dst_host: str, duration: int = 10, 
                      bandwidth: str = '1M', protocol: str = 'TCP') -> None:
        src = self.net.get(src_host)
        dst = self.net.get(dst_host)
        
        if not src or not dst:
            logger.error(f"Host {src_host} or {dst_host} not found")
            return

        proto_flag = '-u' if protocol.upper() == 'UDP' else ''
        bw_flag = f'-b {bandwidth}' if protocol.upper() == 'UDP' else ''

        # Start server
        server_cmd = f'iperf -s {proto_flag} -p 5001'
        dst.popen(server_cmd)

        time.sleep(1)

        # Start client
        client_cmd = f'iperf -c {dst.IP()} {bw_flag} -t {duration} -p 5001'
        proc = src.popen(client_cmd)
        self.active_procs.append(proc)

        logger.info(f"Started iperf {protocol} {bandwidth} from {src_host} → {dst_host} ({duration}s)")

    def add_ping_flow(self, src_host: str, dst_host: str, count: int = 10, interval: float = 0.2) -> None:
        src = self.net.get(src_host)
        dst = self.net.get(dst_host)
        if not src or not dst:
            return

        cmd = f'ping -c {count} -i {interval} {dst.IP()}'
        proc = src.popen(cmd)
        self.active_procs.append(proc)
        logger.info(f"Started ping from {src_host} → {dst_host} ({count} packets)")

    def start_burst_traffic(self, src_hosts: List[str], dst_hosts: List[str], intensity: str = 'high'):
        """Dùng để tạo burst traffic overload controller"""
        bw = '10M' if intensity == 'high' else '5M'
        for src in src_hosts:
            for dst in dst_hosts:
                self.add_iperf_flow(src, dst, duration=30, bandwidth=bw, protocol='UDP')
                time.sleep(0.5)  # stagger để tránh simultaneous start

    def wait_for_completion(self, timeout: int = 180):
        logger.info("Waiting for traffic flows to complete...")
        start = time.time()
        while time.time() - start < timeout:
            time.sleep(5)
            # Check if processes are still alive (simple way)
            if all(p.poll() is not None for p in self.active_procs if p.poll() is not None):
                break
        logger.info("Traffic generation completed or timeout reached")


def generate_scenario(net: Mininet, scenario: str = 'basic'):
    tg = TrafficGenerator(net)
    
    hosts = [h.name for h in net.hosts]
    
    if scenario == 'basic':
        tg.add_iperf_flow('h1', 'h2', duration=15, bandwidth='2M')
        tg.add_ping_flow('h1', 'h2', count=8)
    
    elif scenario == 'burst':
        tg.start_burst_traffic(hosts[:3], hosts[3:], intensity='high')
    
    elif scenario == 'mixed':
        tg.add_iperf_flow('h1', 'h4', duration=40, bandwidth='8M', protocol='UDP')
        tg.add_iperf_flow('h2', 'h5', duration=30, bandwidth='5M', protocol='TCP')
        tg.add_ping_flow('h3', 'h6', count=15)
    
    elif scenario == 'poisson':  # Sẽ mở rộng sau với random sleep
        for _ in range(8):
            tg.add_iperf_flow(hosts[0], hosts[-1], duration=8, bandwidth='3M')
            time.sleep(2)
    
    return tg


if __name__ == '__main__':
    # Test standalone
    from custom_topo import create_network
    net = create_network(topo_type='tree', depth=2, fanout=2, enable_cli=False)
    tg = generate_scenario(net, scenario='burst')
    tg.wait_for_completion(timeout=60)
    net.stop()