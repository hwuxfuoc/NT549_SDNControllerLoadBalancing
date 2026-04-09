import time
import logging
from typing import Tuple, Optional, List
from mininet.net import Mininet

logger = logging.getLogger(__name__)


class TrafficGenerator:
    def __init__(self, net: Mininet):
        self.net = net
        # BUG CŨ: self.flows lưu metadata (tuple mô tả), nhưng wait_for_completion() đọc tuple
        # để sleep — không track được process thực sự đã xong chưa.
        # FIX: lưu riêng processes và metadata.
        self.flow_metadata: List[tuple] = []
        self.active_processes: List = []

    def add_iperf_flow(
        self,
        src_host: str,
        dst_host: str,
        duration: int = 10,
        bandwidth: str = '1M',
        protocol: str = 'TCP'
    ) -> Optional[Tuple]:
        src = self.net.get(src_host)
        dst = self.net.get(dst_host)

        if not src or not dst:
            logger.error(f"Host {src_host} hoặc {dst_host} không tồn tại")
            return None

        logger.info(f"Starting iperf: {src_host} -> {dst_host} ({bandwidth}, {duration}s, {protocol})")

        protocol_flag = '-u' if protocol.upper() == 'UDP' else ''
        server_cmd = f'iperf -s {protocol_flag} -p 5001'
        server_proc = dst.popen(server_cmd)
        time.sleep(0.5)  # Chờ server sẵn sàng

        # BUG CŨ: bandwidth_flag chỉ set cho UDP nhưng TCP cũng có thể cần giới hạn bw.
        # FIX: Luôn truyền -b nếu có bandwidth, không phân biệt protocol.
        client_cmd = f'iperf -c {dst.IP()} -b {bandwidth} -t {duration} -p 5001 {protocol_flag}'
        client_proc = src.popen(client_cmd)

        self.flow_metadata.append(('iperf', src_host, dst_host, protocol, bandwidth, duration))
        self.active_processes.extend([server_proc, client_proc])
        return server_proc, client_proc

    def add_ping_flow(self, src_host: str, dst_host: str, count: int = 4) -> Optional[object]:
        src = self.net.get(src_host)
        dst = self.net.get(dst_host)

        if not src or not dst:
            logger.error(f"Host {src_host} hoặc {dst_host} không tồn tại")
            return None

        logger.info(f"Pinging: {src_host} -> {dst_host} ({count} packets)")
        ping_proc = src.popen(f'ping -c {count} {dst.IP()}')

        self.flow_metadata.append(('ping', src_host, dst_host, count))
        self.active_processes.append(ping_proc)
        return ping_proc

    def add_burst_flow(self, src_host: str, dst_host: str, bandwidth: str = '10M', duration: int = 5) -> Optional[Tuple]:
        """
        Sinh burst traffic ngắn với bandwidth cao — dùng cho Kịch Bản 1 (Burst Traffic).
        """
        logger.info(f"Burst traffic: {src_host} -> {dst_host} ({bandwidth} trong {duration}s)")
        return self.add_iperf_flow(src_host, dst_host, duration=duration, bandwidth=bandwidth, protocol='UDP')

    def add_poisson_flow(self, src_host: str, dst_host: str, base_bw: str = '1M', duration: int = 30) -> Optional[Tuple]:
        """
        Giả lập traffic theo phân phối Poisson bằng cách random interval giữa các burst —
        dùng cho Kịch Bản 4 (Random Traffic).
        """
        import threading
        import random

        def _poisson_sender():
            src = self.net.get(src_host)
            dst = self.net.get(dst_host)
            end_time = time.time() + duration
            while time.time() < end_time:
                interval = random.expovariate(1.0)  # mean=1s
                time.sleep(min(interval, 3.0))
                burst_bw = f'{random.randint(1, 8)}M'
                src.popen(f'iperf -c {dst.IP()} -b {burst_bw} -t 1 -p 5002 -u')

        t = threading.Thread(target=_poisson_sender, daemon=True)
        t.start()
        logger.info(f"Poisson traffic thread started: {src_host} -> {dst_host}")
        return t

    def wait_for_completion(self, timeout: int = 300) -> None:
        """
        Chờ tất cả processes kết thúc.

        BUG CŨ: wait_for_completion() không gọi process.wait() thực sự mà chỉ sleep
        theo duration trong metadata → không biết process đã xong chưa,
        và lặp sleep hai lần (trong while và sau while).
        FIX: Gọi proc.wait() với timeout cho từng process.
        """
        logger.info(f"Chờ {len(self.active_processes)} processes hoàn tất (timeout={timeout}s)...")
        deadline = time.time() + timeout

        for proc in self.active_processes:
            remaining = deadline - time.time()
            if remaining <= 0:
                logger.warning("Timeout! Còn processes chưa xong.")
                break
            try:
                proc.wait(timeout=remaining)
            except Exception:
                logger.warning(f"Process timeout hoặc đã kết thúc: {proc}")

        logger.info("Tất cả flows đã hoàn tất.")

    def kill_all(self) -> None:
        """Dừng tất cả flows đang chạy."""
        for proc in self.active_processes:
            try:
                proc.kill()
            except Exception:
                pass
        self.active_processes.clear()
        self.flow_metadata.clear()
        logger.info("Đã dừng tất cả traffic flows.")

    def start_constant_load(
        self,
        src_host: str,
        dst_host: str,
        bandwidth: str = '1M',
        protocol: str = 'TCP'
    ) -> Optional[Tuple]:
        """Traffic nền liên tục — dùng làm baseline load trước khi burst."""
        return self.add_iperf_flow(src_host, dst_host, duration=300, bandwidth=bandwidth, protocol=protocol)

    def get_host_bandwidth(self, host_name: str) -> Optional[str]:
        host = self.net.get(host_name)
        if not host:
            logger.error(f"Host {host_name} không tồn tại")
            return None
        output = host.cmd("cat /sys/class/net/eth0/speed").strip()
        return output


# BUG CŨ: generate_traffic_scenario() là method của class nhưng không có self,
# và if __name__ == '__main__' nằm bên trong class → không bao giờ chạy được.
# FIX: Chuyển ra ngoài class thành standalone function + __main__ đúng vị trí.

def generate_traffic_scenario(net: Mininet, scenario: str = 'basic') -> TrafficGenerator:
    """
    Factory function tạo TrafficGenerator với kịch bản có sẵn.

    Scenarios:
        basic   — TCP đơn giản + ping
        mixed   — UDP + TCP song song
        burst   — Kịch bản 1: traffic thấp rồi burst mạnh
        poisson — Kịch bản 4: traffic ngẫu nhiên Poisson
    """
    traffic_gen = TrafficGenerator(net)

    hosts = [h.name for h in net.hosts]
    if len(hosts) < 2:
        logger.error("Cần ít nhất 2 hosts để sinh traffic")
        return traffic_gen

    h1, h2 = hosts[0], hosts[1]

    if scenario == 'basic':
        traffic_gen.add_iperf_flow(h1, h2, duration=10, bandwidth='1M', protocol='TCP')
        traffic_gen.add_ping_flow(h1, h2, count=5)

    elif scenario == 'mixed':
        traffic_gen.add_iperf_flow(h1, h2, duration=20, bandwidth='5M', protocol='UDP')
        traffic_gen.add_iperf_flow(h2, h1, duration=20, bandwidth='3M', protocol='TCP')
        traffic_gen.add_ping_flow(h1, h2, count=10)

    elif scenario == 'burst':
        # Kịch bản 1: load nền thấp, sau đó burst
        traffic_gen.add_iperf_flow(h1, h2, duration=60, bandwidth='1M', protocol='TCP')
        time.sleep(10)  # load nền 10 giây
        traffic_gen.add_burst_flow(h1, h2, bandwidth='10M', duration=10)

    elif scenario == 'poisson':
        # Kịch bản 4: traffic ngẫu nhiên
        for i in range(0, len(hosts) - 1, 2):
            traffic_gen.add_poisson_flow(hosts[i], hosts[i + 1], base_bw='2M', duration=60)

    else:
        logger.warning(f"Không nhận diện được scenario '{scenario}', dùng 'basic'")
        traffic_gen.add_iperf_flow(h1, h2, duration=10, bandwidth='1M', protocol='TCP')
        traffic_gen.add_ping_flow(h1, h2, count=5)

    return traffic_gen


if __name__ == '__main__':
    from mininet.net import Mininet
    from mininet.topo import SingleSwitchTopo
    from mininet.node import RemoteController

    topo = SingleSwitchTopo(k=2)
    net = Mininet(topo=topo, controller=RemoteController)
    net.start()

    traffic_gen = generate_traffic_scenario(net, scenario='basic')
    traffic_gen.wait_for_completion(timeout=60)

    net.stop()