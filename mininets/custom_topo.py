from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.link import TCLink
from mininet.topo import Topo
from mininet.cli import CLI
from mininet.util import dumpNodeConnections
import logging
import sys

logger = logging.getLogger(__name__)


class TreeTopology(Topo):
    def __init__(self, depth=2, fanout=2):
        super(TreeTopology, self).__init__()
        self.depth = depth
        self.fanout = fanout
        self.host_count = 0
        self.switch_count = 0
        self._create_tree(None, 0)

    def _create_tree(self, parent_switch, level):
        if level >= self.depth:
            return

        # BUG CŨ: num_switches = self.fanout ** level → tạo nhiều switch ngang hàng cùng lúc ở root,
        # phá vỡ cấu trúc tree (root phải chỉ có 1 switch).
        # FIX: Mỗi lần gọi đệ quy chỉ tạo đúng 1 switch tại level hiện tại.
        self.switch_count += 1
        switch = f"s{self.switch_count}"
        self.addSwitch(switch)

        if parent_switch:
            self.addLink(parent_switch, switch)

        if level == self.depth - 1:
            # Leaf switch → gắn host
            for j in range(self.fanout):
                self.host_count += 1
                host = f"h{self.host_count}"
                self.addHost(host)
                self.addLink(switch, host)
        else:
            # Internal switch → đệ quy tạo fanout con
            for j in range(self.fanout):
                self._create_tree(switch, level + 1)


class LinearTopology(Topo):
    def __init__(self, num_switches=3):
        super(LinearTopology, self).__init__()
        self.num_switches = num_switches

        self.addHost("h1")
        self.addHost("h2")

        previous_switch = None
        for i in range(1, num_switches + 1):
            switch = f"s{i}"
            self.addSwitch(switch)

            if previous_switch:
                self.addLink(previous_switch, switch)
            else:
                # Switch đầu tiên kết nối h1
                self.addLink('h1', switch)

            previous_switch = switch

        # BUG CŨ: addLink(previous_switch, "h1") → nối h1 hai lần (đầu và cuối chuỗi),
        # tạo loop, gây lỗi Spanning Tree trong OVS.
        # FIX: Switch cuối chỉ nối với h2.
        self.addLink(previous_switch, "h2")


def create_network(
    topo_type='tree',
    controller_ips=None,          # FIX: hỗ trợ cluster (list IP:port), thay vì chỉ 1 controller
    controller_ip='127.0.0.1',    # giữ lại để backward compatible
    controller_port=6633,
    depth=2,
    fanout=2,
    num_hosts=4,
    num_switches=3,
    link_bw=10,
    enable_cli=False
):
    """
    Tạo mạng Mininet và kết nối với cluster Ryu controllers.

    Args:
        controller_ips: List các tuple (ip, port) cho cluster.
                        Ví dụ: [('127.0.0.1', 6633), ('127.0.0.1', 6634), ('127.0.0.1', 6635)]
                        Nếu None thì dùng controller_ip + controller_port đơn.
    """
    logger.info(f"Creating {topo_type} topology...")

    if topo_type == 'tree':
        topology = TreeTopology(depth=depth, fanout=fanout)
    elif topo_type == 'linear':
        topology = LinearTopology(num_switches=num_switches)
    else:
        raise ValueError(f"Invalid topology type: {topo_type}. Chọn 'tree' hoặc 'linear'.")

    # FIX: hỗ trợ multi-controller cho cluster
    if controller_ips is None:
        controller_ips = [(controller_ip, controller_port)]

    controllers = []
    for idx, (ip, port) in enumerate(controller_ips):
        c = RemoteController(f'c{idx}', ip=ip, port=port, protocol='tcp')
        controllers.append(c)
        logger.info(f"  Controller c{idx}: {ip}:{port}")

    # BUG CŨ: config bandwidth TRƯỚC net.start() → net.links chưa có interface thực,
    # gọi link.intf1.config(bw=...) sẽ raise lỗi hoặc không có tác dụng.
    # FIX: truyền link_bw vào TCLink qua topo, hoặc config SAU net.start().
    net = Mininet(
        topo=topology,
        controller=controllers[0] if len(controllers) == 1 else None,
        switch=OVSSwitch,
        link=TCLink,
        autoSetMacs=True
    )

    # Thêm các controllers vào net nếu là cluster
    for c in controllers:
        net.addController(c)

    logger.info("Starting network...")
    net.start()

    # Config bandwidth SAU khi start (interfaces đã sẵn sàng)
    for link in net.links:
        try:
            link.intf1.config(bw=link_bw)
            link.intf2.config(bw=link_bw)
        except Exception as e:
            logger.warning(f"Không thể set bandwidth cho link {link}: {e}")

    logger.info("Dumping host connections")
    dumpNodeConnections(net.hosts)
    dumpNodeConnections(net.switches)

    if enable_cli:
        CLI(net)

    # BUG CŨ: net.stop() nằm SAU return → không bao giờ được gọi (dead code).
    # FIX: Bỏ net.stop() khỏi đây. Caller tự chịu trách nhiệm gọi net.stop().
    return net


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Mininet SDN Topology")
    parser.add_argument('--topo', default='tree', choices=['tree', 'linear'])
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--fanout', type=int, default=3)
    parser.add_argument('--switches', type=int, default=3)
    parser.add_argument('--bw', type=int, default=10)
    parser.add_argument('--controllers', default='127.0.0.1:6633,127.0.0.1:6634,127.0.0.1:6635',
                        help='Comma-separated list of controller IP:port')
    args = parser.parse_args()

    # Parse cluster controllers
    ctrl_list = []
    for entry in args.controllers.split(','):
        ip, port = entry.strip().split(':')
        ctrl_list.append((ip, int(port)))

    net = create_network(
        topo_type=args.topo,
        controller_ips=ctrl_list,
        depth=args.depth,
        fanout=args.fanout,
        num_switches=args.switches,
        link_bw=args.bw,
        enable_cli=True
    )
    net.stop()