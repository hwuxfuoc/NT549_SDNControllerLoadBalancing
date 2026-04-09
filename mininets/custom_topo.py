from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.link import TCLink
from mininet.topo import Topo
from mininet.cli import CLI
from mininet.util import dumpNodeConnections
import logging
import argparse

logger = logging.getLogger(__name__)

class TreeTopology(Topo):
    def __init__(self, depth=2, fanout=2):
        super().__init__()
        self.depth = depth
        self.fanout = fanout
        self.host_count = 0
        self.switch_count = 0
        self._create_tree(None, 0, "s")

    def _create_tree(self, parent, level, prefix):
        if level >= self.depth:
            return
        
        num_nodes = self.fanout ** level if level > 0 else 1
        
        for i in range(num_nodes):
            self.switch_count += 1
            switch_name = f"s{self.switch_count}"
            switch = self.addSwitch(switch_name)
            
            if parent:
                self.addLink(parent, switch)
            
            if level == self.depth - 1:
                for j in range(self.fanout):
                    self.host_count += 1
                    host_name = f"h{self.host_count}"
                    host = self.addHost(host_name)
                    self.addLink(switch, host)
            else:
                self._create_tree(switch, level + 1, prefix)


class LinearTopology(Topo):
    def __init__(self, num_switches=4):
        super().__init__()
        hosts = ['h1', 'h2']
        for h in hosts:
            self.addHost(h)
        
        prev = None
        for i in range(1, num_switches + 1):
            sw = f"s{i}"
            self.addSwitch(sw)
            if prev:
                self.addLink(prev, sw)
            else:
                self.addLink('h1', sw)
            prev = sw
        self.addLink(prev, 'h2')


def create_network(topo_type='tree', depth=2, fanout=3, num_switches=4,
                   controllers=None, link_bw=10, enable_cli=False):
    """
    controllers: list of tuples [(ip, port), (ip, port), ...]
    """
    logger.info(f"Creating {topo_type} topology...")

    if topo_type == 'tree':
        topology = TreeTopology(depth=depth, fanout=fanout)
    elif topo_type == 'linear':
        topology = LinearTopology(num_switches=num_switches)
    else:
        raise ValueError("topo_type must be 'tree' or 'linear'")

    if not controllers:
        controllers = [('127.0.0.1', 6633)] 

    net_controllers = []
    for i, (ip, port) in enumerate(controllers):
        c = RemoteController(f'c{i}', ip=ip, port=port, protocol='tcp')
        net_controllers.append(c)

    net = Mininet(topo=topology, 
                  controllers=net_controllers,
                  switch=OVSSwitch,
                  link=TCLink,
                  autoSetMacs=True,
                  autoStaticArp=True)

    for link in net.links:
        link.intf1.config(bw=link_bw)
        link.intf2.config(bw=link_bw)

    logger.info("Starting network...")
    net.start()

    logger.info("Network started with %d controllers and %d switches", 
                len(controllers), len(net.switches))
    dumpNodeConnections(net.hosts)

    if enable_cli:
        CLI(net)

    return net

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--topo', type=str, default='tree', choices=['tree', 'linear'])
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--fanout', type=int, default=3)
    parser.add_argument('--cli', action='store_true')
    args = parser.parse_args()

    controllers = [('127.0.0.1', 6633), ('127.0.0.1', 6634), ('127.0.0.1', 6635)]
    
    net = create_network(topo_type=args.topo, 
                         depth=args.depth, 
                         fanout=args.fanout,
                         controllers=controllers,
                         enable_cli=args.cli)
    
    if not args.cli:
        input("Press Enter to stop the network...")
        net.stop()