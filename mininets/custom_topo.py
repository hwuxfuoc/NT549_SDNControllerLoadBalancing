from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.link import TCLink
from mininet.topo import Topo
from mininet.cli import CLI
from mininet.util import dumpNodeConnections
import logging

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
        
        num_switches = self.fanout ** level if level > 0 else 1
        
        for i in range(num_switches):
            self.switch_count += 1
            switch = f"s{self.switch_count}"
            self.addSwitch(switch)
            
            if parent_switch:
                self.addLink(parent_switch, switch)
            
            if level == self.depth - 1:
                for j in range(self.fanout):
                    self.host_count += 1
                    host = f"h{self.host_count}"
                    self.addHost(host)
                    self.addLink(switch, host)
            else:
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
                self.addLink('h1', switch)

            previous_switch = switch
            
        self.addLink(previous_switch, "h1")
        self.addLink(previous_switch, "h2")

def create_network(topo_type='tree', controller_ip='127.0.0.1', controller_port=6633, depth=2, fanout=2, num_hosts=4, num_switches=3, link_bw=10, enable_cli=False):
    logger.info(f"Creating {topo_type} topology...")

    if topo_type == 'tree':
        fanout = max(1, int(num_hosts ** 0.5))
        if fanout * fanout < num_hosts:
            fanout += 1
        topology = TreeTopology(depth=2, fanout=fanout)
    elif topo_type == 'linear':
        num_switches = max(1, num_hosts - 1)
        topology = LinearTopology(num_switches=num_switches)
    else:
        raise ValueError("Invalid topology type")

    controller = RemoteController('c0', ip=controller_ip, port=controller_port, protocol='tcp')

    net = Mininet(topo=topology, controller=controller, switch=OVSSwitch, link=TCLink)
    
    for link in net.links:
        link.intf1.config(bw=link_bw)
        link.intf2.config(bw=link_bw)

    logger.info("Starting network...")
    net.start()

    logger.info("Dumping host connections")
    dumpNodeConnections(net.hosts)
    dumpNodeConnections(net.switches)

    if enable_cli:
        cli = CLI(net)
    
    return net
    net.stop()

if __name__ == '__main__':
    import sys
    topo = sys.argv[1] if len(sys.argv) > 1 else 'tree'
    net = create_network(topo_type=topo, enable_cli=True)
    net.stop()